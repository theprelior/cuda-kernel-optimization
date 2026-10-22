// SGEMM: C (m x n) = A (m x k) * B (k x n), row-major, FP32.
//
// Two kernels plus a library baseline:
//   matmul_naive : one thread per C element, all operands read from global memory
//   matmul_tiled : shared-memory tiling, boundary-safe for arbitrary m, k, n
//   cuBLAS       : cublasSgemm, the vendor-optimized reference
//
// All are validated against a CPU reference. Inputs are small integers,
// so every partial sum is exactly representable in FP32 and the comparison
// can be exact.
//
// Build:  nvcc -O3 -arch=sm_75 -DTILE=16 matmul.cu -o matmul -lcublas
// Run:    ./matmul

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cublas_v2.h>
#include "../common/cuda_check.h"

#ifndef TILE
#define TILE 16
#endif

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st_ = (call);                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %d\n  at %s:%d\n",               \
                    (int)st_, __FILE__, __LINE__);                         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void matmul_cpu(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
}

// ---------------------------------------------------------------------------
// Naive kernel: each thread computes one C element.
// x maps to columns so that a warp reads consecutive elements of B (coalesced).
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) sum += A[row * k + p] * B[p * n + col];
        C[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Tiled kernel: the block cooperatively stages a TILE x TILE tile of A and B
// in shared memory, so each global element is loaded once per block instead
// of once per thread.
//
// Boundary handling: out-of-range loads write 0 into the tile (zero padding
// does not change the dot product). Every thread executes both
// __syncthreads() calls; only the global loads and the final store are
// guarded, which avoids barrier divergence.
// ---------------------------------------------------------------------------
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int m, int k, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int num_tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + tx;   // column of A loaded by this thread
        int b_row = t * TILE + ty;   // row of B loaded by this thread

        As[ty][tx] = (row < m && a_col < k)   ? A[row * k + a_col]   : 0.0f;
        Bs[ty][tx] = (b_row < k && col < n)   ? B[b_row * n + col]   : 0.0f;
        __syncthreads();             // tile fully loaded before anyone reads it

        for (int p = 0; p < TILE; p++)
            sum += As[ty][p] * Bs[p][tx];
        __syncthreads();             // everyone done reading before next overwrite
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

// ---------------------------------------------------------------------------
// 1D register tiling: each thread computes RT = 4 outputs in one column.
// Block = 32 x 8 threads, computes a 32 x 32 tile of C; k is processed in
// slices of 8. Per inner step a thread reads 1 value of Bs into a register
// and 4 values of As, and performs 4 FMAs (5 shared loads / 4 FMA instead of
// 2 / 1 in the tiled kernel).
//
// Each thread wears two hats:
//   loading   : moves exactly one element of A and one of B into shared memory
//   computing : accumulates its own 4 outputs in registers
// ---------------------------------------------------------------------------
#define RB_M 32   // rows of the C tile
#define RB_N 32   // columns of the C tile
#define RB_K 8    // depth of a k slice
#define RT   4    // outputs per thread

__global__ void matmul_reg(const float* A, const float* B, float* C,
                           int m, int k, int n) {
    __shared__ float As[RB_M][RB_K];
    __shared__ float Bs[RB_K][RB_N];

    int tx = threadIdx.x;               // 0..31
    int ty = threadIdx.y;               // 0..7
    int id = ty * RB_N + tx;            // 0..255, linear id used for loading

    // loading hat: which shared-memory slot do I fill?
    int aTileRow = id / RB_K, aTileCol = id % RB_K;
    int bTileRow = ty,        bTileCol = tx;

    // computing hat: which C elements are mine?
    int firstRow = blockIdx.y * RB_M + ty * RT;
    int col      = blockIdx.x * RB_N + tx;

    float acc[RT] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < (k + RB_K - 1) / RB_K; t++) {
        int aRow = blockIdx.y * RB_M + aTileRow;   // tile start + offset
        int aCol = t * RB_K + aTileCol;
        As[aTileRow][aTileCol] = (aRow < m && aCol < k) ? A[aRow * k + aCol] : 0.0f;

        int bRow = t * RB_K + bTileRow;
        int bCol = blockIdx.x * RB_N + bTileCol;
        Bs[bTileRow][bTileCol] = (bRow < k && bCol < n) ? B[bRow * n + bCol] : 0.0f;

        __syncthreads();

        for (int p = 0; p < RB_K; p++) {
            float b = Bs[p][tx];                   // one shared load, kept in a register
            for (int i = 0; i < RT; i++)
                acc[i] += As[ty * RT + i][p] * b;  // reused RT times
        }
        __syncthreads();
    }

    for (int i = 0; i < RT; i++) {
        int row = firstRow + i;
        if (row < m && col < n) C[row * n + col] = acc[i];
    }
}

// ---------------------------------------------------------------------------
// Timing helper: one warm-up launch, then the average of `reps` launches.
// ---------------------------------------------------------------------------
typedef void (*MatmulKernel)(const float*, const float*, float*, int, int, int);

float time_kernel(MatmulKernel kernel, dim3 grid, dim3 block,
                  const float* dA, const float* dB, float* dC,
                  int m, int k, int n, int reps) {
    kernel<<<grid, block>>>(dA, dB, dC, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        kernel<<<grid, block>>>(dA, dB, dC, m, k, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

// ---------------------------------------------------------------------------
// cuBLAS baseline.
// cuBLAS assumes column-major storage. A row-major matrix read as column-major
// is its transpose, so C = A*B (row-major) is computed as C^T = B^T * A^T
// (column-major): pass B first, then A, with dimensions (n, m, k).
// No data is copied or transposed; only the interpretation changes.
// ---------------------------------------------------------------------------
void matmul_cublas(cublasHandle_t handle, const float* dA, const float* dB,
                   float* dC, int m, int k, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             dB, n,     // B^T : n x k, leading dimension n
                             dA, k,     // A^T : k x m, leading dimension k
                             &beta,
                             dC, n));   // C^T : n x m, leading dimension n
}

float time_cublas(cublasHandle_t handle, const float* dA, const float* dB,
                  float* dC, int m, int k, int n, int reps) {
    matmul_cublas(handle, dA, dB, dC, m, k, n);   // warm-up (also lazy init)
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        matmul_cublas(handle, dA, dB, dC, m, k, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

int count_mismatches(const float* a, const float* b, int len) {
    int bad = 0;
    for (int i = 0; i < len; i++) if (a[i] != b[i]) bad++;
    return bad;
}

// ---------------------------------------------------------------------------
// Run one problem size: CPU reference, both kernels, validation, timing.
// ---------------------------------------------------------------------------
void run(int m, int k, int n) {
    size_t bytesA = (size_t)m * k * sizeof(float);
    size_t bytesB = (size_t)k * n * sizeof(float);
    size_t bytesC = (size_t)m * n * sizeof(float);

    float* hA   = (float*)malloc(bytesA);
    float* hB   = (float*)malloc(bytesB);
    float* hC   = (float*)malloc(bytesC);
    float* hRef = (float*)malloc(bytesC);
    for (int i = 0; i < m * k; i++) hA[i] = (float)(i % 5);
    for (int i = 0; i < k * n; i++) hB[i] = (float)(i % 7);

    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(hA, hB, hRef, m, k, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE,    // x -> columns
              (m + TILE - 1) / TILE);   // y -> rows

    const int reps = 10;
    double flop = 2.0 * m * n * k;

    printf("===== m=%d k=%d n=%d  (block %dx%d) =====\n", m, k, n, TILE, TILE);
    printf("CPU (1 thread, naive): %10.3f ms\n", cpu_ms);

    dim3 block_reg(RB_N, RB_M / RT);              // 32 x 8 threads
    dim3 grid_reg((n + RB_N - 1) / RB_N, (m + RB_M - 1) / RB_M);

    struct { const char* name; MatmulKernel fn; dim3 g; dim3 b; } kernels[] = {
        {"naive", matmul_naive, grid,     block},
        {"tiled", matmul_tiled, grid,     block},
        {"reg1d", matmul_reg,   grid_reg, block_reg},
    };
    float tiled_ms = 0.0f;
    float reg_ms = 0.0f;
    for (auto& kern : kernels) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        float ms = time_kernel(kern.fn, kern.g, kern.b, dA, dB, dC, m, k, n, reps);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
        int bad = count_mismatches(hC, hRef, m * n);
        printf("GPU %-6s %s  %8.3f ms  %7.1f GFLOP/s\n", kern.name,
               bad == 0 ? "OK  " : "FAIL", ms, flop / (ms * 1e-3) / 1e9);
        if (kern.fn == matmul_tiled) tiled_ms = ms;
        if (kern.fn == matmul_reg)   reg_ms = ms;
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    float cublas_ms = time_cublas(handle, dA, dB, dC, m, k, n, reps);
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    int bad = count_mismatches(hC, hRef, m * n);
    printf("cuBLAS     %s  %8.3f ms  %7.1f GFLOP/s\n",
           bad == 0 ? "OK  " : "FAIL", cublas_ms, flop / (cublas_ms * 1e-3) / 1e9);
    printf("tiled reaches %.0f%% of cuBLAS, reg1d reaches %.0f%%\n\n",
           100.0 * cublas_ms / tiled_ms, 100.0 * cublas_ms / reg_ms);
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hRef);
}

int main() {
    run(777, 513, 300);     // correctness: non-square, not a multiple of TILE
    run(1024, 1024, 1024);  // performance
    return 0;
}
