// SGEMM: C (m x n) = A (m x k) * B (k x n), row-major, FP32.
//
// Two kernels:
//   matmul_naive : one thread per C element, all operands read from global memory
//   matmul_tiled : shared-memory tiling, boundary-safe for arbitrary m, k, n
//
// Both are validated against a CPU reference. Inputs are small integers,
// so every partial sum is exactly representable in FP32 and the comparison
// can be exact.
//
// Build:  nvcc -O3 -arch=sm_75 -DTILE=16 matmul.cu -o matmul
// Run:    ./matmul

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "../common/cuda_check.h"

#ifndef TILE
#define TILE 16
#endif

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

    struct { const char* name; MatmulKernel fn; } kernels[] = {
        {"naive", matmul_naive},
        {"tiled", matmul_tiled},
    };
    for (auto& kern : kernels) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        float ms = time_kernel(kern.fn, grid, block, dA, dB, dC, m, k, n, reps);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
        int bad = count_mismatches(hC, hRef, m * n);
        printf("GPU %-6s %s  %8.3f ms  %7.1f GFLOP/s\n", kern.name,
               bad == 0 ? "OK  " : "FAIL", ms, flop / (ms * 1e-3) / 1e9);
    }
    printf("\n");

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
