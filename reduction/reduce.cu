// Parallel sum reduction of an FP32 array: five versions, each fixing one
// bottleneck of the previous one.
//
//   v1  interleaved addressing    (tid % (2*stride) -> warp divergence)
//   v2  sequential addressing     (tid < stride     -> active threads packed
//                                                      into whole warps)
//   v3  first add during load     (each thread loads 2 elements; half the blocks)
//   v4  warp-shuffle final stage  (last 32 lanes reduce in registers with
//                                  __shfl_down_sync, no shared memory / barriers)
//   v5  v4 + atomicAdd            (whole reduction finishes on the GPU)
//
// v1-v4 write one partial sum per block; the host adds the partials in double.
// Inputs are small integers so the result can be compared exactly.
//
// Build:  nvcc -O3 -arch=sm_75 reduce.cu -o reduce
// Run:    ./reduce

#include <cstdio>
#include <cstdlib>
#include "../common/cuda_check.h"

#define BLOCK 256   // must be a power of two >= 64 (v4/v5 assume this)

// --------------------------- v1: interleaved -------------------------------
__global__ void reduce_v1(const float* in, float* partial, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          // active threads are scattered
            s[tid] += s[tid + stride];        // across all warps -> divergence
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// --------------------------- v2: sequential --------------------------------
__global__ void reduce_v2(const float* in, float* partial, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)                     // active threads are contiguous
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// --------------------------- v3: add during load ---------------------------
__global__ void reduce_v3(const float* in, float* partial, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    // Each element gets its own bounds check, so a partially filled last
    // block still contributes its valid elements.
    float a = (i < n)              ? in[i]              : 0.0f;
    float b = (i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f;
    s[tid] = a + b;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// Shared body of v4/v5: returns the block sum in lane 0 of warp 0.
__device__ float block_sum_v4(const float* in, int n, float* s) {
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    float a = (i < n)              ? in[i]              : 0.0f;
    float b = (i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f;
    s[tid] = a + b;
    __syncthreads();

    // Shared-memory stages while more than one warp is active.
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride)
            s[tid] += s[tid + stride];
        __syncthreads();
    }

    // Last warp: stride-32 step from shared memory, then register shuffles.
    float v = 0.0f;
    if (tid < 32) {
        v = s[tid] + s[tid + 32];
        for (int offset = 16; offset > 0; offset /= 2)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);   // all 32 lanes participate
    }
    return v;
}

// --------------------------- v4: warp shuffle ------------------------------
__global__ void reduce_v4(const float* in, float* partial, int n) {
    __shared__ float s[BLOCK];
    float v = block_sum_v4(in, n, s);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// --------------------------- v5: atomic finish -----------------------------
// One atomicAdd per block (not per element) keeps contention low.
// Note: FP32 addition is not associative, so for large inputs the result may
// differ in the last bits between runs because block order is not fixed.
__global__ void reduce_v5(const float* in, float* total, int n) {
    __shared__ float s[BLOCK];
    float v = block_sum_v4(in, n, s);
    if (threadIdx.x == 0) atomicAdd(total, v);
}

// ---------------------------------------------------------------------------
typedef void (*ReduceKernel)(const float*, float*, int);

// Warm-up + average of `reps` launches; sums the per-block partials on the host.
float time_reduce(ReduceKernel kernel, int blocks, const float* d_in,
                  float* d_partial, float* h_partial, int n, double* result) {
    kernel<<<blocks, BLOCK>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int reps = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        kernel<<<blocks, BLOCK>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int b = 0; b < blocks; b++) sum += h_partial[b];
    *result = sum;
    return ms / reps;
}

void run(int n) {
    int blocks_1x = (n + BLOCK - 1) / BLOCK;               // v1, v2: 1 element/thread
    int blocks_2x = (n + 2 * BLOCK - 1) / (2 * BLOCK);     // v3-v5: 2 elements/thread

    float* h_in      = (float*)malloc((size_t)n * sizeof(float));
    float* h_partial = (float*)malloc(blocks_1x * sizeof(float));   // larger of the two
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 3);

    double expected = 0.0;
    for (int i = 0; i < n; i++) expected += h_in[i];

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, blocks_1x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    struct { const char* name; ReduceKernel fn; int blocks; } versions[] = {
        {"v1 interleaved     ", reduce_v1, blocks_1x},
        {"v2 sequential      ", reduce_v2, blocks_1x},
        {"v3 add-during-load ", reduce_v3, blocks_2x},
        {"v4 warp shuffle    ", reduce_v4, blocks_2x},
    };

    printf("===== n = %d =====\n", n);
    printf("expected sum: %.0f\n", expected);
    float prev_ms = 0.0f;
    for (int v = 0; v < 4; v++) {
        double result;
        float ms = time_reduce(versions[v].fn, versions[v].blocks,
                               d_in, d_partial, h_partial, n, &result);
        double gbs = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("%s %s %8.3f ms  %6.1f GB/s", versions[v].name,
               result == expected ? "OK  " : "FAIL", ms, gbs);
        if (v > 0) printf("  (%.2fx vs previous)", prev_ms / ms);
        printf("\n");
        prev_ms = ms;
    }

    // v5: the accumulator must be zeroed before every launch.
    float* d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(float)));
    reduce_v5<<<blocks_2x, BLOCK>>>(d_in, d_total, n);
    CUDA_CHECK(cudaGetLastError());
    float h_total;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    printf("v5 atomic finish    %s result %.0f\n\n",
           (double)h_total == expected ? "OK  " : "FAIL", h_total);
    CUDA_CHECK(cudaFree(d_total));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in); free(h_partial);
}

int main() {
    run(1000003);   // correctness: last block only partially filled
    run(1 << 24);   // performance (~67 MB of input)
    return 0;
}
