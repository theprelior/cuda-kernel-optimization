// 256-bin histogram of an 8-bit input array: three versions.
//
//   naive  : one global atomicAdd per element
//   shared : per-block histogram in shared memory (privatization), then one
//            global atomicAdd per bin per block
//   grid   : shared + grid-stride loop, so a fixed, small grid processes the
//            whole input and each block amortizes its setup/flush cost over
//            many elements
//
// Two input distributions are tested:
//   A: values spread uniformly over 0..255
//   B: all values identical (worst case for atomic contention)
//
// Build:  nvcc -O3 -arch=sm_75 hist.cu -o hist
// Run:    ./hist

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../common/cuda_check.h"

#define BINS  256
#define BLOCK 256          // == BINS: each thread owns one bin when zeroing/flushing
#define GRID_BLOCKS 1024   // fixed grid size for the grid-stride version

static_assert(BLOCK == BINS, "kernels assume one thread per bin");

// ------------------------------- naive -------------------------------------
__global__ void hist_naive(const unsigned char* in, unsigned int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[in[i]], 1u);
}

// ------------------------------- shared ------------------------------------
__global__ void hist_shared(const unsigned char* in, unsigned int* hist, int n) {
    __shared__ unsigned int local[BINS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    local[tid] = 0;                          // thread tid owns bin tid
    __syncthreads();                         // all bins zeroed before counting

    if (i < n)
        atomicAdd(&local[in[i]], 1u);        // bin chosen by the VALUE
    __syncthreads();                         // all counting done before flush

    atomicAdd(&hist[tid], local[tid]);       // flush own bin to global
}

// --------------------------- shared + grid-stride --------------------------
__global__ void hist_grid(const unsigned char* in, unsigned int* hist, int n) {
    __shared__ unsigned int local[BINS];
    int tid = threadIdx.x;

    local[tid] = 0;
    __syncthreads();

    // Consecutive threads read consecutive bytes in every iteration (coalesced);
    // the whole grid then advances by its total thread count.
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&local[in[i]], 1u);
    __syncthreads();

    atomicAdd(&hist[tid], local[tid]);
}

// ---------------------------------------------------------------------------
typedef void (*HistKernel)(const unsigned char*, unsigned int*, int);

void measure(const char* name, HistKernel kernel, int blocks,
             const unsigned char* d_in, unsigned int* d_hist,
             const unsigned int* ref, int n) {
    // warm-up
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
    kernel<<<blocks, BLOCK>>>(d_in, d_hist, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int reps = 10;
    float total_ms = 0.0f;
    for (int r = 0; r < reps; r++) {
        // atomics accumulate, so the histogram is reset before every run
        // (outside the timed region)
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<blocks, BLOCK>>>(d_in, d_hist, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    unsigned int h_hist[BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    int bad_bins = 0;
    for (int b = 0; b < BINS; b++) if (h_hist[b] != ref[b]) bad_bins++;

    float ms = total_ms / reps;
    printf("  %-7s %s %8.3f ms  %6.1f GB/s\n", name,
           bad_bins == 0 ? "OK  " : "FAIL", ms, (double)n / (ms * 1e-3) / 1e9);
}

void run_dataset(const char* title, const unsigned char* h_in, int n) {
    unsigned int ref[BINS];
    memset(ref, 0, sizeof(ref));
    for (int i = 0; i < n; i++) ref[h_in[i]]++;

    unsigned char* d_in;
    unsigned int*  d_hist;
    CUDA_CHECK(cudaMalloc(&d_in, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n, cudaMemcpyHostToDevice));

    int blocks_full = (n + BLOCK - 1) / BLOCK;   // one thread per element

    printf("%s\n", title);
    measure("naive",  hist_naive,  blocks_full, d_in, d_hist, ref, n);
    measure("shared", hist_shared, blocks_full, d_in, d_hist, ref, n);
    measure("grid",   hist_grid,   GRID_BLOCKS, d_in, d_hist, ref, n);
    printf("\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_hist));
}

int main() {
    int n = 1 << 24;
    unsigned char* h_in = (unsigned char*)malloc(n);

    // A: pseudo-random, roughly uniform over 0..255 (multiplicative hash)
    for (int i = 0; i < n; i++)
        h_in[i] = (unsigned char)(((unsigned int)i * 2654435761u) >> 24);
    run_dataset("Dataset A (uniform)", h_in, n);

    // B: all values identical (e.g. an all-black image)
    for (int i = 0; i < n; i++) h_in[i] = 7;
    run_dataset("Dataset B (single value)", h_in, n);

    free(h_in);
    return 0;
}
