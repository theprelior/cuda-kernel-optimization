# CUDA Kernel Optimization

Step-by-step optimization of three classic GPU kernels — **SGEMM**, **parallel
reduction** and **histogram** — written in CUDA C++, each version validated
against a CPU reference and measured on an NVIDIA Tesla T4.

The goal is not only the fastest kernel but understanding *why* each step is
faster: every version targets one specific bottleneck (memory traffic, warp
divergence, idle threads, synchronization cost, atomic contention).

## Hardware and methodology

- **GPU:** NVIDIA Tesla T4 (Turing, `sm_75`, ~320 GB/s peak memory bandwidth), Google Colab
- **CUDA:** 13.0
- **Timing:** `cudaEvent`s, one warm-up launch, average of 10 launches
- **Validation:** every kernel compared against a CPU reference. Inputs are small
  integers so FP32 results are exact and can be compared bit for bit.
- **Correctness tests on awkward sizes:** non-square matrices and array sizes that
  are not a multiple of the block size (e.g. 777×513×300, n = 1,000,003) to exercise
  boundary handling.

Absolute timings vary by a few percent between Colab sessions (shared GPU,
clock boost); ratios within one run are stable.

## Results

### 1. SGEMM — `matmul/`

C = A·B, FP32, 1024 × 1024 × 1024 (2.15 GFLOP)

| Kernel | Block | Time | Throughput |
|---|---|---|---|
| CPU, 1 thread, naive triple loop | – | ~7–8 s | – |
| Naive (one thread per output) | 16×16 | 6.44 ms | 333 GFLOP/s |
| Shared-memory tiled | 16×16 | 4.07 ms | 528 GFLOP/s |
| Naive | 32×32 | 3.72 ms | 578 GFLOP/s |
| Shared-memory tiled | 32×32 | **2.85 ms** | **754 GFLOP/s** |

Techniques:
- `x` mapped to columns so each warp reads consecutive elements of B (coalesced access)
- Shared-memory tiling: each global element is loaded once per block instead of once per thread
- Boundary handling for arbitrary m, k, n: zero padding of partial tiles; every thread
  reaches both `__syncthreads()` barriers, only global loads/stores are guarded

Observation: the jump from 16×16 to 32×32 helps the **naive** kernel too. With
32-wide blocks each warp covers exactly one row of 32 consecutive columns, so its
loads from B are fully contiguous. The tiled kernel adds a further ~1.3× on top.

### 2. Parallel reduction — `reduction/`

Sum of 2²⁴ FP32 values (67 MB)

| Version | Change | Time | Bandwidth | Step speedup |
|---|---|---|---|---|
| v1 | interleaved addressing (`tid % (2*s)`) | 1.431 ms | 47 GB/s | – |
| v2 | sequential addressing (`tid < s`) — removes warp divergence | 0.884 ms | 76 GB/s | 1.62× |
| v3 | first add during load — no idle threads in the first step, half the blocks | 0.463 ms | 145 GB/s | 1.91× |
| v4 | last warp reduced with `__shfl_down_sync` — no shared memory or barriers | **0.291 ms** | **231 GB/s** | 1.59× |
| v5 | v4 + one `atomicAdd` per block — whole sum stays on the GPU | – | – | – |

**v1 → v4: 4.9× faster, ~72% of the T4's peak memory bandwidth.** Reduction is
memory-bound (one add per 4 bytes read), so bandwidth, not FLOP/s, is the right metric.

Bug caught by the odd-size test: an early v3 checked both loaded elements with one
combined condition, which silently dropped the valid elements of the last, partially
filled block. It passed for n = 2²⁴ and failed for n = 1,000,003.

### 3. Histogram — `histogram/`

256 bins over 2²⁴ bytes. Grid version uses a fixed grid of 1024 × 256 threads (64 elements per thread).

| Version | A: uniform data | B: all values identical |
|---|---|---|
| naive (global `atomicAdd` per element) | 9.08 ms | 13.92 ms |
| shared-memory privatization | 1.23 ms | 0.88 ms |
| shared + grid-stride loop | **0.224 ms** (75 GB/s) | **0.599 ms** |
| **total speedup** | **40×** | **23×** |

Observations:
- Atomic performance depends on the **data distribution**: identical values make all
  threads contend for one counter. In the naive kernel the single-value input is ~1.5×
  slower than uniform data.
- Privatization gave 7× even on uniform data although the number of global atomics stays
  the same at one element per thread: flush atomics hit 32 distinct, consecutive bins per
  warp instead of random, colliding addresses.
- After the grid-stride change the bottleneck moved: with uniform data global traffic is
  amortized, but with single-value data all threads of a block still contend on one
  **shared-memory** counter, so B is now slower than A.

## Build and run

```bash
make            # ARCH=sm_75 by default; e.g. make ARCH=sm_80 for A100
make run
make TILE=32    # rebuild SGEMM with 32x32 tiles
```

In Google Colab (Runtime → T4 GPU):

```bash
!git clone https://github.com/theprelior/cuda-kernel-optimization.git
%cd cuda-kernel-optimization
!make run
```

## Repository layout

```
common/cuda_check.h      error-checking macro for CUDA API calls
matmul/matmul.cu         CPU reference, naive and tiled SGEMM
reduction/reduce.cu      reduction v1–v5
histogram/hist.cu        naive, shared, grid-stride histogram
```

## Next steps

- Compare SGEMM against cuBLAS
- Profile with Nsight Compute (memory throughput, occupancy, bank conflicts)
- Register tiling (each thread computes several outputs) and vectorized loads
- Mixed precision: FP16/bf16 inputs with FP32 accumulation, Tensor Cores (WMMA)
