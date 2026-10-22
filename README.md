# CUDA Kernel Optimization

Step-by-step optimization of three classic GPU kernels — **SGEMM**, **parallel
reduction** and **histogram** — written in CUDA C++, each version validated
against a CPU reference and measured on an NVIDIA Tesla T4, with cuBLAS as the
SGEMM baseline.

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

**A note on variance.** The T4 is a 70 W, power-limited card and Colab GPUs are
shared. The same binary has shown up to ~2× different absolute timings across
sessions (clock boost vs. sustained clocks). All numbers below come from a single
session so they are comparable with each other; the most robust figure is the
ratio to cuBLAS measured in the same run, since both move together with the clock.

## Results

### 1. SGEMM — `matmul/`

C = A·B, FP32, 1024 × 1024 × 1024 (2.15 GFLOP)

| Kernel | Block | Time | Throughput | vs cuBLAS |
|---|---|---|---|---|
| CPU, 1 thread, naive triple loop | – | 3.3 s | 0.65 GFLOP/s | – |
| Naive (one thread per output) | 16×16 | 9.17 ms | 234 GFLOP/s | 9% |
| Shared-memory tiled | 16×16 | 5.79 ms | 371 GFLOP/s | 14% |
| Naive | 32×32 | 6.73 ms | 319 GFLOP/s | 12% |
| Shared-memory tiled | 32×32 | **5.31 ms** | **405 GFLOP/s** | **16%** |
| **cuBLAS** `cublasSgemm` | – | 0.83 ms | 2594 GFLOP/s | 100% |

Techniques:
- `x` mapped to columns so each warp reads consecutive elements of B (coalesced access)
- Shared-memory tiling: each global element is loaded once per block instead of once per thread
- Boundary handling for arbitrary m, k, n: zero padding of partial tiles; every thread
  reaches both `__syncthreads()` barriers, only global loads/stores are guarded
- cuBLAS called on row-major data without copies: cuBLAS is column-major, so
  C = A·B is computed as Cᵀ = Bᵀ·Aᵀ by passing B before A

Observations:
- 32×32 blocks help the **naive** kernel too (9.17 → 6.73 ms): each warp then covers
  exactly one row of 32 consecutive columns, so its loads from B are fully contiguous.
- The remaining ~6× gap to cuBLAS is what register tiling (several outputs per thread,
  reusing shared-memory values from registers), vectorized `float4` loads and
  double buffering close. After tiling, shared-memory bandwidth — two shared loads
  per FMA — is the next bottleneck.

### 2. Parallel reduction — `reduction/`

Sum of 2²⁴ FP32 values (67 MB)

| Version | Change | Time | Bandwidth | Step speedup |
|---|---|---|---|---|
| v1 | interleaved addressing (`tid % (2*s)`) | 1.294 ms | 52 GB/s | – |
| v2 | sequential addressing (`tid < s`) — removes warp divergence | 0.800 ms | 84 GB/s | 1.62× |
| v3 | first add during load — no idle threads in the first step, half the blocks | 0.419 ms | 160 GB/s | 1.91× |
| v4 | last warp reduced with `__shfl_down_sync` — no shared memory or barriers | **0.272 ms** | **247 GB/s** | 1.54× |
| v5 | v4 + one `atomicAdd` per block — whole sum stays on the GPU | – | – | – |

**v1 → v4: 4.8× faster, ~77% of the T4's peak memory bandwidth.** Reduction is
memory-bound (one add per 4 bytes read), so bandwidth, not FLOP/s, is the right metric.

Bug caught by the odd-size test: an early v3 checked both loaded elements with one
combined condition, which silently dropped the valid elements of the last, partially
filled block. It passed for n = 2²⁴ and failed for n = 1,000,003.

### 3. Histogram — `histogram/`

256 bins over 2²⁴ bytes. Grid version uses a fixed grid of 1024 × 256 threads (64 elements per thread).

| Version | A: uniform data | B: all values identical |
|---|---|---|
| naive (global `atomicAdd` per element) | 8.19 ms | 16.51 ms |
| shared-memory privatization | 1.13 ms | 0.88 ms |
| shared + grid-stride loop | **0.212 ms** (79 GB/s) | **0.607 ms** (28 GB/s) |
| **total speedup** | **39×** | **27×** |

Observations:
- Atomic performance depends on the **data distribution**: identical values make all
  threads contend for one counter. In the naive kernel the single-value input is 2× slower.
- Privatization gave 7× even on uniform data although the number of global atomics stays
  the same at one element per thread: flush atomics hit 32 distinct, consecutive bins per
  warp instead of random, colliding addresses.
- After the grid-stride change the bottleneck moved: with uniform data global traffic is
  amortized, but with single-value data all threads of a block still contend on one
  **shared-memory** counter, so B is now ~3× slower than A.

## Build and run

```bash
make            # ARCH=sm_75 by default; e.g. make ARCH=sm_80 for A100
make run
make clean && make TILE=32 run    # SGEMM with 32x32 tiles
```

Find your GPU's architecture with `nvidia-smi --query-gpu=name,compute_cap --format=csv`
(compute capability 8.0 → `ARCH=sm_80`).

In Google Colab (Runtime → T4 GPU):

```bash
!git clone https://github.com/theprelior/cuda-kernel-optimization.git
%cd cuda-kernel-optimization
!make run
```

## Repository layout

```
common/cuda_check.h      error-checking macro for CUDA API calls
matmul/matmul.cu         CPU reference, naive and tiled SGEMM, cuBLAS baseline
reduction/reduce.cu      reduction v1–v5
histogram/hist.cu        naive, shared, grid-stride histogram
```

## Next steps

- Register tiling and vectorized loads to close the gap to cuBLAS
- Profile with Nsight Compute (memory throughput, occupancy, bank conflicts)
- Run the suite on other GPUs (L4, A100) and compare
- Mixed precision: FP16/bf16 inputs with FP32 accumulation, Tensor Cores (WMMA)
