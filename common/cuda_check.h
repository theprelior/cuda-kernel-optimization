#pragma once
#include <cstdio>
#include <cstdlib>

// Wraps a CUDA API call and aborts with file/line information on failure.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d\n",                \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
