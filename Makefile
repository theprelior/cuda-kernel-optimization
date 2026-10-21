NVCC  ?= nvcc
# sm_75 = T4 (Turing). Use sm_80 for A100, sm_86 for RTX 30xx, sm_89 for RTX 40xx / L4
ARCH  ?= sm_75
TILE  ?= 16
FLAGS  = -O3 -arch=$(ARCH)

all: matmul/matmul reduction/reduce histogram/hist

matmul/matmul: matmul/matmul.cu common/cuda_check.h
	$(NVCC) $(FLAGS) -DTILE=$(TILE) $< -o $@

reduction/reduce: reduction/reduce.cu common/cuda_check.h
	$(NVCC) $(FLAGS) $< -o $@

histogram/hist: histogram/hist.cu common/cuda_check.h
	$(NVCC) $(FLAGS) $< -o $@

run: all
	./matmul/matmul
	./reduction/reduce
	./histogram/hist

clean:
	rm -f matmul/matmul reduction/reduce histogram/hist

.PHONY: all run clean
