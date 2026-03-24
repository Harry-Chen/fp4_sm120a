CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120f

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_CODE_FLAGS := -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) 
NVCC_COMMON_FLAGS := -Xptxas=-v -O3 -std=c++20 -g
NVCC_FLAGS := $(NVCC_CODE_FLAGS) $(NVCC_COMMON_FLAGS)

SRC := $(wildcard *.cu)
EXE := $(SRC:%.cu=%.exe)
PTX := $(SRC:%.cu=%.ptx)

all: $(EXE)

compare.exe: compare.cu compare.cuh
	$(NVCC) -gencode=arch=compute_100a,code=sm_100a $(NVCC_COMMON_FLAGS) $< -o $@

%.exe: %.cu %.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

%.ptx: %.cu %.cuh
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

.PHONY: all clean

clean:
	rm -rf $(EXE) $(PTX)

