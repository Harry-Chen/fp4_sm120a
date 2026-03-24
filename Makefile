CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -gencode=arch=compute_120a,code=sm_120a -Xptxas=-v -O3 -std=c++20 -g

SRC := $(wildcard *.cu)
EXE := $(SRC:%.cu=%.exe)
PTX := $(SRC:%.cu=%.ptx)

all: $(EXE)

%.exe: %.cu %.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

%.ptx: %.cu %.cuh
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

.PHONY: all clean

clean:
	rm -rf *.exe *.ptx
