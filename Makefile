#
# Makefile
#
# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-8.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# CUDA code generation flags: first is binary, second is assembly (PTX), and
# the number is the NVIDIA "compute capability", e.g., 3.5.  Runtime will then
# select that most appropriate for the platform.
GCODE_SM20    := -gencode arch=compute_20,code=sm_20
GCODE_SM30    := -gencode arch=compute_30,code=sm_30
GCODE_SM35    := -gencode arch=compute_35,code=sm_35
GCODE_FLAGS   := $(GCODE_SM20) $(GCODE_SM30) $(GCODE_SM35)

LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64
NVCCFLAGS := -m64

all: helloworld

helloworld.o: helloworld.cu helloworld.h helloworld_kernels.cu
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC_PATH) $(GCODE_FLAGS) -o $@ -c $<

helloworld: helloworld.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f helloworld helloworld.o 

