CC=gcc
NVCC=nvcc

CFLAGS= -Wall -o

all: des_cpu des_gpu

des_cpu: des_cpu.c	
	$(CC) $(CFLAGS) des_cpu des_cpu.c 
des_gpu: des_gpu.c	
	$(NVCC) des_gpu des_gpu.cu
.PHONY: 
	clean all
clean:
	rm des_cpu des_gpu