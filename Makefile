CC = nvcc
CXXFLAGS = -O3

DEPS = gemm_cpu_naive.h gemm_cpu_simd.h gemm_gpu_1thread.h gemm_gpu_mult_thread.h gemm_gpu_mult_block.h gemm_gpu_mult_block_no_restrict.h
OBJS = gemm_cpu_naive.o gemm_cpu_simd.o gemm_test.o gemm_gpu_1thread.o gemm_gpu_mult_thread.o gemm_gpu_mult_block.o gemm_gpu_mult_block_no_restrict.o

.PHONY: all clean

all: gemm_test

%.o: %.cc $(DEPS)	# for .cc files
	$(CC) -c $(CXXFLAGS) $< -o $@

%.o: %.cu $(DEPS)	# for .cu files
	$(CC) -c $(CXXFLAGS) $< -o $@

gemm_test: $(OBJS)
	$(CC) $(CXXFLAGS) $^ -o gemm_test

clean: 
	rm -f *.o gemm_test