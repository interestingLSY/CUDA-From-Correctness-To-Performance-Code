CC = nvcc
CXXFLAGS = -O3

DEPS = gemm_cpu_naive.h gemm_cpu_simd.h gemm_gpu_1thread.h
OBJS = gemm_cpu_naive.o gemm_cpu_simd.o gemm_test.o gemm_gpu_1thread.o

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