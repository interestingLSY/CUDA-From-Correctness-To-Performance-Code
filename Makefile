CC = nvcc
CXXFLAGS = -O3

DEPS = gemm_cpu_naive.h gemm_cpu_simd.h
OBJS = gemm_cpu_naive.o gemm_cpu_simd.o gemm_test.o

.PHONY: all clean

all: gemm_test

%.o: %.cc $(DEPS)
	$(CC) -c $(CXXFLAGS) $< -o $@

gemm_test: $(OBJS)
	$(CC) $(CXXFLAGS) $^ -o gemm_test

clean: 
	rm -f *.o gemm_test