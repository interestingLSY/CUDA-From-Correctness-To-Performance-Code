#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>

#include <cuda_runtime.h>

#include "gemm_cpu_naive.h"
#include "gemm_cpu_simd.h"
#include "gemm_gpu_1thread.h"
#include "gemm_gpu_mult_thread.h"
#include "gemm_gpu_mult_block.h"
#include "gemm_gpu_mult_block_no_restrict.h"
#include "gemm_gpu_mult_block_no_restrict_reg.h"

// gemm_impl_t - A function pointer type for gemm implementations
typedef void (*gemm_impl_t)(
	int* __restrict__ C,
	const int* __restrict__ A,
	const int* __restrict__ B,
	const int n,
	const int m,
	const int k
);

struct GemmImpl {
	std::string name;
	gemm_impl_t impl;
	bool is_gpu;
};

// All GEMM implementations to benchmark
std::vector<GemmImpl> gemm_impls = {
	{ "cpu_naive", gemm_cpu_naive, false },
	{ "cpu_simd", gemm_cpu_simd, false },
	{ "gpu_1thread", gemm_gpu_1thread, true },
	{ "gpu_mult_thread", gemm_gpu_mult_thread, true },
	{ "gpu_mult_block", gemm_gpu_mult_block, true },
	{ "gpu_mult_block_no_restrict", gemm_gpu_mult_block_no_restrict, true },
	{ "gpu_mult_block_no_restrict_reg", gemm_gpu_mult_block_no_restrict_reg, true }
};

// cuda_sync_check_error - Sync with the CUDA device, check if there
// is any error, and print the error message if there is any.
void cuda_sync_check_error_helper(const char* filename, const int line) {
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error at %s:%d: %s\n", filename, line, cudaGetErrorString(error));
		exit(1);
	}
}
#define cuda_sync_check_error() cuda_sync_check_error_helper(__FILE__, __LINE__)

constexpr int BENCHMARK_ROUNDS = 8;
// benchmark_gemm_impl - Benchmark a gemm implementation, return the
// avg time usage on BENCKMARK_ROUNDS rounds
double benchmark_gemm_impl(
	GemmImpl gemm_impl,
	const int n,
	const int m,
	const int k
) {
	// Prepare test data

	// Allocate A, B, C, and C_ref on CPU
	int* A = new int[n * k];
	int* B = new int[k * m];
	int* C = new int[n * m];
	int* C_ref = new int[n * m];

	// Allocate A_gpu, B_gpu, and C_gpu on GPU
	int* A_gpu;
	int* B_gpu;
	int* C_gpu;
	cudaMalloc(&A_gpu, sizeof(int) * n * k);
	cudaMalloc(&B_gpu, sizeof(int) * k * m);
	cudaMalloc(&C_gpu, sizeof(int) * n * m);

	// Initialize A and B and copy them to GPU
	for (int i = 0; i < n * k; ++i) A[i] = rand() % 1000;
	for (int i = 0; i < k * m; ++i) B[i] = rand() % 1000;
	cudaMemcpy(A_gpu, A, sizeof(int) * n * k, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(int) * k * m, cudaMemcpyHostToDevice);

	// Initialize C_ref
	memset(C_ref, 0, sizeof(int) * n * m);
	gemm_cpu_naive(C_ref, A, B, n, m, k);

	// run_once: Run the gemm_impl once, and return its time usage (in microseconds (us))
	std::function<long(void)> run_once = [&]() -> long {
		if (gemm_impl.is_gpu) {
			cudaMemset(C_gpu, 0, sizeof(int) * n * m);
			auto start = std::chrono::high_resolution_clock::now();
			gemm_impl.impl(C_gpu, A_gpu, B_gpu, n, m, k);
			cuda_sync_check_error();
			auto end = std::chrono::high_resolution_clock::now();
			return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		} else {
			memset(C, 0, sizeof(int) * n * m);
			auto start = std::chrono::high_resolution_clock::now();
			gemm_impl.impl(C, A, B, n, m, k);
			auto end = std::chrono::high_resolution_clock::now();
			return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		}
	};

	// Warm up
	printf("Warming up...\n");
	run_once();

	// Verift its correctness
	if (gemm_impl.is_gpu) {
		cudaMemcpy(C, C_gpu, sizeof(int) * n * m, cudaMemcpyDeviceToHost);
	}
	for (int i = 0; i < n * m; ++i) {
		if (C[i] != C_ref[i]) {
			printf("Verification failed!\n");
			printf("C[%d, %d] = %d, C_ref[%d, %d] = %d\n", i / m, i % m, C[i], i / m, i % m, C_ref[i]);
			return -1;
		}
	}
	std::cout << "Verification passed!" << std::endl;

	// Warm up again since correct verification may corrupt cache
	printf("Warming up (again)...\n");
	run_once();

	// Benchmark
	long total_time_usage = 0;
	for (int round = 0; round < BENCHMARK_ROUNDS; ++round) {
		long time_usage = run_once();
		printf("Round %d: %ld us\n", round, time_usage);
		total_time_usage += time_usage;
	}
	double avg_time_usage = total_time_usage / (double)BENCHMARK_ROUNDS;

	// Free memory
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] C_ref;
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

	return avg_time_usage;
}

int main(int argc, char* argv[]) {
	if (argc != 4 && argc != 5) {
		printf("Usage: %s <n> <m> <k> [implementation]\n", argv[0]);
		exit(1);
	}

	// Parse command line arguments
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	int k = atoi(argv[3]);
	std::string impl = argc == 5 ? argv[4] : "*";
	assert (n > 0 && m > 0 && k > 0);

	// We allocate a small block of memory on GPU here to initialize the
	// CUDA context. If we does not do this and the user hasn't enabled
	// GPU's persistent mode, then the first CUDA call will take a long
	// time to finish
	int* dummy;
	cudaMalloc(&dummy, sizeof(int));
	
	std::vector<std::pair<std::string, double>> results;
	for (auto gemm_impl : gemm_impls) {
		if (impl == "*" || gemm_impl.name == impl) {
			printf("----------------\n");
			printf("Benchmarking %s...\n", gemm_impl.name.c_str());
			double avg_time_usage = benchmark_gemm_impl(gemm_impl, n, m, k);
			printf("Average time usage: %lf us\n", avg_time_usage);
			results.push_back({ gemm_impl.name, avg_time_usage });
		}
	}

	// Print results
	printf("----------------\n");
	printf("Results:\n");
	for (auto result : results) {
		printf("%16s %16.2lf us\n", result.first.c_str(), result.second);
	}

	return 0;
}