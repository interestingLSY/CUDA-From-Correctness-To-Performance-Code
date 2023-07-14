#include "gemm_gpu_1thread.h"

#include <cuda_runtime_api.h>

// gemm_gpu_1thread - GEMM on GPU, using only one thread
__global__
void gemm_gpu_1thread_kernel(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j) {
			int res = 0;
			for (int l = 0; l < k; ++l) {
				res += A[i * k + l] * B[l * m + j];
			}
			C[i * m + j] = res;
		}
}

void gemm_gpu_1thread(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_1thread_kernel<<<1, 1>>>(C, A, B, n, m, k);
}
