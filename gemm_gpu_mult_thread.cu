#include "gemm_gpu_mult_thread.h"

#include <cuda_runtime_api.h>

// gemm_gpu_mult_thread - GEMM on GPU, using only one block
// The block size is N
__global__
void gemm_gpu_mult_thread_kernel(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	const int i = threadIdx.x;
	for (int j = 0; j < m; ++j)
		for (int l = 0; l < k; ++l) {
			C[i * m + j] += A[i * k + l] * B[l * m + j];
		}
}

void gemm_gpu_mult_thread(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mult_thread_kernel<<<1, n>>>(C, A, B, n, m, k);
}
