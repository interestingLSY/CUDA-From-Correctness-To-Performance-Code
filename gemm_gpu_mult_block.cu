#include "gemm_gpu_mult_block.h"

#include <cuda_runtime_api.h>

// gemm_gpu_mult_block - GEMM on GPU, using many blocks
// The block size is N
// The grid size is M
__global__
void gemm_gpu_mult_block_kernel(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	const int i = threadIdx.x;
	const int j = blockIdx.x;
	int res = 0;
	for (int l = 0; l < k; ++l) {
		res += A[i * k + l] * B[l * m + j];
	}
	C[i * m + j] = res;
}

void gemm_gpu_mult_block(
	int* __restrict__ C,		// [n, m], on gpu
	const int* __restrict__ A,	// [n, k], on gpu
	const int* __restrict__ B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mult_block_kernel<<<m, n>>>(C, A, B, n, m, k);
}
