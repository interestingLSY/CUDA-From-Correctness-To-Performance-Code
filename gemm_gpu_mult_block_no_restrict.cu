#include "gemm_gpu_mult_block.h"

#include <cuda_runtime_api.h>

// gemm_gpu_mult_block_no_restrict - GEMM on GPU, using many blocks
// and without the __restrict__ keyword
// The block size is N
// The grid size is M
__global__
void gemm_gpu_mult_block_no_restrict_kernel(
	int* C,		// [n, m], on gpu
	const int* A,	// [n, k], on gpu
	const int* B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	const int i = threadIdx.x;
	const int j = blockIdx.x;
	for (int l = 0; l < k; ++l) {
		C[i * m + j] += A[i * k + l] * B[l * m + j];
	}
}

void gemm_gpu_mult_block_no_restrict(
	int* C,		// [n, m], on gpu
	const int* A,	// [n, k], on gpu
	const int* B,	// [k, m], on gpu
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mult_block_no_restrict_kernel<<<m, n>>>(C, A, B, n, m, k);
}
