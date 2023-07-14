#include "gemm_cpu_simd.h"

// gemm_cpu_simd - A SIMD gemm (GEneral Matrix Multiply) implementation on CPU
// Input:
//	- A: A n x k matrix
//	- B: A k x m matrix
//	- n: The number of rows of A
//	- m: The number of columns of B
//	- k: The number of columns of A and the number of rows of B
// Output:
//	- C: A n x m matrix. The result of A * B
// Requirements:
//	- Please make sure C is initialized to 0 before calling this function
__attribute__((optimize("O3")))	// Enforce O3 optimization to utilize loop unrolling and SIMD
void gemm_cpu_simd(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	for (int l = 0; l < k; ++l)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j) {
				C[i * m + j] += A[i * k + l] * B[l * m + j];
			}
}
