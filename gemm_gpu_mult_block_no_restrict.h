#pragma once

void gemm_gpu_mult_block_no_restrict(
	int* C,		// [n, m]
	const int* A,	// [n, k]
	const int* B,	// [k, m]
	const int n,
	const int m,
	const int k
);

