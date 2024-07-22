#ifndef float64
#define float64 double
#endif

#define D0 get_global_id(2)
#define D1 get_global_id(1)
#define D2 get_global_id(0)

#define L0 get_global_size(2)
#define L1 get_global_size(1)

#define IDXC D2*L1 + D1*L0 + D0

#define IDX_A_N0B1N0 (D2* 1 +  0)*L0 + D0
#define IDX_B_N0B1N0 (D2*L1 + D1)*L0 + D0
#define IDX_A_N0B1B2 (D2* 1 +  0)*L0 + D0
#define IDX_B_N0B1B2 (D2*L1 + D1)* 1 +  0
#define IDX_A_N0B2N0 (D2*L1 + D1)*L0 + D0
#define IDX_B_N0B2N0 (D2* 1 +  0)*L0 + D0
#define IDX_A_N0B2B1 (D2*L1 + D1)* 1 +  0
#define IDX_B_N0B2B1 (D2* 1 +  0)*L0 + D0

#define IDX_A_B1N0B1 ( 0*L1 + D1)* 1 +  0
#define IDX_B_B1N0B1 (D2*L1 + D1)*L0 + D0
#define IDX_A_B1N0B2 ( 0*L1 + D1)*L0 + D2
#define IDX_B_B1N0B2 (D2*L1 + D1)* 1 +  0
#define IDX_A_B1B2N0 ( 0*L1 + D1)*L0 + D2
#define IDX_B_B1B2N0 (D2* 1 +  0)*L0 + D2
#define IDX_A_B1B2B1 ( 0*L1 + D1)* 1 +  0
#define IDX_B_B1B2B1 (D2* 1 +  0)*L0 + D2

#define IDX_A_B2N0B1 (D2*L1 + D1)* 1 +  0
#define IDX_B_B2N0B1 ( 0*L1 + D1)*L0 + D0
#define IDX_A_B2N0B2 (D2*L1 + D1)*L0 + D0
#define IDX_B_B2N0B2 ( 0*L1 + D1)* 1 +  0
#define IDX_A_B2B1N0 (D2* 1 +  0)*L0 + D0
#define IDX_B_B2B1N0 ( 0*L1 + D1)* 1 +  0
#define IDX_A_B2B1B2 (D2* 1 +  0)*L0 + D0
#define IDX_B_B2B1B2 ( 0*L1 + D1)* 1 +  0


void kernel tensor_tensor_broadcast_add_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] + B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_add_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] + B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_add_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] + B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_add_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] + B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_add_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] + B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_add_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] + B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_add_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] + B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_add_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] + B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_add_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] + B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_add_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] + B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_add_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] + B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_add_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] + B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_sub_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] - B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_sub_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] - B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_sub_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] - B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_sub_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] - B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_sub_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] - B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_sub_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] - B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_sub_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] - B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_sub_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] - B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_sub_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] - B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_sub_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] - B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_sub_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] - B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_sub_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] - B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_mul_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] * B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_mul_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] * B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_mul_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] * B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_mul_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] * B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_mul_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] * B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_mul_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] * B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_mul_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] * B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_mul_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] * B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_mul_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] * B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_mul_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] * B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_mul_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] * B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_mul_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] * B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_div_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] / B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_div_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] / B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_div_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] / B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_div_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] / B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_div_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] / B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_div_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] / B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_div_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] / B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_div_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] / B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_div_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] / B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_div_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] / B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_div_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] / B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_div_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] / B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_equals_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] == B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_equals_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] == B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_equals_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] == B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_equals_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] == B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_equals_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] == B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_equals_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] == B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_equals_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] == B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_equals_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] == B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_equals_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] == B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_equals_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] == B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_equals_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] == B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_equals_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] == B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_gt_equals_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] >= B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_gt_equals_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] >= B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_gt_equals_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] >= B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_gt_equals_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] >= B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_gt_equals_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] >= B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_gt_equals_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] >= B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_gt_equals_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] >= B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_gt_equals_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] >= B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_gt_equals_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] >= B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_gt_equals_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] >= B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_gt_equals_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] >= B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_gt_equals_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] >= B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_lt_equals_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] <= B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_lt_equals_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] <= B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_lt_equals_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] <= B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_lt_equals_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] <= B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_lt_equals_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] <= B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_lt_equals_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] <= B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_lt_equals_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] <= B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_lt_equals_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] <= B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_lt_equals_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] <= B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_lt_equals_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] <= B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_lt_equals_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] <= B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_lt_equals_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] <= B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_gt_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] > B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_gt_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] > B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_gt_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] > B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_gt_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] > B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_gt_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] > B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_gt_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] > B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_gt_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] > B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_gt_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] > B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_gt_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] > B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_gt_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] > B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_gt_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] > B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_gt_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] > B[IDX_B_B2B1B2]; }

void kernel tensor_tensor_broadcast_lt_n0_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1N0] < B[IDX_B_N0B1N0]; }
void kernel tensor_tensor_broadcast_lt_n0_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B1B2] < B[IDX_B_N0B1B2]; }
void kernel tensor_tensor_broadcast_lt_n0_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2N0] < B[IDX_B_N0B2N0]; }
void kernel tensor_tensor_broadcast_lt_n0_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_N0B2B1] < B[IDX_B_N0B2B1]; }
void kernel tensor_tensor_broadcast_lt_b1_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B1] < B[IDX_B_B1N0B1]; }
void kernel tensor_tensor_broadcast_lt_b1_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1N0B2] < B[IDX_B_B1N0B2]; }
void kernel tensor_tensor_broadcast_lt_b1_b2_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2N0] < B[IDX_B_B1B2N0]; }
void kernel tensor_tensor_broadcast_lt_b1_b2_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B1B2B1] < B[IDX_B_B1B2B1]; }
void kernel tensor_tensor_broadcast_lt_b2_n0_b1(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B1] < B[IDX_B_B2N0B1]; }
void kernel tensor_tensor_broadcast_lt_b2_n0_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2N0B2] < B[IDX_B_B2N0B2]; }
void kernel tensor_tensor_broadcast_lt_b2_b1_n0(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1N0] < B[IDX_B_B2B1N0]; }
void kernel tensor_tensor_broadcast_lt_b2_b1_b2(constant float64* A, constant float64* B, global float64* C) { C[IDXC] = A[IDX_A_B2B1B2] < B[IDX_B_B2B1B2]; }