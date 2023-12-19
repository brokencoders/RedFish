#define float64 double

void kernel tensor_scalar_add(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B;
}

void kernel tensor_tensor_add(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}

void kernel tensor_scalar_sub(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B;
}

void kernel scalar_tensor_sub(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = B - A[get_global_id(0)];
}

void kernel tensor_minus(constant float64* A, global float64* C)
{
    C[get_global_id(0)] = -A[get_global_id(0)];
}

void kernel tensor_tensor_sub(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B[get_global_id(0)];
}

void kernel tensor_scalar_mul(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B;
}

void kernel tensor_tensor_mul(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];
}

void kernel tensor_scalar_div(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B;
}

void kernel scalar_tensor_div(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = B / A[get_global_id(0)];
}

void kernel tensor_tensor_div(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B[get_global_id(0)];
}

void kernel tensor_tensor_equals(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] == B[get_global_id(0)];
}

void kernel tensor_scalar_equals(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] == B;
}

void kernel tensor_tensor_gt_equals(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] >= B[get_global_id(0)];
}

void kernel tensor_scalar_gt_equals(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] >= B;
}

void kernel tensor_tensor_lt_equals(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] <= B[get_global_id(0)];
}

void kernel tensor_scalar_lt_equals(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] <= B;
}

void kernel tensor_tensor_gt(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] > B[get_global_id(0)];
}

void kernel tensor_scalar_gt(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] > B;
}

void kernel tensor_tensor_lt(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] < B[get_global_id(0)];
}

void kernel tensor_scalar_lt(constant float64* A, const float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] < B;
}

void kernel tensor_print(global float64* buffer, global int* shape, int shape_size)
{
    printf("Tensor\n");

    

}
