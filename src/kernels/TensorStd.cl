#ifndef float64
#define float64 double
#endif

void kernel tensor_sqrt(constant float64* A, global float64* C)
{
    C[get_global_id(0)] = sqrt(A[get_global_id(0)]);
}

void kernel tensor_exp(constant float64* A, global float64* C)
{
    C[get_global_id(0)] = exp(A[get_global_id(0)]);
}

void kernel tensor_log(constant float64* A, global float64* C)
{
    C[get_global_id(0)] = log(A[get_global_id(0)]);
}

void kernel tensor_pow(constant float64* A, const float64 power, global float64* C)
{
    C[get_global_id(0)] = pow(power, A[get_global_id(0)]);
}

void kernel tensor_tensor_pow(constant float64* A, constant float64* B, global float64* C)
{
    C[get_global_id(0)] = pow(B[get_global_id(0)], A[get_global_id(0)]);
}
