void kernel tensor_scalar_add(global const double* A, double B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B;
}

void kernel tensor_tensor_add(global const double* A, global const double* B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}

void kernel tensor_scalar_sub(global const double* A, double B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B;
}

void kernel tensor_tensor_sub(global const double* A, global const double* B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B[get_global_id(0)];
}

void kernel tensor_scalar_mul(global const double* A, double B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B;
}

void kernel tensor_tensor_mul(global const double* A, global const double* B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];
}

void kernel tensor_scalar_div(global const double* A, double B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B;
}

void kernel tensor_tensor_div(global const double* A, global const double* B, global double* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B[get_global_id(0)];
}