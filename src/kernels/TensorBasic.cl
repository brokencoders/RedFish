#define float64 double

void kernel tensor_scalar_add(global const float64* A, float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B;
}

void kernel tensor_tensor_add(global const float64* A, global const float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}

void kernel tensor_scalar_sub(global const float64* A, float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B;
}

void kernel tensor_tensor_sub(global const float64* A, global const float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] - B[get_global_id(0)];
}

void kernel tensor_scalar_mul(global const float64* A, float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B;
}

void kernel tensor_tensor_mul(global const float64* A, global const float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];
}

void kernel tensor_scalar_div(global const float64* A, float64 B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B;
}

void kernel tensor_tensor_div(global const float64* A, global const float64* B, global float64* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] / B[get_global_id(0)];
}

void kernel tensor_ones(global float64* buffer)
{
    buffer[get_global_id(0)] = 1.0;
}

void kernel tensor_print(global float64* buffer, global int* shape, global int shape_size)
{
    printf("Tensor\n");

    

}
