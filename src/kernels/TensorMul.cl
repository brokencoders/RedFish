// Matrix(M, K) * Matrix(K, N) = Matrix(M, N)
__kernel void tensor_tensor_math_mul(                 
        const int M, const int N, const int K, 
        const __global double* A, 
        const __global double* B,
        __global double* C) 
{
    int k;
    int i = get_global_id(0);   // Row 0, M
    int j = get_global_id(1);   // Col 0, N


    double tmp = 0.0f;
    for(k = 0; k < K; k++)
        tmp += A[i * K + k] * B[k * N + j];

    C[i * N + j] = tmp;
}