// Matrix(M, K) * Matrix(K, N) = Matrix(M, N)
// Only Square Matrix
__kernel void tensor_tensor_math_mul(                 
        const int M, const int N, const int K, 
        const __global float* A, 
        const __global float* B,
        __global float* C) 
{
    int k;
    int i = get_global_id(0);   // Row 0, M
    int j = get_global_id(1);   // Col 0, N


    float tmp = 0.0f;
    for(k = 0; k < K; k++)
    {
        tmp += A[i * M + k] * B[k * K + j];
    }

    C[i * M + j] = tmp;
}