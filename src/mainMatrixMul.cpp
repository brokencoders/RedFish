#include "RedFish.h"

// Check If the result match 
bool check(float* A, float* B, int n)
{
    int size = n * n;
    for (size_t i = 0; i < size; i++)
        if(A[i] != B[i]) return false; 
    
    return true;
}

// Naive Matrix Multiplication
// Matrix(M, K) * Matrix(K, N) = Matrix(M, N)
void cpp_matrix_mull(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
    for (int m=0; m < M; m++) {
        for (int n=0; n < N; n++) {
            float acc = 0.0f;
            for (int k=0; k < K; k++) {
                acc += A[k * N + m] * B[n * N + k];
            }
            C[n*N + m] = acc;
        }
    }
}

int main()
{
    // Only Square Matrix
    int M = 2;
    int N = 2;
    int K = 2;
    int dim = M * N;

    float A[] = { 1, 2, 3, 4};
    float B[] = { 1, 2, 3, 4};
    float C[dim];
    float C_OPENCL[dim];

    // Cpu TEST
    cpp_matrix_mull(M, N, K, A, B, C);

    // OpenCL TEST
    OpenCLManager::init();
    OpenCLManager::createSourceFromFile("../src/kernels/TensorMul.cl");
    OpenCLManager::createProgram();

    Kernel kernel = OpenCLManager::createKernel("tensor_tensor_math_mul");

    Buffer buffer_A = OpenCLManager::createBuffer<float>(dim);
    Buffer buffer_B = OpenCLManager::createBuffer<float>(dim);
    Buffer buffer_C = OpenCLManager::createBuffer<float>(dim);
   
    OpenCLManager::loadWriteBuffer<float>(buffer_A, dim, A);
    OpenCLManager::loadWriteBuffer<float>(buffer_B, dim, B);
    
    OpenCLManager::execute(kernel, { M, N, K}, { buffer_A, buffer_B, buffer_C}, N);
    OpenCLManager::loadReadBuffer<float>(buffer_C, dim, C_OPENCL);

    if (check(C, C_OPENCL, N))
    {
        std::cout << "Correct" << std::endl;
    }

    return 0;
}