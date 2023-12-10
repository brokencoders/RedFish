#include <iostream>
#include <random>
#include "OpenCLManager.h"
#include "RedFish.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> distribution(0, 10);
int random_number = distribution(gen);

void init_matrix(double* M, int dim)
{
    for(int i = 0; i < dim; i++)
        M[i] = distribution(gen);
}

void print_matrix(double* Mat, int M, int N)
{
    for (int i = 0; i < M; i++)      // Row
    {
        for (int j = 0; j < N; j++)  // Col 
            std::cout << Mat[i * N + j] << "  ";
        std::cout << std::endl;
    }
}

bool check(double* A, double* B, int n)
{
    int size = n * n;
    for (int i = 0; i < size; i++)
        if(A[i] != B[i]) 
        {
            return false; 
        }
    return true;
}

void cpp_matrix_mul(const int M, const int N, const int K, const double* A, const double* B, double* C)
{
    for (int m = 0; m < M; m++) {             // Row
        for (int n = 0; n < N; n++) {         // Col
            double acc = 0.0f;
            for (int k = 0; k < K; k++) {     // K 
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

int main()
{

    OpenCLManager::init();
    OpenCLManager::createSourceFromFile("../src/kernels/TensorMul.cl");
    OpenCLManager::createProgram();
    Kernel mat_mul = OpenCLManager::createKernel("tensor_tensor_math_mul");

    std::vector<std::vector<std::pair<double, double>>> benchmarcks;
    std::vector<std::pair<double, double>> benchmarcks_C;
    std::vector<std::pair<double, double>> benchmarcks_C_OPENGL;
    std::vector<std::pair<double, double>> benchmarcks_BOSE;

    for(int i = 16; i <= 512; i*=2)
    {
        size_t M = i;
        size_t N = 2 * i;
        size_t K = 3 * i;
        double* A = new double[M * K];
        double* B = new double[K * N];
        double* C = new double [M * N];
        double* C1 = new double[M * N];
        double* C2 = new double[M * N];
        
        init_matrix(A, M * K);
        init_matrix(B, K * N);


        Buffer bufferA = OpenCLManager::createBuffer<double>(M * K);
        Buffer bufferB = OpenCLManager::createBuffer<double>(K * N);
        Buffer bufferC = OpenCLManager::createBuffer<double>(M * N);

        auto start_opencl = std::chrono::high_resolution_clock::now();
        
        OpenCLManager::loadWriteBuffer<double>(bufferA, M * K, A);
        OpenCLManager::loadWriteBuffer<double>(bufferB, K * N, B);
        OpenCLManager::execute(mat_mul, {(int)M, (int)N, (int)K}, {bufferA, bufferB, bufferC}, {M, N}, {16, 16});
        OpenCLManager::loadReadBuffer<double>(bufferC, M * N, C);
        
        auto stop_opencl = std::chrono::high_resolution_clock::now();
        auto duration_opencl = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_opencl - start_opencl);
        std::cout << "Time taken by Daniel OpenCL: " << duration_opencl.count() << " ms" << std::endl;
        benchmarcks_C_OPENGL.push_back(std::make_pair(M * N, duration_opencl.count()));

        auto start_bose = std::chrono::high_resolution_clock::now();
        
        std::fill(C1, C1 + M * N, 0.);
        matmul_gotoblas(C1, A, B, M, K, N, N, K, N);
        
        auto stop_bose = std::chrono::high_resolution_clock::now();
        auto duration_bose = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_bose - start_bose);
        std::cout << "Time taken by Bose: " << duration_bose.count() << " ms" << std::endl;
        benchmarcks_BOSE.push_back(std::make_pair(M * N, duration_bose.count()));

        cpp_matrix_mul(M, N, K, A, B, C2);

        std::cout << i << std::endl;
        if(check(C, C2, M * N))
            std::cout << ":)" << std::endl;
        else 
            std::cout << ":(" << std::endl;
        
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C1;
    }

    benchmarcks.push_back(benchmarcks_C_OPENGL);
    benchmarcks.push_back(benchmarcks_BOSE);

    plot_function_data(benchmarcks, {0, 512}, {0, 2000000}, { "OPENCL", "BOSE"});

    return 0;
}