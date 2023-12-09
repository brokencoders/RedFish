#include <iostream>
#include <random>
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

// Check If the result match 
bool check(double* A, double* B, int n)
{
    int size = n * n;
    for (int i = 0; i < size; i++)
        if(A[i] != B[i]) return false; 
    return true;
}

// Naive implementation to check if mat mull is correct
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

    // Cpu TEST
    OpenCLManager::init();
    OpenCLManager::createSourceFromFile("../src/kernels/TensorMul.cl");
    OpenCLManager::createProgram();
    
    Kernel kernel = OpenCLManager::createKernel("tensor_tensor_math_mul");

    std::vector<std::vector<std::pair<double, double>>> benchmarcks;
    std::vector<std::pair<double, double>> benchmarcks_C;
    std::vector<std::pair<double, double>> benchmarcks_C_OPENGL;
    std::vector<std::pair<double, double>> benchmarcks_BOSE;

    for(size_t i = 1; i < 10; i++)
    {
        int square_size = 100 * i;
        int M = square_size;
        int N = square_size;
        int K = square_size;
        int dim = M * N;

        double* A = new double[M*K];           // A        (M, K)
        double* B = new double[K*N];           // B        (K, N)
        double* C = new double[dim];           // C        (M, N)
        double* C_OPENCL = new double[dim];    // C_OPENCL (M, N)

        init_matrix(A, M*K);
        init_matrix(B, K*N);

        // DELETE BUFFERS
        Buffer buffer_A = OpenCLManager::createBuffer<double>(M*K);
        Buffer buffer_B = OpenCLManager::createBuffer<double>(K*N);
        Buffer buffer_C = OpenCLManager::createBuffer<double>(dim);
    
        auto start_opencl = std::chrono::high_resolution_clock::now();
        OpenCLManager::loadWriteBuffer<double>(buffer_B, K*N, B);
        OpenCLManager::loadWriteBuffer<double>(buffer_A, M*K, A);
        
        // Take multiple data and plot 

        OpenCLManager::execute(kernel, { M, N, K}, { buffer_A, buffer_B, buffer_C}, {(size_t)M, (size_t)N}, {(size_t)10, (size_t)10});
        OpenCLManager::loadReadBuffer<double>(buffer_C, dim, C_OPENCL);
        auto stop_opencl = std::chrono::high_resolution_clock::now();
        auto duration_opencl = std::chrono::duration_cast<std::chrono::microseconds>(stop_opencl - start_opencl);
        std::cout << "Time taken by Daniel OpenCL: " << duration_opencl.count() << " microseconds" << std::endl;
        benchmarcks_C_OPENGL.push_back(std::make_pair(square_size, duration_opencl.count()));

        auto start_c = std::chrono::high_resolution_clock::now();
        cpp_matrix_mul(M, N, K, A, B, C);
        auto stop_c = std::chrono::high_resolution_clock::now();
        auto duration_c = std::chrono::duration_cast<std::chrono::microseconds>(stop_c - start_c);
        std::cout << "Time taken by C: " << duration_c.count() << " microseconds" << std::endl;
        benchmarcks_C.push_back(std::make_pair(square_size, duration_c.count()));

        Tensor TA({(size_t)M, (size_t)K}, A);
        Tensor TB({(size_t)K, (size_t)N}, B);
        auto start_bose = std::chrono::high_resolution_clock::now();
        Tensor TC = TA * TB;
        auto stop_bose = std::chrono::high_resolution_clock::now();
        auto duration_bose = std::chrono::duration_cast<std::chrono::microseconds>(stop_bose - start_bose);
        std::cout << "Time taken by Bose: " << duration_bose.count() << " microseconds" << std::endl;
        benchmarcks_BOSE.push_back(std::make_pair(square_size, duration_bose.count()));

        if (check(C, C_OPENCL, N))
            std::cout << "OPENCL Correct" << std::endl;
        
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_OPENCL;
    }

    benchmarcks.push_back(benchmarcks_C);
    benchmarcks.push_back(benchmarcks_C_OPENGL);
    benchmarcks.push_back(benchmarcks_BOSE);

    plot_function_data(benchmarcks, {0, 1000}, {0, 100000}, {"C", "OPENCL", "BOSE"});

    return 0;
}