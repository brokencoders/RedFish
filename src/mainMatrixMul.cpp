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
    RedFish::showDevice();
    RedFish::device(RedFish::Platform::INTEL, 0);

    Tensor t1({10, 10});                       // Load tensor in GPU        done!
    t1.toDevice();
    t1.ones();                                 // Set Tensor Value          done!
    std::cout << t1 << std::endl;              // Print Tensor in GPU       to-do

    return 0;
}