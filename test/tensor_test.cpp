#include <gtest/gtest.h>

#include <chrono>
//#define USE_PROFILING
// #define AUTO_PRINT_PROFILER_STATS
#include "Tensor.h"
using namespace std;
using namespace RedFish;

// EXPECT_EQ(val1 , val2);   equal 
// EXPECT_NE(val1 , val2);   notequal
// EXPECT_TRUE(VAL);
// EXPECT_FALSE(VAL);


TEST(TensorTest, equalTest)
{
    Tensor t1({5, 5, 5});
    Tensor t2({5, 5, 5});
    t1.ones();
    t2.ones();
    EXPECT_TRUE(t1 == t2);
    
    Tensor t3({5, 5, 5});
    Tensor t4({5, 5, 5});
    
    t3.rand();
    t4.rand();
    EXPECT_FALSE(t3 == t4);


    Tensor t5({5, 5});
    t5.ones();
    EXPECT_FALSE(t1 == t5);
}

TEST(TensorTest, stackTest)
{

    Tensor t1({3, 3, 2});
    Tensor t2({3, 3, 2});
    
    t1.rand();
    t2.rand();

    //std::cout << t1 << t2 << stack(t1, t2, 0);

}

TEST(TensorTest, matmul)
{
    Tensor t1({2,3,2}), t2({2,3});
    /* Tensor t3({1024,1024}), t4({1024,1024});
    Tensor t5({1024,1024}), t6({1024,1024});
    t1.rand();
    t2.rand();
    t3.rand();
    t4.rand();
    t5.rand();
    t6.rand(); */
    for (size_t i = 0; i < 6*2; i++) t1(i) = i;
    for (size_t i = 0; i < 6; i++) t2(i) = i;
    
    std::cout << "\n\n\nMatmul test \n" << t1 << t2 << matmul(t1, t2) << "\n\n\n";
    
    /* auto time = std::chrono::high_resolution_clock::now();
    matmul(t1, t2, LEFT);
    std::cout << "Matmul left took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
    time = std::chrono::high_resolution_clock::now();
    matmul(t3, t4, NONE);
    std::cout << "Matmul took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
    time = std::chrono::high_resolution_clock::now();
    matmul(t5, t6, RIGHT);
    std::cout << "Matmul right took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n"; */
}

TEST(TensorTest, broadcastSum)
{
    Tensor t1({3,3,3}), t2({3,1,3});
    t1.rand();
    t2.rand();
    //std::cout << "\n\n\nBroadcast sum test \n" << t1 << t2 << t1 + t2 << "\n\n\n";
    
    //auto time = std::chrono::high_resolution_clock::now();
    t1 + t2;
    //std::cout << "Broadcast sum took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
}

TEST(TensorTest, broadcastSumAssign)
{
    Tensor t1({3,3,3}), t2({3,1,3});
    t1.rand();
    t2.rand();
    /* std::cout << "\n\n\nBroadcast sum assign test \n" << t1 << t2;
    t2 += t1;
    std::cout << t2 << "\n\n\n"; */
}

TEST(TensorTest, alongAxisTest)
{
    Tensor t1({5, 5, 5});
    t1.rand(10, 20);

    // std::cout << t1 << t1.max(0) << t1.max(1) << t1.max(2) << t1.max(0).max(1);
}

TEST(TensorTest, cross1d)
{
    Tensor t({5}), k({3});
    //t.rand(), k.rand();
    k((size_t)0) = k((size_t)1) = k((size_t)2) = 1./3;
    for (size_t i = 0; i < 5; i++)
        t(i) = i;
        
    //std::cout << t << k << t.crossCorrelation1d(k) << t.crossCorrelation1d(k, 1) << t.crossCorrelation1d(k, 1, 2) << t.crossCorrelation1d(k, 1, 1, 3);
    /* auto time = std::chrono::high_resolution_clock::now();
    auto tt = t.crossCorrelation1d(k, 1);
    std::cout << "Cross correlation took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n"; */
}

TEST(TensorTest, cross2d)
{
    Tensor t({1,1088, 1920}), k({51,51});
    //t.rand(), k.rand();
    /* Tensor t({2,5,5}), k({3, 3});
    for (size_t i = 0; i < 3*3; i++)
        k(i) = 1./9;
    for (size_t i = 0; i < 2*5*5; i++)
        t(i) = i; */

    //std::cout << t << k << t.crossCorrelation2d(k, 1, 1, 3);
    auto time = std::chrono::high_resolution_clock::now();
    //auto tt = t.crossCorrelation2d(k, 0, 1, 1);
    std::cout << "Cross correlation took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
}

TEST(TensorTest, transposedTest)
{
    Tensor t1({10, 50, 50});
    t1.rand(10, 20);
    
    auto time = std::chrono::high_resolution_clock::now();
    auto tt = t1.T();
    // std::cout << "Transposition took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
}

TEST(TensorTest, transposedDft)
{
    /* float64 buff[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::complex<float64> buff2[16];
    fft_impl_reversed(buff2, buff, 16);
    for (size_t i = 0; i < sizeof(buff2) / sizeof(float64)/2; i++)
        std::cout << buff2[reverse_bits(i) >> 60] << "\n"; */
        
    /* float64 buff[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::complex<float64> buff2[16];
    fft2d_impl_reversed(buff2, buff, 4);
    for (size_t j = 0; j < 4; j++, std::cout << "\n")
    for (size_t i = 0; i < 4; i++)
        std::cout << buff2[(reverse_bits(j) >> 62)*4 + (reverse_bits(i) >> 62)] << " "; */
    
    size_t size = 1024;
    std::unique_ptr<float64[]> buff = std::make_unique<float64[]>(size*1024);
    std::unique_ptr<std::complex<float64>[]> buff2 = std::make_unique<std::complex<float64>[]>(size*1024);
    
    auto time = std::chrono::high_resolution_clock::now();
    //for (size_t i = 0; i < 1024; i++)
        fft2d_impl_reversed(buff2.get()/*+ i*size */, buff.get()/* +i*size */, 1024);
    std::cout << "fft took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}