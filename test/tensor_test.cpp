#include <gtest/gtest.h>

#include <chrono>
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

    std::cout << t1 << t2 << stack(t1, t2, 0);

}

TEST(TensorTest, matmul)
{
    Tensor t1({1000,1000}), t2({1000,1000});
    t1.rand();
    t2.rand();
    //for (size_t i = 0; i < 6; i++) t1(i) = i;
    //for (size_t i = 0; i < 6; i++) t2(i) = i;
    
    std::cout << "\n\n\nMatmul test \n" << t1 << t2 << matmul(t1, t2) << "\n\n\n";
    
    auto time = std::chrono::high_resolution_clock::now();
    matmul(t1, t2);
    std::cout << "Matmul took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
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
    std::cout << "\n\n\nBroadcast sum assign test \n" << t1 << t2;
    t2 += t1;
    std::cout << t2 << "\n\n\n";
}

TEST(TensorTest, alongAxisTest)
{
    Tensor t1({5, 5, 5});
    t1.rand(10, 20);

    // std::cout << t1 << t1.max(0) << t1.max(1) << t1.max(2) << t1.max(0).max(1);
}

TEST(TensorTest, transposedTest)
{
    Tensor t1({10, 50, 50});
    t1.rand(10, 20);
    
    auto time = std::chrono::high_resolution_clock::now();
    auto tt = t1.T();
    // std::cout << "Transposition took " << (std::chrono::high_resolution_clock::now() - time).count() * 1e-9 << "s\n";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}