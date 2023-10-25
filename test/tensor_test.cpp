#include <gtest/gtest.h>

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

TEST(TensorTest, sumTest)
{

}

TEST(TensorTest, subTest)
{

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}