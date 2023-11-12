#include <gtest/gtest.h>

#include "Layer.h"
#include "ConvLayer.h"
using namespace std;
using namespace RedFish;

// EXPECT_EQ(val1 , val2);   equal 
// EXPECT_NE(val1 , val2);   notequal
// EXPECT_TRUE(VAL);
// EXPECT_FALSE(VAL);

TEST(RedFishTest, linearLayer)
{
    Conv1dLayer l(2, 2, 5);
    Tensor X({3, 2, 50}), kernels({2,2,5});
    X.rand();
    kernels.rand();
    //std::cout << kernels <<.crossCorrelation1d(kernels.getRow({0, 0}));
    std::cout << X << l.farward(X);

}
