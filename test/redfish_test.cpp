#include <gtest/gtest.h>

#include "Layer.h"
#include "FlattenLayer.h"
#include "DropoutLayer.h"

using namespace std;
using namespace RedFish;

TEST(RedFishTest, RedFishTestFlattenLayer)
{
    Tensor t1({2, 2, 2}, {1,2,3,4,5,6,7,8});
    Tensor expected_result({}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor expected_result_2({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    
    FlattenLayer l;
    EXPECT_TRUE(l.farward(t1) == expected_result);
    
    FlattenLayer l2;
    EXPECT_TRUE(l2.farward(t1) == expected_result);
}

TEST(RedFishTest, RedFishTestDropOutLayer)
{
    Tensor t({5, 2, 2});
    t.rand();
    DropoutLayer layer(0.5);
}


TEST(RedFishTest, RedFishTestWhatever)
{

}