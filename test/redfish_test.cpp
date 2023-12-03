#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "Layer.h"
#include "FlattenLayer.h"
#include "DropoutLayer.h"
#include "MaxPoolLayer.h"
#include "ConvLayer.h"

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

TEST(RedFishTest, ConvLearning)
{
    size_t in_ch = 3;
    size_t out_ch = 2;
    size_t ker_size = 5;
    size_t stride = 2;
    size_t padding = 0;
    size_t dilation = 1;
    PaddingMode padding_mode = ZERO;
    Adam opt;
    Conv1dLayer ground_truth1d(in_ch, out_ch, ker_size, &opt, stride, padding, dilation, padding_mode);
    Conv2dLayer ground_truth2d(in_ch, out_ch, ker_size, &opt, stride, padding, dilation, padding_mode);
    Conv1dLayer learner1d(in_ch, out_ch, ker_size, &opt, stride, padding, dilation, padding_mode);
    Conv2dLayer learner2d(in_ch, out_ch, ker_size, &opt, stride, padding, dilation, padding_mode);
    
    opt.setLearningRate(1);
    for (size_t i = 0, epochs = 10000, batch_size = 5; i < epochs; i++)
    {
        const Tuple2d size = {20, 50};
        Tensor X2d({batch_size, in_ch, size.h, size.w}), X1d({batch_size, in_ch, size.w});
        X1d.rand();
        X2d.rand();

        auto gt1d = ground_truth1d.farward(X1d);
        auto gt2d = ground_truth2d.farward(X2d);

        auto l1d = learner1d.farward(X1d);
        auto l2d = learner2d.farward(X2d);

        auto dLdl1d = l1d - gt1d;
        auto dLdl2d = l2d - gt2d;

        learner1d.backward(X1d, dLdl1d);
        learner2d.backward(X2d, dLdl2d);

        /* cout << learner2d.getKernels() - ground_truth2d.getKernels();
        cout << learner2d.getBiases() - ground_truth2d.getBiases(); */
        opt.step();
    }
    
    double kmaxerr1d = (ground_truth1d.getKernels() - learner1d.getKernels()).max();
    double kmaxerr2d = (ground_truth2d.getKernels() - learner2d.getKernels()).max();
    double bmaxerr1d = (ground_truth1d.getBiases() - learner1d.getBiases()).max();
    double bmaxerr2d = (ground_truth2d.getBiases() - learner2d.getBiases()).max();

    cout << "ConvLayer maxerr from ground truth:\nk1d: " << kmaxerr1d << "\nb1d: " << bmaxerr1d << "\nk2d: " << kmaxerr2d << "\nb2d: " << bmaxerr2d << endl;

}

TEST(RedFishTest, maxPooling)
{
    Tensor t({1, 1, 20});
    t.rand();
    MaxPool1dLayer layer(2, 1, 0);
    //cout << t << endl;
    //cout << layer.farward(t) << endl;
}
