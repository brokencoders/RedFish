#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class Conv1dLayer : public Layer {
    public:
        Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0, size_t dilation = 1, PaddingMode pm = ZERO);

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

    private:
        Tensor kernels, bias;
        size_t stride;
        size_t padding;
        size_t dilation;

    };
    
    class Conv2dLayer : public Layer {
    public:
        Conv2dLayer(size_t in_channels, size_t out_channels, Tuple2d kernel_size, Tuple2d stride = 1, Tuple2d padding = 0, Tuple2d dilation = 1, PaddingMode pm = ZERO);

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    
    private:
        Tensor kernels, bias;
        size_t stride;
        size_t padding;
        size_t dilation;

    };
    
    class Conv3dLayer : public Layer {
    public:
        Conv3dLayer(size_t in_channels, size_t out_channels, Tuple3d kernel_size, Tuple3d stride = 1, Tuple3d padding = 0, Tuple3d dilation = 1, PaddingMode pm = ZERO);

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

    private:
        Tensor kernels, bias;
        size_t stride;
        size_t padding;
        size_t dilation;

    };
}