#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class MaxPool1dLayer : public Layer {
    public:
        MaxPool1dLayer(size_t kernel_size, size_t stride = 1, size_t padding = 0);
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    private:
        size_t kernel_size;
        size_t stride;
        size_t padding;
    };


    class MaxPool2dLayer : public Layer {
    public:
        MaxPool2dLayer();
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    private:
        size_t stride;
        size_t padding;
    };

    class MaxPool3dLayer : public Layer {
    public:
        MaxPool3dLayer();
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    private:
        size_t stride;
        size_t padding;
    };
}