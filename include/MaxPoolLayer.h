#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class MaxPool1dLayer : public Layer {
    public:
        MaxPool1dLayer(size_t kernel_size, size_t stride = 1, size_t padding = 0,  size_t dilation = 1);
        MaxPool1dLayer(std::ifstream& file);
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;
        
    private:
        size_t kernel_size;
        size_t stride;
        size_t padding;
        size_t dilation;
    };


    class MaxPool2dLayer : public Layer {
    public:
        MaxPool2dLayer(Tuple2d kernel_size, Tuple2d stride = 1, Tuple2d padding = 0,  Tuple2d dilation = 1);
        MaxPool2dLayer(std::ifstream& file);
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        Tuple2d kernel_size;
        Tuple2d stride;
        Tuple2d padding;
        Tuple2d dilation;
    };

    class MaxPool3dLayer : public Layer {
    public:
        MaxPool3dLayer(Tuple3d kernel_size, Tuple3d stride = 1, Tuple3d padding = 0,  Tuple3d dilation = 1);
        MaxPool3dLayer(std::ifstream& file);
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        Tuple3d kernel_size;
        Tuple3d stride;
        Tuple3d padding;
        Tuple3d dilation;
    };
}