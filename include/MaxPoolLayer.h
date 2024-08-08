#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class MaxPool1dLayer : public Layer {
    public:
        MaxPool1dLayer(size_t kernel_size, size_t stride = 0, size_t padding = 0, size_t dilation = 1);
        MaxPool1dLayer(std::ifstream& file);
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;
        
    private:
        size_t kernel_size;
        size_t stride;
        size_t padding;
        size_t dilation;
    };


    class MaxPool2dLayer : public Layer {
    public:
        MaxPool2dLayer(TupleNd<2> kernel_size, TupleNd<2> stride = 0, TupleNd<2> padding = 0, TupleNd<2> dilation = 1);
        MaxPool2dLayer(std::ifstream& file);
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        TupleNd<2> kernel_size;
        TupleNd<2> stride;
        TupleNd<2> padding;
        TupleNd<2> dilation;
    };

    class MaxPool3dLayer : public Layer {
    public:
        MaxPool3dLayer(TupleNd<3> kernel_size, TupleNd<3> stride = 0, TupleNd<3> padding = 0, TupleNd<3> dilation = 1);
        MaxPool3dLayer(std::ifstream& file);
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        TupleNd<3> kernel_size;
        TupleNd<3> stride;
        TupleNd<3> padding;
        TupleNd<3> dilation;
    };
}