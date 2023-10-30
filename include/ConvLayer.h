#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class ConvLayer : public Layer {
    public:
        ConvLayer(std::vector<size_t> kernel_size, size_t stride, bool padding)
            : kernels(kernel_size), stride(stride), padding(padding)
        {}

    private:
        Tensor kernels;
        size_t stride;
        bool padding;

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    
        friend class Model;
    };


    inline Tensor ConvLayer::farward(const Tensor& X)
    {
        return Tensor();
    }

    inline Tensor ConvLayer::backward(const Tensor& X, const Tensor& d) 
    {
        return Tensor();
    }
}