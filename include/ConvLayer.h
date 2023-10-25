#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class ConvLayer : public Layer {
    public:
        ConvLayer(size_t kernel_count, size_t kernel_width, size_t kernel_height, size_t stride, bool padding)
            : kernel_count(kernel_count), kernel_width(kernel_width), kernel_height(kernel_height), stride(stride), padding(padding)
        {
            for (size_t i = 0; i < kernel_count; i++)
            {
                kernels.emplace_back(kernel_width, kernel_height);
            }
            
        }

    private:
        size_t stride;
        size_t kernel_count;
        size_t kernel_width;
        size_t kernel_height;
        bool padding;
        std::vector<Tensor> kernels;

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
    
        friend class Model;
    };


    inline Tensor ConvLayer::farward(const Tensor& X)
    {
        for (size_t i = 0; i < kernel_count; i++)
        {
            
        }
        
    }

    inline Tensor ConvLayer::backward(const Tensor& X, const Tensor& d) 
    {
        
    }
}