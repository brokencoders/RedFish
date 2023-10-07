#pragma once 

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
        std::vector<Algebra::Matrix> kernels;

        Algebra::Matrix farward(const Algebra::Matrix& X) override;
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001) override;
    
        friend class Model;
    };


    inline Algebra::Matrix ConvLayer::farward(const Algebra::Matrix& X)
    {
        for (size_t i = 0; i < kernel_count; i++)
        {
            
        }
        
    }

    inline Algebra::Matrix ConvLayer::backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate) 
    {
        
    }
}