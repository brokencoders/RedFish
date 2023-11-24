#pragma once 

#include "Layer.h"

namespace RedFish 
{
    class DropoutLayer : public Layer 
    {
    public:
        DropoutLayer(float64 rate, std::vector<size_t> shape);
        Tensor farward(const Tensor& X);
        Tensor backward(const Tensor& X, const Tensor& d);
    private:
        float64 rate;
        std::vector<size_t> shape;
    };
}