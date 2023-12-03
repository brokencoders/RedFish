#pragma once 

#include "Layer.h"

namespace RedFish 
{
    class DropoutLayer : public Layer 
    {
    public:
        DropoutLayer(float64 rate);
        Tensor farward(const Tensor& X);
        Tensor backward(const Tensor& X, const Tensor& d);
        uint64_t save(std::ofstream& file) const override;
        
    private:
        Tensor output;
        float64 rate;
        size_t skip_size;
        size_t batch_size;
        float64 factor;
    };
}