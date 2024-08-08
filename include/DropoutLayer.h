#pragma once 

#include "Layer.h"

namespace RedFish 
{
    class DropoutLayer : public Layer 
    {
    public:
        DropoutLayer(float64 rate);
        DropoutLayer(std::ifstream& file);
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;
        
    private:
        Tensor mask;
        float64 rate;
    };
}