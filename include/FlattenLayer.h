#pragma once 

#include "Layer.h"

namespace RedFish {

    class FlattenLayer : public Layer 
    {
    public:
        FlattenLayer(size_t start_dim = 0, size_t end_dim = SIZE_MAX);
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;
        
    private:
        size_t start_dim;
        size_t end_dim;
    };
}