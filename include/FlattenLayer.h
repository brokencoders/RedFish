#pragma once 

#include "Layer.h"

namespace RedFish {

    class FlattenLayer : public Layer 
    {
    public:
        FlattenLayer(size_t start_dim = 1, size_t end_dim = -1);
        FlattenLayer(std::ifstream& file);
        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;
        
    private:
        size_t start_dim;
        size_t end_dim;
    };
}