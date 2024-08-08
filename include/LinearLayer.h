#pragma once 

#include "Layer.h"
#include "Optimizer.h"
#include "Tensor.h"

namespace RedFish {
    
    class LinearLayer : public Layer  {
    public:
        LinearLayer(size_t input_size, size_t output_size);
        LinearLayer(std::ifstream& file);
        void useOptimizer(Optimizer& optimizer) override;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        Tensor W, b;
        size_t W_id, b_id;
    };
    
}
