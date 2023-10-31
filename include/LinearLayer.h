#pragma once 

#include "Layer.h"
#include "Optimizer.h"
#include "Tensor.h"

namespace RedFish {
    
    class LinearLayer : public Layer  {
    public:
        LinearLayer(size_t input_size, size_t neuron_count, Optimizer* optimizer);

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

        void print();

    private:
        Tensor weights, biases;
        size_t w_id, b_id;
        Optimizer* optimizer;
    };
    
}
