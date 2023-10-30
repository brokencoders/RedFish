#pragma once 

#include "Layer.h"
#include "Optimizer.h"
#include "Tensor.h"

namespace RedFish {
    
    class LinearLayer : public Layer  {
    public:
        LinearLayer(size_t input_size, size_t neuron_count, Optimizer* optimizer) 
            : weights({input_size, neuron_count}), biases({neuron_count}), optimizer(optimizer)
        {
            weights.rand(-.5, .5);
            biases.rand(-.5, .5);
            w_id = optimizer->allocateParameter(weights);
            b_id = optimizer->allocateParameter(biases);
        }

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

        void print();

    private:
        Tensor weights, biases;
        size_t w_id, b_id;
        Optimizer* optimizer;
    };

    /* -------- class LinearLayer ------- */

    inline Tensor LinearLayer::farward(const Tensor &X)
    {
        return X.matmul(weights) + biases;
    }

    inline Tensor LinearLayer::backward(const Tensor &X, const Tensor &d)
    {
        Tensor dX = d.matmul(weights.T());
        Tensor grad = X.T().matmul(d) * (1./d.rowSize());
        Tensor bias_grad = d.sum((size_t)0) / d.rowSize();

        optimizer->updateParameter(w_id, weights, grad);
        optimizer->updateParameter(b_id, biases, bias_grad);

        optimizer->step();

        return dX;
    }

    inline void LinearLayer::print()
    {
        std::cout << "w = \n" << weights << "b = " << biases << "\n";
    }

    /* -------- class LinearLayer ------- */
}
