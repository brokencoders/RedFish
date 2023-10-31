#include "LinearLayer.h"

namespace RedFish {
    
    LinearLayer::LinearLayer(size_t input_size, size_t neuron_count, Optimizer* optimizer) 
        : weights({input_size, neuron_count}), biases({neuron_count}), optimizer(optimizer)
    {
        weights.rand(-.5, .5);
        biases.rand(-.5, .5);
        w_id = optimizer->allocateParameter(weights);
        b_id = optimizer->allocateParameter(biases);
    }

    Tensor LinearLayer::farward(const Tensor &X)
    {
        return X.matmul(weights) + biases;
    }

    Tensor LinearLayer::backward(const Tensor &X, const Tensor &d)
    {
        const float64 lambda = .001;
        Tensor dX = d.matmul(weights.T());
        Tensor grad = X.T().matmul(d) + weights * lambda;
        Tensor bias_grad = d.sum((size_t)0);

        optimizer->updateParameter(w_id, weights, grad);
        optimizer->updateParameter(b_id, biases, bias_grad);

        optimizer->step();

        return dX;
    }

    void LinearLayer::print()
    {
        std::cout << "w = \n" << weights << "b = " << biases << "\n";
    }

}
