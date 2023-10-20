#pragma once 

#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include "Layer.h"
#include "Activation.h"
#include "Optimizer.h"

namespace RedFish {
    
    class Neuron {
    public:
        Neuron(int input_size, const Optimizer* optimizer) 
            : weights(input_size), bias((double)std::rand()/RAND_MAX - 0.5), opt(optimizer->instanziate(input_size + 1))
        {
            for (size_t i = 0; i < input_size; i++) 
                weights(i) = (double)std::rand()/RAND_MAX - 0.5;
        }
        Neuron(const Neuron& n) : weights(n.weights), bias(n.bias), opt(n.opt->instanziate(n.weights.getSize() + 1)) {}

        Algebra::Matrix farward(const Algebra::Matrix& X);
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001);

        Algebra::Matrix gradEst(const Algebra::Matrix& X);

        void print();

    private:
        Algebra::Matrix weights;
        double bias;

        std::unique_ptr<Optimizer> opt;

        friend class Model;
    };

    class LinearLayer : public Layer  {
    public:
        LinearLayer(int input_size, int neuron_size, Activation::AF af, const Optimizer* optimizer) 
            : neurons(), act_fn(af), af(af)
        {
            neurons.reserve(neuron_size);
            for (size_t i = 0; i < neuron_size; i++)
                neurons.emplace_back(input_size, optimizer);
        }

        Algebra::Matrix farward(const Algebra::Matrix& X);
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001);

    private:
        std::vector<Neuron> neurons;
        Activation::AF af;
        Activation::Function act_fn;
        Algebra::Matrix linout;
    
        friend class Model;
    };

    /* ---------- class Neuron ---------- */

    inline Algebra::Matrix Neuron::farward(const Algebra::Matrix& X)
    {
        return X * weights + bias;
    }

    inline Algebra::Matrix Neuron::backward(const Algebra::Matrix& X, const Algebra::Matrix &d, double learning_rate)
    {
        Algebra::Matrix dX = d * weights.T();
        auto grad = X.T() * d * (1./d.rows());
        double bias_der = d.sum() / d.rows();

        for (size_t i = 0; i < weights.getSize(); i++)
            weights(i) += opt->updateParameter(i, weights(i), grad(i), learning_rate);
        
        bias += opt->updateParameter(weights.getSize(), bias, bias_der, learning_rate);

        opt->step();

        return dX;
    }

    inline Algebra::Matrix Neuron::gradEst(const Algebra::Matrix &X)
    {
        return Algebra::Matrix();
    }

    inline void Neuron::print()
    {
        std::cout << "w = \n";
        weights.print();
        std::cout << "b = " << bias << "\n";
    }

    /* ---------- class Neuron ---------- */


    /* -------- class LinearLayer ------- */

    inline Algebra::Matrix LinearLayer::farward(const Algebra::Matrix &X)
    {
        linout = Algebra::Matrix(X.rows(), neurons.size());
        size_t col = 0;

        for (auto& neuron : neurons)
            linout.setCol(col++, neuron.farward(X));
        
        return act_fn.farward(linout);
    }

    inline Algebra::Matrix LinearLayer::backward(const Algebra::Matrix &X, const Algebra::Matrix &d, double learning_rate)
    {
        Algebra::Matrix dd = act_fn.backward(linout, d);

        Algebra::Matrix grad = neurons[0].backward(X, dd.getCol(0), learning_rate);

        for (size_t i = 1; i < neurons.size(); i++)
            grad += neurons[i].backward(X, dd.getCol(i), learning_rate);
        
        return grad;
    }

    /* -------- class LinearLayer ------- */
}
