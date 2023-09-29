#pragma once 

#include "Algebra.h"
#include "Activation.h"
#include <iostream>

namespace RedFish {
    
    class Neuron {
    public:
        Neuron(int input_size) : weights(input_size), bias(random()/2147483647. - 0.5) {for (size_t i = 0; i < input_size; i++) weights(i) = random()/2147483647. - 0.5;}

        Algebra::Matrix farward(const Algebra::Matrix& X);
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001);

        Algebra::Matrix gradEst(const Algebra::Matrix& X);

        void print();

    private:
        Algebra::Matrix weights;
        double bias;

        friend class Model;
    };

    class LinearLayer {
    public:
        LinearLayer(int input_size, int neuron_size, Activation::AF af) 
            : neurons(), act_fn(af), af(af)
        {
            neurons.reserve(neuron_size);
            for (size_t i = 0; i < neuron_size; i++)
                neurons.emplace_back(input_size);
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

        weights -= learning_rate * grad;
        bias    -= learning_rate * d.sum() / d.rows();

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
        
        grad *= 1./neurons.size();

        return grad;
    }

    /* -------- class LinearLayer ------- */
}
