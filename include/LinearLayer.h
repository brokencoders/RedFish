#pragma once 

#include "Algebra.h"
#include "Activation.h"
#include <array>
#include <iostream>

using namespace Algebra;

namespace RedFish {
    
    template<int I>
    class Neuron {
    public:
        Neuron() : weights(I), bias(random()/1073741823.5) {for (size_t i = 0; i < I; i++) weights(i) = random()/1073741823.5;}

        Matrix farward(const Matrix& X);
        Matrix backward(const Matrix& X, const Matrix& d, double learning_rate = 0.001);

        void print();

    private:
        Matrix weights;
        double bias;
    };


    template<int I, int N, typename ActFn>
    class LinearLayer {
    public:
        LinearLayer() {}

        Matrix farward(const Matrix& X);
        Matrix backward(const Matrix& X, const Matrix& d, double learning_rate = 0.001);

    private:
        std::array<Neuron<I>, N> neurons;
        Matrix linout;
    };

    /* ---------- class Neuron ---------- */

    template <int N>
    inline Matrix Neuron<N>::farward(const Matrix& X)
    {
        return X * weights + bias;
    }

    template <int N>
    inline Matrix Neuron<N>::backward(const Matrix& X, const Matrix &d, double learning_rate)
    {
        auto grad = X.T() * d;

        weights -= learning_rate * grad;
        bias    -= learning_rate * d.sum();

        return d * weights.T();
    }

    template <int N>
    void Neuron<N>::print()
    {
        std::cout << "w = \n";
        weights.print();
        std::cout << "b = " << bias << "\n";
    }

    /* ---------- class Neuron ---------- */


    /* -------- class LinearLayer ------- */

    template <int I, int N, typename ActFn>
    Matrix LinearLayer<I, N, ActFn>::farward(const Matrix &X)
    {
        linout = Matrix(X.rows(), neurons.size());
        size_t col = 0;

        for (auto& neuron : neurons)
            linout.setCol(col++, neuron.farward(X));
        
        return ActivationFn::farward<ActFn>(linout);
    }

    template <int I, int N, typename ActFn>
    Matrix LinearLayer<I, N, ActFn>::backward(const Matrix &X, const Matrix &d, double learning_rate)
    {
        Matrix dd = ActivationFn::backward<ActFn>(linout, d);

        Matrix grad = neurons[0].backward(X, dd.getCol(0), learning_rate);

        for (size_t i = 1; i < neurons.size(); i++)
            grad += neurons[i].backward(X, dd.getCol(i), learning_rate);
        
        grad *= 1./neurons.size();

        return grad;
    }

    /* -------- class LinearLayer ------- */
}
