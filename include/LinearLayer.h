#pragma once 

#include "Activation.h"
#include <iostream>
#include <cstdlib>
/* #include "Layer.h" */

namespace RedFish {
    
    class Neuron {
    public:
        Neuron(int input_size) 
            : weights(input_size), bias((double)std::rand()/RAND_MAX - 0.5), 
            mw(input_size), vw(input_size), mw_b(0), vw_b(0), t(1)
        {
            for (size_t i = 0; i < input_size; i++) 
                weights(i) = (double)std::rand()/RAND_MAX - 0.5;

            Algebra::zero(mw);
            Algebra::zero(vw);
        }

        Algebra::Matrix farward(const Algebra::Matrix& X);
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001);

        Algebra::Matrix gradEst(const Algebra::Matrix& X);

        void print();

    private:
        Algebra::Matrix weights;
        double bias;
        int t;

        Algebra::Matrix mw, vw;
        double mw_b, vw_b;
        const double b1 = 0.9;
        const double b2 = 0.999;
        const double epsilon = 1e-8;

        friend class Model;
    };

    class LinearLayer : public Layer  {
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

        mw = b1 * mw + (1 - b1) * grad;
        vw = b2 * vw + (1 - b2) * grad.forEach([](double d) { return d * d; }); 

        Algebra::Matrix m_hat = mw * (1 / (1 - std::pow(b1, t))); 
        Algebra::Matrix v_hat = vw * (1 / (1 - std::pow(b2, t)));

        weights -= learning_rate * (m_hat / (v_hat.forEach([](double d) { return std::sqrt(d); }) - epsilon));

        double bias_der = d.sum() / d.rows();

        mw_b = b1 * mw_b + (1 - b1) * bias_der;
        vw_b = b2 * vw_b + (1 - b2) * std::pow(bias_der, 2); 

        double m_hat_b = mw_b * (1 / (1 - std::pow(b1, t))); 
        double v_hat_b = vw_b * (1 / (1 - std::pow(b2, t)));

        bias    -= learning_rate * m_hat_b / ( std::sqrt(vw_b) - epsilon);

        t++;

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
