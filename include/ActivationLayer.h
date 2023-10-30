#pragma once 

#include <iostream>
#include "Layer.h"
#include "Tensor.h"

namespace RedFish::Activation
{
    
    class Identity : public Layer  {

    public:
        Tensor farward(const Tensor& X) override { return X; }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d; }

    };

    class ReLU : public Layer  {
        static double relu(double n)   { return n < 0. ? 0. : n; }      /* { return (n >= 0.) * n; } */
        static double relu_d(double n) { return n < 0. ? 0. : 1.; }     /* { return (double)(n >= 0.); } */

    public:
        Tensor farward(const Tensor& X) override { return forEach<relu>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<relu_d>(X); }

    };

    class LeakyReLU : public Layer  {
        static double lrelu(double n)   { return n < 0. ? 0.01 * n : n; }   /* { return ((n < 0.) * .01 + (n >= 0.)) * n; } */
        static double lrelu_d(double n) { return n < 0. ? 0.01 : 1.; }      /* { return (n < 0.) * .01  + (n >= 0.) * 1.; } */

    public:
        Tensor farward(const Tensor& X) override { return forEach<lrelu>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<lrelu_d>(X); }

    };

    class PReLU : public Layer  {
        static double prelu(double n, double a)   { return n < 0. ? a * n : n; }    /* { return ((n < 0.) * a + (n >= 0.)) * n; } */
        static double prelu_d(double n, double a) { return n < 0. ? a : 1.; }       /* { return (n < 0.) * a  + (n >= 0.) * 1.; } */

        std::function<double(double)> prelua, prelua_d;

    public:
        PReLU(double a)
            : prelua(std::bind(prelu, std::placeholders::_1, a)),
              prelua_d(std::bind(prelu_d, std::placeholders::_1, a))
        {}

        Tensor farward(const Tensor& X) override { return forEach(X, prelua); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach(X, prelua_d); }

    };

    class Sigmoid : public Layer  {
        static double sigmoid(double n)   { return 1. / (1. + std::exp(-n)); }
        static double sigmoid_d(double n) { double sig_n = sigmoid(n); return sig_n * (1 - sig_n); }

    public:
        Tensor farward(const Tensor& X) override { return forEach<sigmoid>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<sigmoid_d>(X); }

    };

    class TanH : public Layer  {
        static double tanh(double n)   { double exp_n = std::exp(n), exp_m_n = 1. / exp_n; return (exp_n - exp_m_n) / (exp_n + exp_m_n); }
        static double tanh_d(double n) { return 1 - std::pow(tanh(n), 2); }

    public:
        Tensor farward(const Tensor& X) override { return forEach<tanh>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<tanh_d>(X); }

    };

    class Softplus : public Layer  {
        static double softmax(double n)   { return std::log(1 + std::exp(n)); }
        static double softmax_d(double n) { return 1 / (1 + std::exp(-n)); }

    public:
        Tensor farward(const Tensor& X) override { return forEach<softmax>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<softmax_d>(X); }

    };

    class SiLU : public Layer  {
        static double silu(double n)   { return n / (1 + std::exp(-n)); }
        static double silu_d(double n) { auto exp_n = std::exp(-n), one_exp_n = 1 + exp_n; return (one_exp_n + n * exp_n) / std::pow(one_exp_n, 2); }

    public:
        Tensor farward(const Tensor& X) override { return forEach<silu>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<silu_d>(X); }

    };

    class Gaussian : public Layer  {
        static double gaussian(double n)   { return std::exp(-n*n); }
        static double gaussian_d(double n) { return -2 * n * gaussian(n); }

    public:
        Tensor farward(const Tensor& X) override { return forEach<gaussian>(X); }
        Tensor backward(const Tensor& X, const Tensor& d) override { return d * forEach<gaussian_d>(X); }

    };

    class Softmax : public Layer  {

    public:
        Tensor farward(const Tensor& X) override
        {
            Tensor max = X.max(0);
            Tensor expv = std::exp(X - max);
            Tensor sum = expv.sum(0);
            Tensor div = 1. / sum;
            expv *= div;
            return expv;
        }

        Tensor backward(const Tensor& X, const Tensor& d) override
        {
            Tensor grad({d.rowSize(), X.colSize()});
            Tensor dX({X.colSize()});
            Tensor g = farward(X);

            for (size_t r = 0, rs = X.rowSize(), cs = X.colSize(); r < rs; r++) {

                float64 g_i = g(r, 0);
                float64 d_r_i = d(r, 0);
                grad(r, 0) = d_r_i * g_i * (1. - g_i);
                for(size_t j = 1; j < cs; j++)
                {
                    float64 dX = -g_i*g(r, j);
                    grad(r, 0) += d(r, j) * dX;
                    grad(r, j)  = d_r_i   * dX;
                }
                for (size_t i = 1; i < cs; i++)
                {
                    g_i = g(r, i);
                    d_r_i = d(r, i);
                    grad(r, i) += d_r_i * g_i * (1. - g_i);
                    for(size_t j = i + 1; j < cs; j++)
                    {
                        float64 dX = -g_i*g(r, j);
                        grad(r, i) += d(r, j) * dX;
                        grad(r, j) += d_r_i   * dX;
                    }
                }

                //grad.setRow(r, d.getRow(r).matmul(dX));
            }

            return grad;
        }

    };

}
