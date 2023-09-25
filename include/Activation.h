#pragma once 

#include "Algebra.h"
#include <functional>

namespace RedFish::ActivationFn {

    class Identity
    {
    public:
        static double fn(double n) { return n; }
        static double bn(double n) { return 1.; }
    };

    class ReLU
    {
    public:
        static double fn(double n) { return n < 0. ? 0. : n; }
        static double bn(double n) { return n < 0. ? 0. : 1.; }
    };

    class LeakyReLU
    {
    public:
        static double fn(double n) { return n < 0. ? 0.01 * n : n; }
        static double bn(double n) { return n < 0. ? 0.01 : 1.; }
    };

    class Sigmoid
    {
    public:
        static double fn(double n) { return 1. / (1. + std::exp(-n)); }
        static double bn(double n) { return fn(n) * (1 - fn(n)); }
    };
    
    class TanH
    {
    public:
        static double fn(double n) { return (std::exp(n) - std::exp(-n)) / (std::exp(n) + std::exp(-n)); }
        static double bn(double n) { return 1 - std::pow(fn(n), 2); }
    };

    class Softplus
    {
    public:
        static double fn(double n) { return std::log(1 + std::exp(n)); }
        static double bn(double n) { return 1 / (1 + std::exp(-n)); }
    };

    class SiLU
    {
    public:
        static double fn(double n) { return n / (1 + std::exp(-n)); }
        static double bn(double n) { return (1 +std::exp(-n) + n * std::exp(-n)) / std::pow(1 + std::exp(-n), 2); }
    };
    
    class Gaussian
    {
    public:
        static double fn(double n) { return std::exp(-std::pow(n, 2)); }
        static double bn(double n) { return -2 * n * fn(n); }
    };
    
    class Softmax
    {
    public:
        static Algebra::Matrix fn(const Algebra::Matrix& n)
        {
            double sum = n.forEach(std::exp).sum();
            return n.forEach([sum](double x){ return std::exp(x) / sum; });
        }
        static Algebra::Matrix bn(const Algebra::Matrix& n)
        {
            Algebra::Matrix d(1, n.cols());
            Algebra::Matrix g = fn(n);

            zero(d);
            for (int i = 0; i < n.cols(); i++)
                for(int j = 0; j < n.cols(); j++)
                    d(0, i) += g(i) * ((j != i) - g(j));

            d /= n.cols();

            return d;
        }
    };

    template<typename ActFn>
    Algebra::Matrix farward(const Algebra::Matrix& X)
    {
        return X.forEach(ActFn::fn);
    }
    
    template<typename ActFn>
    Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d)
    {
        Algebra::Matrix ret(X.rows(), X.cols());

        for (size_t r = 0; r < X.rows(); r++)
            for (size_t c = 0; c < X.cols(); c++)
                ret(r,c) = d(r,c) * ActFn::bn(X(r,c));
        
        return ret;
    }

    template<>
    Algebra::Matrix farward<Softmax>(const Algebra::Matrix& X)
    {
        return X.forEachRow(Softmax::fn);
    }

    template<>
    Algebra::Matrix backward<Softmax>(const Algebra::Matrix& X, const Algebra::Matrix& d)
    {
        Algebra::Matrix grad(d.rows(), X.cols());

        for (size_t i = 0; i < d.rows(); i++)
            grad.setRow(i, Softmax::bn(X.getRow(i)));

        return grad;
    }
}