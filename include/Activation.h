#pragma once 

#include "Algebra.h"
#include <functional>

namespace RedFish::Activation {

    namespace Private {

        class Identity
        {
        public:
            inline static double fn(double n) { return n; }
            inline static double bn(double n) { return 1.; }
        };

        class ReLU
        {
        public:
            inline static double fn(double n) { return n < 0. ? 0. : n; }
            inline static double bn(double n) { return n < 0. ? 0. : 1.; }
        };

        class LeakyReLU
        {
        public:
            inline static double fn(double n) { return n < 0. ? 0.01 * n : n; }
            inline static double bn(double n) { return n < 0. ? 0.01 : 1.; }
        };

        class Sigmoid
        {
        public:
            inline static double fn(double n) { return 1. / (1. + std::exp(-n)); }
            inline static double bn(double n) { return fn(n) * (1 - fn(n)); }
        };
        
        class TanH
        {
        public:
            inline static double fn(double n) { return (std::exp(n) - std::exp(-n)) / (std::exp(n) + std::exp(-n)); }
            inline static double bn(double n) { return 1 - std::pow(fn(n), 2); }
        };

        class Softplus
        {
        public:
            inline static double fn(double n) { return std::log(1 + std::exp(n)); }
            inline static double bn(double n) { return 1 / (1 + std::exp(-n)); }
        };

        class SiLU
        {
        public:
            inline static double fn(double n) { return n / (1 + std::exp(-n)); }
            inline static double bn(double n) { return (1 +std::exp(-n) + n * std::exp(-n)) / std::pow(1 + std::exp(-n), 2); }
        };
        
        class Gaussian
        {
        public:
            inline static double fn(double n) { return std::exp(-std::pow(n, 2)); }
            inline static double bn(double n) { return -2 * n * fn(n); }
        };
        
        class Softmax
        {
        public:
            
            inline static Algebra::Matrix fn(const Algebra::Matrix& n)
            {
                double max = n.max();
                double sum = n.forEach([max](double x){ return std::exp(x - max); }).sum();
                double offset = max + log(sum);
                return n.forEach([offset](double x){ return std::exp(x - offset); });
            }

            inline static Algebra::Matrix bn(const Algebra::Matrix& n)
            {
                Algebra::Matrix d(1, n.cols());
                Algebra::Matrix g = fn(n);

                zero(d);
                for (int i = 0; i < n.cols(); i++)
                    for(int j = 0; j < n.cols(); j++)
                        d(0, i) += g(j) * ((j != i) - g(i));

                d /= n.cols();

                return d;
            }
        };

    }

    enum AF : uint32_t {
        Identity = 0,
        ReLU,
        LeakyReLU,
        Sigmoid,
        TanH,
        Softplus,
        SiLU, 
        Gaussian,
        Softmax
    };

    class Function {
    public:
        Function(AF activation_function)
        {
            using namespace Private;
            farward_ = &RedFish::Activation::Function::farwardStd;
            backward_ = &RedFish::Activation::Function::backwardStd;
            switch (activation_function)
            {
            case AF::Identity:
                fn = Identity::fn, bn = Identity::bn;
                break;
            case AF::ReLU:
                fn = ReLU::fn, bn = ReLU::bn;
                break;
            case AF::LeakyReLU:
                fn = LeakyReLU::fn, bn = LeakyReLU::bn;
                break;
            case AF::Sigmoid:
                fn = Sigmoid::fn, bn = Sigmoid::bn;
                break;
            case AF::TanH:
                fn = TanH::fn, bn = TanH::bn;
                break;
            case AF::Softplus:
                fn = Softplus::fn, bn = Softplus::bn;
                break;
            case AF::SiLU:
                fn = SiLU::fn, bn = SiLU::bn;
                break;
            case AF::Gaussian:
                fn = Gaussian::fn, bn = Gaussian::bn;
                break;
            case AF::Softmax:
                fn = nullptr, bn = nullptr;
                farward_ = &RedFish::Activation::Function::farwardSoftmax;
                backward_ = &RedFish::Activation::Function::backwardSoftmax;
                break;
            default:
                fn = nullptr, bn = nullptr;
                farward_ = nullptr, backward_ = nullptr;
                break;
            }
        }

        Algebra::Matrix farward(const Algebra::Matrix& X) { return (this->*farward_)(X); }
        Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d) { return (this->*backward_)(X, d); }

    private:

        Algebra::Matrix(Function::*farward_)(const Algebra::Matrix&);
        Algebra::Matrix(Function::*backward_)(const Algebra::Matrix&, const Algebra::Matrix&);
        double(*fn)(double);
        double(*bn)(double);

        inline Algebra::Matrix farwardStd(const Algebra::Matrix& X)
        {
            return X.forEach(fn);
        }
        
        inline Algebra::Matrix backwardStd(const Algebra::Matrix& X, const Algebra::Matrix& d)
        {
            Algebra::Matrix ret(X.rows(), X.cols());

            for (size_t r = 0; r < X.rows(); r++)
                for (size_t c = 0; c < X.cols(); c++)
                    ret(r,c) = d(r,c) * bn(X(r,c));
            
            return ret;
        }

        inline Algebra::Matrix farwardSoftmax(const Algebra::Matrix& X)
        {
            return X.forEachRow(Private::Softmax::fn);
        }

        inline Algebra::Matrix backwardSoftmax(const Algebra::Matrix& X, const Algebra::Matrix& d)
        {
            // return d;

            Algebra::Matrix grad(d.rows(), X.cols());

            for (size_t i = 0; i < d.rows(); i++)
                grad.setRow(i, Private::Softmax::bn(X.getRow(i)));

            return grad;
        }
    };

}