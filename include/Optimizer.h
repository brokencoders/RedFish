#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"

namespace RedFish
{

    class Optimizer
    {
    public:
        Optimizer() {}
        Optimizer(const Optimizer&) = delete;
        virtual ~Optimizer() {}
        virtual size_t allocateParameter(const Tensor&) = 0;
        virtual void updateParameter(size_t i, Tensor& value, const Tensor& grad) = 0;
        virtual void step() = 0;
    };



    class Adam : public Optimizer
    {
    public:
        Adam(double learning_rate) : mw(), vw(), im1(1 / (1 - b1)), im2(1 / (1 - b2)), learning_rate(learning_rate), t(1) {}
        size_t allocateParameter(const Tensor& t) override
        {
            mw.emplace_back(empty_like(t));
            vw.emplace_back(empty_like(t));
            return mw.size() - 1;
        }
        void updateParameter(size_t i, Tensor& value, const Tensor& grad) override
        {
            mw[i] *= b1;
            vw[i] *= b2;
            mw[i] += grad      * one_minus_b1;
            vw[i] += grad*grad * one_minus_b2; 

            Tensor m_hat = mw[i] * im1; 
            Tensor v_hat = vw[i] * im2;

            value -= learning_rate * m_hat / (std::sqrt(v_hat) - epsilon);
        }
        virtual void step() override
        {
            t++;
            im1 = 1 / (1 - std::pow(b1, t));
            im2 = 1 / (1 - std::pow(b2, t));
        }

    private:
        std::vector<Tensor> mw, vw;
        double im1, im2, learning_rate;
        uint32_t t;
        static constexpr const double b1 = 0.9;
        static constexpr const double b2 = 0.999;
        static constexpr const double one_minus_b1 = 1 - b1;
        static constexpr const double one_minus_b2 = 1 - b2;
        static constexpr const double epsilon = 1e-8;
    };

}