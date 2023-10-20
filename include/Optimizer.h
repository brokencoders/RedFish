#pragma once
#include <iostream>
#include <memory>
#include "Algebra.h"

namespace RedFish
{

    class Optimizer
    {
    public:
        Optimizer() {}
        Optimizer(const Optimizer&) = delete;
        virtual ~Optimizer() {}
        virtual double updateParameter(size_t i, double value, double grad, double learning_rate) = 0;
        virtual void step() = 0;
        virtual std::unique_ptr<Optimizer> instanziate(size_t size) const = 0;
    };



    class Adam : public Optimizer
    {
    public:
        Adam(size_t size) : mw(size, 0.), vw(size, 0.), im1(1 / (1 - b1)), im2(1 / (1 - b2)), t(1) {}
        double updateParameter(size_t i, double value, double grad, double learning_rate) override
        {
            mw[i] = b1 * mw[i] + one_minus_b1 * grad;
            vw[i] = b2 * vw[i] + one_minus_b2 * grad*grad; 

            double m_hat = mw[i] * im1; 
            double v_hat = vw[i] * im2;

            return -learning_rate * m_hat / ( std::sqrt(v_hat) - epsilon);
        }
        virtual void step()
        {
            t++;
            im1 = 1 / (1 - std::pow(b1, t));
            im2 = 1 / (1 - std::pow(b2, t));
        }
        std::unique_ptr<Optimizer> instanziate(size_t size) const { return std::make_unique<Adam>(size); };
        static const Adam* get() { return &impl; }

    private:
        std::vector<double> mw, vw;
        double im1, im2;
        uint32_t t;
        static constexpr const double b1 = 0.9;
        static constexpr const double b2 = 0.999;
        static constexpr const double one_minus_b1 = 1 - b1;
        static constexpr const double one_minus_b2 = 1 - b2;
        static constexpr const double epsilon = 1e-8;

        static const Adam impl;
    };
    inline const Adam Adam::impl(0);

}