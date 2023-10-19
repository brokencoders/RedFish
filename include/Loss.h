#pragma once
#include "Algebra.h"
#include <limits>

namespace RedFish {

    class Loss {
    public:
        virtual double farward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const = 0;
        virtual Algebra::Matrix backward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const = 0;
    };



    class SquareLoss : public Loss {
    public:

        double farward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const override
        { return (prediction - ground_truth).normSquare(); }

        Algebra::Matrix backward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const override
        { return 2 * (prediction - ground_truth); }

        static const SquareLoss* get() { return &impl; }

    private:
        static SquareLoss impl;

    };
    inline SquareLoss SquareLoss::impl;



    class CrossEntropyLoss : public Loss {
    public:

        double farward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const override
        {
            double ret = 0.;
            for (size_t r = 0; r < prediction.rows(); r++)
            {
                size_t idx = (size_t)ground_truth(r, 0);
                ret -= std::log(prediction(r, idx));
            }
            return ret / prediction.rows();
//            return -(ground_truth * prediction.forEach([](Algebra::float64 n) { return std::min(std::numeric_limits<Algebra::float64>::max(), std::log(n)); }).T()).sum();
        }

        Algebra::Matrix backward(const Algebra::Matrix& prediction, const Algebra::Matrix& ground_truth) const override
        {
            Algebra::Matrix ret(prediction.rows(), prediction.cols());

            Algebra::float64 avg = 1. / prediction.rows();
            zero(ret);
            for (size_t r = 0; r < ret.rows(); r++)
            {
                size_t idx = (size_t)ground_truth(r, 0);
                ret(r, idx) = std::min(- avg / prediction(r, idx), std::numeric_limits<Algebra::float64>::max());
            }
            return ret;
        }

        static const CrossEntropyLoss* get() { return &impl; }

    private:
        static CrossEntropyLoss impl;

    };
    inline CrossEntropyLoss CrossEntropyLoss::impl;

}