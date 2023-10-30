#pragma once
#include "Tensor.h"
#include <limits>

namespace RedFish {

    class Loss {
    public:
        virtual double farward(const Tensor& prediction, const Tensor& ground_truth) const = 0;
        virtual Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const = 0;
    };



    class SquareLoss : public Loss {
    public:

        double farward(const Tensor& prediction, const Tensor& ground_truth) const override
        { return (prediction - ground_truth).squareSum(); }

        Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const override
        { return 2 * (prediction - ground_truth); }

        static const SquareLoss* get() { return &impl; }

    private:
        static SquareLoss impl;

    };
    inline SquareLoss SquareLoss::impl;



    class CrossEntropyLoss : public Loss {
    public:

        double farward(const Tensor& prediction, const Tensor& ground_truth) const override
        {
            double ret = 0.;
            for (size_t r = 0; r < prediction.rowSize(); r++)
            {
                size_t idx = (size_t)ground_truth(r);
                ret -= std::log(prediction(r, idx));
            }
            return ret / prediction.rowSize();
        }

        Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const override
        {
            Tensor ret({prediction.rowSize(), prediction.colSize()});

            float64 avg = 1. / prediction.rowSize();
            ret.zero();
            for (size_t r = 0; r < ret.rowSize(); r++)
            {
                size_t idx = (size_t)ground_truth(r, 0);
                ret(r, idx) = std::min(- avg / prediction(r, idx), std::numeric_limits<float64>::max());
            }
            return ret;
        }

        static const CrossEntropyLoss* get() { return &impl; }

    private:
        static CrossEntropyLoss impl;

    };
    inline CrossEntropyLoss CrossEntropyLoss::impl;

}