#include "Loss.h"

namespace RedFish {

    double SquareLoss::farward(const Tensor& prediction, const Tensor& ground_truth) const
    {
        return (prediction - ground_truth).squareSum();
    }

    Tensor SquareLoss::backward(const Tensor& prediction, const Tensor& ground_truth) const
    {
        return 2 * (prediction - ground_truth);
    }

    uint64_t SquareLoss::save(std::ofstream &file) const
    {
        const char name[] = "Loss::Square";
        file.write(name, sizeof(name));
        uint64_t i = 0;
        file.write((char*)&i, sizeof(i));
        return sizeof(name) + sizeof(i);
    }

    double CrossEntropyLoss::farward(const Tensor& prediction, const Tensor& ground_truth) const
    {
        double ret = 0.;
        for (size_t r = 0; r < prediction.rowSize(); r++)
        {
            size_t idx = (size_t)ground_truth(r);
            ret -= std::log(prediction(r, idx));
        }
        return ret / prediction.rowSize();
    }

    Tensor CrossEntropyLoss::backward(const Tensor& prediction, const Tensor& ground_truth) const
    {
        Tensor ret({prediction.rowSize(), prediction.colSize()});

        float64 avg = 1. / prediction.rowSize();
        ret.zero();
        for (size_t r = 0; r < ret.rowSize(); r++)
        {
            size_t idx = (size_t)ground_truth(r, (size_t)0);
            ret(r, idx) = std::min(- avg / prediction(r, idx), std::numeric_limits<float64>::max());
        }
        return ret;
    }

    uint64_t CrossEntropyLoss::save(std::ofstream &file) const
    {
        const char name[] = "Loss::CrossEntropy";
        file.write(name, sizeof(name));
        uint64_t i = 0;
        file.write((char*)&i, sizeof(i));
        return sizeof(name) + sizeof(i);
    }

    Loss* make_loss(uint32_t l)
    {
        switch (l)
        {
        case SQUARE_LOSS:        return new SquareLoss();
        case CROSS_ENTROPY_LOSS: return new CrossEntropyLoss();
                
        default: return nullptr;
        }
    }

}