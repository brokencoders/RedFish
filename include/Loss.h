#pragma once
#include "Tensor.h"
#include <limits>
#include <fstream>

namespace RedFish {

    class Loss {
    public:
        virtual double farward(const Tensor& prediction, const Tensor& ground_truth) const = 0;
        virtual Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const = 0;
        virtual uint64_t save(std::ofstream& file) const = 0;
    };

    enum : uint32_t {
        SQUARE_LOSS,
        CROSS_ENTROPY_LOSS
    };


    class SquareLoss : public Loss {
    public:

        double farward(const Tensor& prediction, const Tensor& ground_truth) const override;
        Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const override;
        uint64_t save(std::ofstream& file) const override;
    };



    class CrossEntropyLoss : public Loss {
    public:

        double farward(const Tensor& prediction, const Tensor& ground_truth) const override;
        Tensor backward(const Tensor& prediction, const Tensor& ground_truth) const override;
        uint64_t save(std::ofstream& file) const override;
    };
    
    Loss* make_loss(uint32_t l);
    Loss* make_loss(std::ifstream& file);

}