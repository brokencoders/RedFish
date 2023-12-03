#pragma once
#include <fstream>
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
        virtual void setLearningRate(float64) = 0;
        virtual uint64_t save(std::ofstream& file) const = 0;

    };

    enum : uint32_t {
        ADAM_OPTIMIZER
    };



    class Adam : public Optimizer
    {
    public:
        Adam();
        size_t allocateParameter(const Tensor& t) override;
        void updateParameter(size_t i, Tensor& value, const Tensor& grad) override;
        void step() override;
        void setLearningRate(float64 lr) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        std::vector<Tensor> mw, vw;
        float64 im1, im2, learning_rate;
        uint32_t t;
        static constexpr const float64 b1 = 0.9;
        static constexpr const float64 b2 = 0.999;
        static constexpr const float64 one_minus_b1 = 1 - b1;
        static constexpr const float64 one_minus_b2 = 1 - b2;
        static constexpr const float64 epsilon = 1e-8;
    };


    Optimizer* make_optimizer(uint32_t o);

}