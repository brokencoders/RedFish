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

    enum OPTIMIZER : uint32_t {
        ADAM_OPT,
        SGD_OPT,
    };

    class SGD : public Optimizer
    {
    public:
        SGD(float64 weight_decay = 0, float64 momentum = 0, float64 dampening = 0, bool nesterov = false);
        SGD(std::ifstream& file);
        size_t allocateParameter(const Tensor& t) override;
        void updateParameter(size_t i, Tensor& value, const Tensor& grad) override;
        void step() override;
        void setLearningRate(float64 lr) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        std::vector<Tensor> b;
        float64 weight_decay, momentum, dampening, learning_rate;
        bool nesterov;
        uint32_t t;
    };

    class Adam : public Optimizer
    {
    public:
        Adam();
        Adam(std::ifstream& file);
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


    template <OPTIMIZER o, typename... Args>
    Optimizer* make_optimizer(Args... args)
    {
        if constexpr (o == OPTIMIZER::ADAM_OPT) return new Adam(args...);
        if constexpr (o == OPTIMIZER::SGD_OPT)  return new SGD(args...);
        return nullptr;
    }
    
    Optimizer* make_optimizer(const uint32_t o);
    Optimizer* make_optimizer(std::ifstream& file);

}