#pragma once 

#include <iostream>
#include "Layer.h"
#include "Tensor.h"

namespace RedFish::Activation
{

    Layer* new_Activation(std::ifstream& file);
    
    class Identity : public Layer  {

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class ReLU : public Layer  {
        static double relu(double n);
        static double relu_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class LeakyReLU : public Layer  {
        static double lrelu(double n);
        static double lrelu_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class PReLU : public Layer  {
        static double prelu(double n, double a);
        static double prelu_d(double n, double a);

        std::function<double(double)> prelua, prelua_d;

    public:
        PReLU(float64 initial = .25);
        PReLU(std::ifstream& file, Optimizer* optimizer);

        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class Sigmoid : public Layer  {
        static double sigmoid(double n);
        static double sigmoid_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class TanH : public Layer  {
        static double tanh(double n);
        static double tanh_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class Softplus : public Layer  {
        static double softplus(double n);
        static double softplus_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class SiLU : public Layer  {
        static double silu(double n);
        static double silu_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class Gaussian : public Layer  {
        static double gaussian(double n);
        static double gaussian_d(double n);

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

    class Softmax : public Layer  {

    public:
        Tensor farward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

    };

}
