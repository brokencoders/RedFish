#include "ActivationLayer.h"
#include <string>

namespace RedFish::Activation
{

    Layer *new_Activation(std::ifstream &file)
    {
        return nullptr;
    }

    uint64_t save_act(std::ofstream &file, std::string name)
    {
        name = "Layer::Activation::" + name;
        file.write(name.c_str(), name.size() + 1);
        uint64_t i = 0;
        file.write((char*)&i, sizeof(i));
        return name.size() + 1 + sizeof(i);
    }

    Tensor Identity::farward(const Tensor& X) { return X; }
    Tensor Identity::backward(const Tensor& X, const Tensor& d) { return d; }
    uint64_t Identity::save(std::ofstream &file) const { return save_act(file, "Identity"); }


    double ReLU::relu(double n)   { return n < 0. ? 0. : n; }      /* { return (n >= 0.) * n; } */
    double ReLU::relu_d(double n) { return n < 0. ? 0. : 1.; }     /* { return (double)(n >= 0.); } */

    Tensor ReLU::farward(const Tensor& X) { return forEach<relu>(X); }
    Tensor ReLU::backward(const Tensor& X, const Tensor& d) { return d * forEach<relu_d>(X); }
    uint64_t ReLU::save(std::ofstream &file) const { return save_act(file, "ReLU"); }


    double LeakyReLU::lrelu(double n)   { return n < 0. ? 0.01 * n : n; }   /* { return ((n < 0.) * .01 + (n >= 0.)) * n; } */
    double LeakyReLU::lrelu_d(double n) { return n < 0. ? 0.01 : 1.; }      /* { return (n < 0.) * .01  + (n >= 0.) * 1.; } */

    Tensor LeakyReLU::farward(const Tensor& X) { return forEach<lrelu>(X); }
    Tensor LeakyReLU::backward(const Tensor& X, const Tensor& d) { return d * forEach<lrelu_d>(X); }
    uint64_t LeakyReLU::save(std::ofstream &file) const { return save_act(file, "LeakyReLU"); }


    double PReLU::prelu(double n, double a)   { return n < 0. ? a * n : n; }    /* { return ((n < 0.) * a + (n >= 0.)) * n; } */
    double PReLU::prelu_d(double n, double a) { return n < 0. ? a : 1.; }       /* { return (n < 0.) * a  + (n >= 0.) * 1.; } */

    PReLU::PReLU(float64 a)
        : prelua([a](double n){return prelu(n, a);}),
          prelua_d([a](double n){return prelu_d(n, a);})
    {}

    PReLU::PReLU(std::ifstream &file, Optimizer* optimizer)
    {
        const std::string name = "Layer::Activation::PReLU";
        char rname[sizeof("Layer::Activation::PReLU")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in PReLU(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        float64 a = .25;
        file.read((char*)&a, sizeof(a));

        prelua   = [a](double n){return prelu(n, a);};
        prelua_d = [a](double n){return prelu_d(n, a);};
    }

    Tensor PReLU::farward(const Tensor& X) { return forEach(X, prelua); }
    Tensor PReLU::backward(const Tensor& X, const Tensor& d) { return d * forEach(X, prelua_d); }
    uint64_t PReLU::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Activation::PReLU";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(float64);
        file.write((char*)&size, sizeof(size));
        float64 a = prelua(-1.) * -1.;
        file.write((char*)&a, sizeof(a));
        return size + sizeof(name) + sizeof(size);
    }


    double Sigmoid::sigmoid(double n)   { return 1. / (1. + std::exp(-n)); }
    double Sigmoid::sigmoid_d(double n) { double sig_n = sigmoid(n); return sig_n * (1 - sig_n); }

    Tensor Sigmoid::farward(const Tensor& X) { return forEach<sigmoid>(X); }
    Tensor Sigmoid::backward(const Tensor& X, const Tensor& d) { return d * forEach<sigmoid_d>(X); }
    uint64_t Sigmoid::save(std::ofstream &file) const { return save_act(file, "Sigmoid"); }


    double TanH::tanh(double n)   { double exp_n = std::exp(n), exp_m_n = 1. / exp_n; return (exp_n - exp_m_n) / (exp_n + exp_m_n); }
    double TanH::tanh_d(double n) { return 1 - std::pow(tanh(n), 2); }

    Tensor TanH::farward(const Tensor& X) { return forEach<tanh>(X); }
    Tensor TanH::backward(const Tensor& X, const Tensor& d) { return d * forEach<tanh_d>(X); }
    uint64_t TanH::save(std::ofstream &file) const { return save_act(file, "TanH"); }


    double Softplus::softplus(double n)   { return std::log(1 + std::exp(n)); }
    double Softplus::softplus_d(double n) { return 1 / (1 + std::exp(-n)); }

    Tensor Softplus::farward(const Tensor& X) { return forEach<softplus>(X); }
    Tensor Softplus::backward(const Tensor& X, const Tensor& d) { return d * forEach<softplus_d>(X); }
    uint64_t Softplus::save(std::ofstream &file) const { return save_act(file, "Softplus"); }


    double SiLU::silu(double n)   { return n / (1 + std::exp(-n)); }
    double SiLU::silu_d(double n) { auto exp_n = std::exp(-n), one_exp_n = 1 + exp_n; return (one_exp_n + n * exp_n) / std::pow(one_exp_n, 2); }

    Tensor SiLU::farward(const Tensor& X) { return forEach<silu>(X); }
    Tensor SiLU::backward(const Tensor& X, const Tensor& d) { return d * forEach<silu_d>(X); }
    uint64_t SiLU::save(std::ofstream &file) const { return save_act(file, "SiLU"); }


    double Gaussian::gaussian(double n)   { return std::exp(-n*n); }
    double Gaussian::gaussian_d(double n) { return -2 * n * gaussian(n); }

    Tensor Gaussian::farward(const Tensor& X) { return forEach<gaussian>(X); }
    Tensor Gaussian::backward(const Tensor& X, const Tensor& d) { return d * forEach<gaussian_d>(X); }
    uint64_t Gaussian::save(std::ofstream &file) const { return save_act(file, "Gaussian"); }


    Tensor Softmax::farward(const Tensor& X)
    {
        Tensor max = X.max(0);
        Tensor expv = std::exp(X - max);
        Tensor sum = expv.sum(0);
        Tensor div = 1. / sum;
        expv *= div;
        return expv;
    }

    Tensor Softmax::backward(const Tensor& X, const Tensor& d)
    {
        Tensor grad({d.rowSize(), X.colSize()});
        Tensor dX({X.colSize()});
        Tensor g = farward(X);

        for (size_t r = 0, rs = X.rowSize(), cs = X.colSize(); r < rs; r++)
        {
            float64 g_i = g(r, (size_t)0);
            float64 d_r_i = d(r, (size_t)0);
            grad(r, (size_t)0) = d_r_i * g_i * (1. - g_i);
            for(size_t j = 1; j < cs; j++)
            {
                float64 dX = -g_i*g(r, j);
                grad(r, (size_t)0) += d(r, j) * dX;
                grad(r, j)  = d_r_i   * dX;
            }
            for (size_t i = 1; i < cs; i++)
            {
                g_i = g(r, i);
                d_r_i = d(r, i);
                grad(r, i) += d_r_i * g_i * (1. - g_i);
                for(size_t j = i + 1; j < cs; j++)
                {
                    float64 dX = -g_i*g(r, j);
                    grad(r, i) += d(r, j) * dX;
                    grad(r, j) += d_r_i   * dX;
                }
            }
        }

        return grad;
    }

    uint64_t Softmax::save(std::ofstream &file) const { return save_act(file, "Softmax"); }

}