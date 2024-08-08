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

    Tensor Identity::forward(const Tensor& X) { if (training) this->X = X; return X; }
    Tensor Identity::backward(const Tensor& d) { return d; }
    uint64_t Identity::save(std::ofstream &file) const { return save_act(file, "Identity"); }


    double ReLU::relu(double n)   { return n < 0. ? 0. : n; }      /* { return (n >= 0.) * n; } */
    double ReLU::relu_d(double n) { return n < 0. ? 0. : 1.; }     /* { return (double)(n >= 0.); } */

    Tensor ReLU::forward(const Tensor& X) { if (training) this->X = X; return forEach<relu>(X); }
    Tensor ReLU::backward(const Tensor& d) { return d * forEach<relu_d>(X); }
    uint64_t ReLU::save(std::ofstream &file) const { return save_act(file, "ReLU"); }


    double LeakyReLU::lrelu(double n)   { return n < 0. ? 0.01 * n : n; }   /* { return ((n < 0.) * .01 + (n >= 0.)) * n; } */
    double LeakyReLU::lrelu_d(double n) { return n < 0. ? 0.01 : 1.; }      /* { return (n < 0.) * .01  + (n >= 0.) * 1.; } */

    Tensor LeakyReLU::forward(const Tensor& X) { if (training) this->X = X; return forEach<lrelu>(X); }
    Tensor LeakyReLU::backward(const Tensor& d) { return d * forEach<lrelu_d>(X); }
    uint64_t LeakyReLU::save(std::ofstream &file) const { return save_act(file, "LeakyReLU"); }


    double PReLU::prelu(double n, double a)   { return n < 0. ? a * n : n; }    /* { return ((n < 0.) * a + (n >= 0.)) * n; } */
    double PReLU::prelu_d(double n, double a) { return n < 0. ? a : 1.; }       /* { return (n < 0.) * a  + (n >= 0.) * 1.; } */

    PReLU::PReLU(float64 a)
        : prelua([a](double n){return prelu(n, a);}),
          prelua_d([a](double n){return prelu_d(n, a);})
    {}

    PReLU::PReLU(std::ifstream &file)
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

    Tensor PReLU::forward(const Tensor& X) { if (training) this->X = X; return forEach(X, prelua); }
    Tensor PReLU::backward(const Tensor& d) { return d * forEach(X, prelua_d); }
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

    Tensor Sigmoid::forward(const Tensor& X) { if (training) this->X = X; return forEach<sigmoid>(X); }
    Tensor Sigmoid::backward(const Tensor& d) { return d * forEach<sigmoid_d>(X); }
    uint64_t Sigmoid::save(std::ofstream &file) const { return save_act(file, "Sigmoid"); }


    double TanH::tanh(double n)   { return std::tanh(n); }
    double TanH::tanh_d(double n) { auto th = std::tanh(n); return 1 - th*th; }

    Tensor TanH::forward(const Tensor& X) { if (training) this->X = X; return forEach<tanh>(X); }
    Tensor TanH::backward(const Tensor& d) { return d * forEach<tanh_d>(X); }
    uint64_t TanH::save(std::ofstream &file) const { return save_act(file, "TanH"); }


    double Softplus::softplus(double n)   { return std::log(1 + std::exp(n)); }
    double Softplus::softplus_d(double n) { return 1 / (1 + std::exp(-n)); }

    Tensor Softplus::forward(const Tensor& X) { if (training) this->X = X; return forEach<softplus>(X); }
    Tensor Softplus::backward(const Tensor& d) { return d * forEach<softplus_d>(X); }
    uint64_t Softplus::save(std::ofstream &file) const { return save_act(file, "Softplus"); }


    double SiLU::silu(double n)   { return n / (1 + std::exp(-n)); }
    double SiLU::silu_d(double n) { auto exp_n = std::exp(-n), one_exp_n = 1 + exp_n; return (one_exp_n + n * exp_n) / std::pow(one_exp_n, 2); }

    Tensor SiLU::forward(const Tensor& X) { if (training) this->X = X; return forEach<silu>(X); }
    Tensor SiLU::backward(const Tensor& d) { return d * forEach<silu_d>(X); }
    uint64_t SiLU::save(std::ofstream &file) const { return save_act(file, "SiLU"); }


    double Gaussian::gaussian(double n)   { return std::exp(-n*n); }
    double Gaussian::gaussian_d(double n) { return -2 * n * gaussian(n); }

    Tensor Gaussian::forward(const Tensor& X) { if (training) this->X = X; return forEach<gaussian>(X); }
    Tensor Gaussian::backward(const Tensor& d) { return d * forEach<gaussian_d>(X); }
    uint64_t Gaussian::save(std::ofstream &file) const { return save_act(file, "Gaussian"); }


    Tensor Softmax::forward(const Tensor& X)
    {
        if (training) this->X = X;
        Tensor expv = std::exp(X - X.max(0));
        Tensor div = 1. / expv.sum(0);
        expv *= div;
        return expv;
    }

    Tensor Softmax::backward(const Tensor& d)
    {
        Tensor S = std::exp(X - X.max(0));
        Tensor div = 1. / S.sum(0);
        S *= div;
        return S*(d-(S*d).sum(0));
    }

    uint64_t Softmax::save(std::ofstream &file) const { return save_act(file, "Softmax"); }

}