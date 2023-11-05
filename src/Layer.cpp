#include "Layer.h"
#include "LinearLayer.h"
#include "ActivationLayer.h"

namespace RedFish {

    Layer* make_layer(const Layer::Descriptor& dsc, Optimizer* opt)
    {
        switch (dsc.type)
        {
        case Layer::Descriptor::LINEAR:
            if (dsc.param.size() < 2) throw std::length_error("Not enouth parameters for LinearLayer construction");
            return new LinearLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, opt);
        
        case Layer::Descriptor::IDENTITY:   return new Activation::Identity();
        case Layer::Descriptor::RELU:       return new Activation::ReLU();
        case Layer::Descriptor::LEAKY_RELU: return new Activation::LeakyReLU();
        case Layer::Descriptor::PRELU:
            if (dsc.param.size() < 1) throw std::length_error("Not enouth parameters for PReLU construction");
            return new Activation::PReLU(dsc.param[0]._double);
        case Layer::Descriptor::SIGMOID:    return new Activation::Sigmoid();
        case Layer::Descriptor::TANH:       return new Activation::TanH();
        case Layer::Descriptor::SOFTPLUS:   return new Activation::Softplus();
        case Layer::Descriptor::SILU:       return new Activation::SiLU();
        case Layer::Descriptor::GAUSSIAN:   return new Activation::Gaussian();
        case Layer::Descriptor::SOFTMAX:    return new Activation::Softmax();

        default: return nullptr;
        }
    }

}