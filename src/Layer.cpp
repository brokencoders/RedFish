#include "Layer.h"
#include "LinearLayer.h"
#include "ActivationLayer.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"
#include "FlattenLayer.h"

namespace RedFish {

    Layer* make_layer(const Layer::Descriptor& dsc, Optimizer* opt)
    {
        switch (dsc.type)
        {
        case Layer::Descriptor::LINEAR:
            if (dsc.param.size() < 2) throw std::length_error("Not enouth parameters for LinearLayer construction");
            return new LinearLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, opt);

        case Layer::Descriptor::CONV1D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv1dLayer construction");
            return new Conv1dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._size_t, dsc.param[4]._size_t, dsc.param[5]._size_t, (PaddingMode)dsc.param[6]._integer8);
        case Layer::Descriptor::CONV2D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv2dLayer construction");
            return new Conv2dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._tuple2d, dsc.param[4]._tuple2d, dsc.param[5]._tuple2d, (PaddingMode)dsc.param[6]._integer8);
        case Layer::Descriptor::CONV3D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv3dLayer construction");
            return new Conv3dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._tuple3d, dsc.param[4]._tuple3d, dsc.param[5]._tuple3d, (PaddingMode)dsc.param[6]._integer8);

        case Layer::Descriptor::MAXPOOL1D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool1dLayer construction");
            return new MaxPool1dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, dsc.param[3]._size_t);
        case Layer::Descriptor::MAXPOOL2D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool2dLayer construction");
            return new MaxPool2dLayer(dsc.param[0]._tuple2d, dsc.param[1]._tuple2d, dsc.param[2]._tuple2d, dsc.param[3]._tuple2d);
        case Layer::Descriptor::MAXPOOL3D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool3dLayer construction");
            return new MaxPool3dLayer(dsc.param[0]._tuple3d, dsc.param[1]._tuple3d, dsc.param[2]._tuple3d, dsc.param[3]._tuple3d);

        case Layer::Descriptor::FLATTEN:
            if (dsc.param.size() < 2) throw std::length_error("Not enouth parameters for FlattenLayer construction");
            return new FlattenLayer(dsc.param[0]._size_t, dsc.param[1]._size_t);
            
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