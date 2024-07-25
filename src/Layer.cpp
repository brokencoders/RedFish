#include "Layer.h"
#include "LinearLayer.h"
#include "ActivationLayer.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"
#include "FlattenLayer.h"
#include "DropoutLayer.h"

namespace RedFish {
    
    Tensor Layer::operator()(const Tensor &X)
    {
        return this->forward(X);
    }

    Layer* make_layer(const Layer::Descriptor& dsc, Optimizer* opt)
    {
        switch (dsc.type)
        {
        case LAYER::LINEAR:
            if (dsc.param.size() < 2) throw std::length_error("Not enouth parameters for LinearLayer construction");
            return new LinearLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, opt);

        case LAYER::CONV1D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv1dLayer construction");
            return new Conv1dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._size_t, dsc.param[4]._size_t, dsc.param[5]._size_t, (PaddingMode)dsc.param[6]._integer8);
        case LAYER::CONV2D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv2dLayer construction");
            return new Conv2dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._tuple2d, dsc.param[4]._tuple2d, dsc.param[5]._tuple2d, (PaddingMode)dsc.param[6]._integer8);
        case LAYER::CONV3D:
            if (dsc.param.size() < 6) throw std::length_error("Not enouth parameters for Conv3dLayer construction");
            return new Conv3dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, opt,
                                   dsc.param[3]._tuple3d, dsc.param[4]._tuple3d, dsc.param[5]._tuple3d, (PaddingMode)dsc.param[6]._integer8);

        case LAYER::MAXPOOL1D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool1dLayer construction");
            return new MaxPool1dLayer(dsc.param[0]._size_t, dsc.param[1]._size_t, dsc.param[2]._size_t, dsc.param[3]._size_t);
        case LAYER::MAXPOOL2D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool2dLayer construction");
            return new MaxPool2dLayer(dsc.param[0]._tuple2d, dsc.param[1]._tuple2d, dsc.param[2]._tuple2d, dsc.param[3]._tuple2d);
        case LAYER::MAXPOOL3D:
            if (dsc.param.size() < 4) throw std::length_error("Not enouth parameters for MaxPool3dLayer construction");
            return new MaxPool3dLayer(dsc.param[0]._tuple3d, dsc.param[1]._tuple3d, dsc.param[2]._tuple3d, dsc.param[3]._tuple3d);

        case LAYER::FLATTEN:
            if (dsc.param.size() < 2) throw std::length_error("Not enouth parameters for FlattenLayer construction");
            return new FlattenLayer(dsc.param[0]._size_t, dsc.param[1]._size_t);
        
        case LAYER::DROPOUT:
            if (dsc.param.size() < 1) throw std::length_error("Not enouth parameters for DropoutLayer construction");
            return new DropoutLayer(dsc.param[0]._float64);
            
        case LAYER::IDENTITY:   return new Activation::Identity();
        case LAYER::RELU:       return new Activation::ReLU();
        case LAYER::LEAKY_RELU: return new Activation::LeakyReLU();
        case LAYER::PRELU:
            if (dsc.param.size() < 1) throw std::length_error("Not enouth parameters for PReLU construction");
            return new Activation::PReLU(dsc.param[0]._float64);
        case LAYER::SIGMOID:    return new Activation::Sigmoid();
        case LAYER::TANH:       return new Activation::TanH();
        case LAYER::SOFTPLUS:   return new Activation::Softplus();
        case LAYER::SILU:       return new Activation::SiLU();
        case LAYER::GAUSSIAN:   return new Activation::Gaussian();
        case LAYER::SOFTMAX:    return new Activation::Softmax();

        default: return nullptr;
        }
    }

    Layer *make_layer(std::ifstream &file, Optimizer *opt)
    {
        auto spos = file.tellg();
        char name[128] = "", c = 1;
        for (size_t i = 0; i < sizeof(name) && c != '\0'; name[i] = c, i++)
            file.get(c);

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        if (size > 0) file.seekg(spos);
        
        const std::string layer = "Layer::";
        if (std::string(name).find_first_of(layer) != 0)
            throw std::runtime_error("Invalid file content in make_layer(std::ifstram&)");

        const std::string layer_name = name + sizeof("Layer::") - 1;

        if (layer_name == "Activation::Identity")  return new Activation::Identity();
        if (layer_name == "Activation::ReLU")      return new Activation::ReLU();
        if (layer_name == "Activation::LeakyReLU") return new Activation::LeakyReLU();
        if (layer_name == "Activation::PReLU")     return new Activation::PReLU(file, opt);
        if (layer_name == "Activation::Sigmoid")   return new Activation::Sigmoid();
        if (layer_name == "Activation::TanH")      return new Activation::TanH();
        if (layer_name == "Activation::Softplus")  return new Activation::Softplus();
        if (layer_name == "Activation::SiLU")      return new Activation::SiLU();
        if (layer_name == "Activation::Gaussian")  return new Activation::Gaussian();
        if (layer_name == "Activation::Softmax")   return new Activation::Softmax();
        if (layer_name == "Conv1d")                return new Conv1dLayer(file, opt);
        if (layer_name == "Conv2d")                return new Conv2dLayer(file, opt);
        if (layer_name == "Conv3d")                return new Conv3dLayer(file, opt);
        if (layer_name == "Dropout")               return new DropoutLayer(file);
        if (layer_name == "Flatten")               return new FlattenLayer(file);
        if (layer_name == "Linear")                return new LinearLayer(file, opt);
        if (layer_name == "MaxPool1d")             return new MaxPool1dLayer(file);
        if (layer_name == "MaxPool2d")             return new MaxPool2dLayer(file);
        if (layer_name == "MaxPool3d")             return new MaxPool3dLayer(file);

        return nullptr;
    }
}