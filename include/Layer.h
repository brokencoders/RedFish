#pragma once 

#include <vector>
#include <stdexcept>
#include <fstream>
#include "Tensor.h"
#include "Optimizer.h"

namespace RedFish {

    union Param
    {
        Param(Tuple2d n)  {_tuple2d = n;}
        Param(Tuple3d n)  {_tuple3d = n;}
        Param(size_t n)   {_size_t = n;}
        Param(uint32_t n) {_uint32 = n;}
        Param(uint16_t n) {_uint16 = n;}
        Param(uint8_t n)  {_uint8 = n;}
        Param(int64_t n)  {_integer64 = n;}
        Param(int32_t n)  {_integer32 = n;}
        Param(int16_t n)  {_integer16 = n;}
        Param(int8_t n)   {_integer8 = n;}
        Param(float n)    {_float = n;}
        Param(double n)   {_double = n;}
        Tuple2d  _tuple2d;
        Tuple3d  _tuple3d;
        size_t   _size_t;
        uint32_t _uint32;
        uint16_t _uint16;
        uint8_t  _uint8;
        int64_t  _integer64;
        int32_t  _integer32;
        int16_t  _integer16;
        int8_t   _integer8;
        float    _float;
        double   _double;
        float64  _float64;
    };

    enum LAYER : uint32_t {
        LINEAR,
        CONV1D,
        CONV2D,
        CONV3D,
        MAXPOOL1D,
        MAXPOOL2D,
        MAXPOOL3D,
        FLATTEN,
        DROPOUT,
        IDENTITY,
        RELU,
        LEAKY_RELU,
        PRELU,
        SIGMOID,
        TANH,
        SOFTPLUS,
        SILU,
        GAUSSIAN,
        SOFTMAX
    };

    class Layer {
    public:
        virtual ~Layer() {};
        Tensor operator()(const Tensor& X);
        virtual Tensor forward(const Tensor& X) = 0;
        virtual Tensor backward(const Tensor& X, const Tensor& d) = 0;
        virtual uint64_t save(std::ofstream& file) const = 0;
    
        struct Descriptor
        {
            LAYER type;
            std::vector<Param> param;
        };


    };

    Layer* make_layer(const Layer::Descriptor& dsc, Optimizer* opt);
    Layer* make_layer(std::ifstream& file, Optimizer* opt);

}