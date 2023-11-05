#pragma once 

#include <vector>
#include <stdexcept>
#include "Tensor.h"
#include "Optimizer.h"

namespace RedFish {

    union Param
    {
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
    };

    class Layer {
    public:
        virtual Tensor farward(const Tensor& X) = 0;
        virtual Tensor backward(const Tensor& X, const Tensor& d) = 0;
    
        struct Descriptor
        {
            enum : uint32_t {
                LINEAR,
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
            } type;
                
            std::vector<Param> param;
        };


    };

    Layer* make_layer(const Layer::Descriptor& dsc, Optimizer* opt);

}