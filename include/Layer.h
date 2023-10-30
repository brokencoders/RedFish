#pragma once 

#include "Tensor.h"

namespace RedFish {

    class Layer {
    public:
        virtual Tensor farward(const Tensor& X) = 0;
        virtual Tensor backward(const Tensor& X, const Tensor& d) = 0;
    
        friend class Model;
    };
}