#pragma once 

#include "Algebra.h"

namespace RedFish {

    class Layer {
    public:
        virtual Algebra::Matrix farward(const Algebra::Matrix& X) = 0;
        virtual Algebra::Matrix backward(const Algebra::Matrix& X, const Algebra::Matrix& d, double learning_rate = 0.001) = 0;
    
        friend class Model;
    };
}