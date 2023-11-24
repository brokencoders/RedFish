#include "DropoutLayer.h"

namespace RedFish 
{
    DropoutLayer::DropoutLayer(float64 rate, std::vector<size_t> shape)
        :rate(rate), shape(shape) { }

    Tensor DropoutLayer::farward(const Tensor& X)
    {
        size_t input_size = 1;
        for (size_t i = 0; i < X.getShape().size(); i++)
            input_size *= X.getShape()[i];

        size_t skip_size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            skip_size *= shape[i];
        
        Tensor output = empty_like(X);

        std::mt19937 gen(Tensor::getRandomDevice()());
        // std::default_random_engine non funziona BRO
        std::bernoulli_distribution d(rate);

        float64 factor = 1 / (1 - rate);

        for (size_t i = 0; i < input_size; i+=skip_size)
        {
            if(d(gen) == true)
                for (size_t j = 0; j < skip_size; j++) output(i+j) = 0;
            else 
                for (size_t j = 0; j < skip_size; j++) output(i+j) = factor * X(i+j);
        }

        return output;
    }

    Tensor DropoutLayer::backward(const Tensor& X, const Tensor& d)
    {
        // Multiply by the rate 
        // if I pass 0 to the next layer the derivative i get is zero?
        return d;
    }
    
}