#include "DropoutLayer.h"

namespace RedFish 
{
    DropoutLayer::DropoutLayer(float64 rate)
        :rate(rate) { }

    Tensor DropoutLayer::farward(const Tensor& X)
    {
        if(output.getShape().size() == 0)
        {
            output = empty_like(X);

            batch_size = X.getShape()[0];
            
            skip_size = 1;
            for (size_t i = 1; i < X.getShape().size(); i++)
                skip_size *= X.getShape()[i];

            factor = 1 / (1 - rate);
        }
        std::mt19937 gen(Tensor::getRandomDevice()());
        std::bernoulli_distribution d(rate);

        for (size_t i = 0; i < skip_size; i++)
        {
            float64 t = 1;
            if(d(gen) == true)
                t = 0;
            
            for (size_t j = 0; j < batch_size; j++) 
                output(i + j * skip_size) = t * factor * X(i + j * skip_size);
        }

        return output;
    }

    Tensor DropoutLayer::backward(const Tensor& X, const Tensor& d)
    {
        Tensor grad = empty_like(X);
        
        for(size_t i = 0; i < batch_size * skip_size; i++)
        {
            if (output(i) == 0) grad(i) = 0;
            else grad(i) = d(i) * factor;
        }

        return grad;
    }
    
}