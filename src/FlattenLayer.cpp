#include "FlattenLayer.h"

namespace RedFish {

    FlattenLayer::FlattenLayer(size_t start_dim, size_t end_dim)
        :start_dim(start_dim), end_dim(end_dim) { }

    Tensor FlattenLayer::farward(const Tensor& X)
    {
        Tensor flatten(X);
        std::vector<size_t> new_dim;
        
        size_t dim = 1;
        for (size_t i = 0; i < X.getShape().size(); i++)
        {
            if (i >= start_dim && i <= end_dim )
            {
                dim *= X.getShape()[i];
                if(i == end_dim || i == X.getShape().size() - 1)
                    new_dim.push_back(dim);
            }
            else 
                new_dim.push_back(X.getShape()[i]);
        }
        

        flatten.reshape(new_dim);
        return flatten;
    }

    Tensor FlattenLayer::backward(const Tensor& X, const Tensor& d)
    {
        Tensor grad(d);
        grad.reshape(X.getShape());
        return grad;
    }
}