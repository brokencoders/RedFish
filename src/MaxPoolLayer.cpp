#include "MaxPoolLayer.h"

namespace RedFish {
    MaxPool1dLayer::MaxPool1dLayer(size_t kernel_size, size_t stride, size_t padding) 
        :kernel_size(kernel_size), stride(stride), padding(padding) {  }

    // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    Tensor MaxPool1dLayer::farward(const Tensor &X)
    {
        for (size_t i = 0; i + 3< X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in MaxPool1d farward");

        size_t N_size = X.getShape().end()[-3];
        size_t C_size = X.getShape().end()[-2];
        size_t L_size = X.getShape().end()[-1];

        size_t L_out_size = std::floor((L_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1);

        Tensor max_pool({N_size, C_size, L_out_size});

        for (size_t i = 0; i < N_size; i++)
            for (size_t j = 0; j < C_size; j++)
                for (size_t k = 0; k < L_out_size; k++)
                {
                    float64 max = 0;
                    for (size_t m = 0; m < kernel_size; m++)
                    {
                        if (max < X(i, j, stride * k + m))
                            max = X(i, j, stride * k + m);
                    }
                    max_pool(i, j, k) = max;
                }

        return max_pool;
    }

    Tensor MaxPool1dLayer::backward(const Tensor &X, const Tensor &d)
    {
        return Tensor();
    }

}