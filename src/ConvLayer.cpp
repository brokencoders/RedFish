#include "ConvLayer.h"

namespace RedFish {

    Conv1dLayer::Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding, size_t dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size}), bias({out_channels, 1, 1}), stride(stride), padding(padding), dilation(dilation)
    {
        kernels.rand(-.5, .5);
        bias.rand(-.5, .5);
    }

    Tensor Conv1dLayer::farward(const Tensor& X)
    {
        size_t dim = X.getShape().end()[-3];
        for (size_t i = 0; i + 3 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in Conv1d farward");

        size_t conv_lenght = 0;
        if (2*padding + X.getShape().back() + stride >= (kernels.getShape().back()-1)*dilation + 1)
            conv_lenght = (2*padding + X.getShape().back() - (kernels.getShape().back()-1)*dilation - 1) / stride + 1;

        Tensor conv({dim, kernels.getShape()[0], conv_lenght});
        conv.zero();
        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < kernels.getShape()[0]; j++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                    conv.getRow({i, j}) += X.getRow({i, k}).crossCorrelation1d(kernels.getRow({j, k}));
                
        conv += bias;
        return conv;
    }

    Tensor Conv1dLayer::backward(const Tensor& X, const Tensor& d) 
    {
        //auto grad = X.crossCorrelation2d(d(:,1,:,:));
        
        return Tensor();
    }

}


/* 



k1 k2 k3

H = k1*R + k2*G + k3*B

X: (N,Cin,h,w),     (N,2,2,3)   (  N,2,2,3)
k: (Cout,Cin,h,w),  (1,2,2,2)   (N,1,1,1,2)
o/d: (N,Cout,h,w),  (N,2,1,2)      N,1,2,2
1°
| x y h |   | a b | = | ax+by+cz+dw  ay+bh+cw+dg |
| z w g | * | c d |                1                            1
2°
| o p j |   | e f | = | eo+fp+rs+qu  ep+fj+ru+qt |
| s u t | * | r q | 

df/da = | x, y | = x + y  df/db = | y, h | = y + h
df/dc = | z, w | = z + w  df/dd = | w, g | = w + g 

df/de = | o, p | = o + p  df/df = | p, j | = p + j
df/dr = | s, u | = s + u  df/dq = | u, t | = u + t 

L(m,n)

dL/dm
dL/dn

h(t) = (X*k)(t) = /X(t')k(t-t')dt'
(X*dk/dt)(t) = (dX/dt*k)(t)
dL/dh*dh/dk


 */
