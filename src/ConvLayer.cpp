#include "ConvLayer.h"

namespace RedFish {

    Conv1dLayer::Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, Optimizer* optimizer, size_t stride, size_t padding, size_t dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size}), bias({out_channels, 1}), stride(stride), padding(padding), dilation(dilation), pm(pm), optimizer(optimizer)
    {
        kernels.randUniform(-.5, .5);
        bias.randUniform(-.5, .5);
        k_id = optimizer->allocateParameter(kernels);
        b_id = optimizer->allocateParameter(bias);
    }

    Conv1dLayer::Conv1dLayer(std::ifstream &file, Optimizer* optimizer)
        : optimizer(optimizer)
    {
        const std::string name = "Layer::Conv1d";
        char rname[sizeof("Layer::Conv1d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Conv1dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&k_id, sizeof(k_id));
        file.read((char*)&b_id, sizeof(b_id));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    Tensor Conv1dLayer::farward(const Tensor& X)
    {
        if (X.getShape().size() < 3)
            throw std::length_error("Invalid size of X in Conv1d farward");

        size_t dim = X.getShape().end()[-3];
        for (size_t i = 0; i + 3 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in Conv1d farward");

        size_t conv_length = 0;
        if (2*padding + X.getShape().back() + stride >= (kernels.getShape().back()-1)*dilation + 1)
            conv_length = (2*padding + X.getShape().back() - (kernels.getShape().back()-1)*dilation - 1) / stride + 1;

        Tensor conv({dim, kernels.getShape()[0], conv_length});
        conv.zero();
        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < kernels.getShape()[0]; j++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                    conv.getRow({i, j}) += X.getRow({i, k}).crossCorrelation1d(kernels.getRow({j, k}), padding, stride, dilation, pm);
                
        conv += bias;
        return conv;
    }

    Tensor Conv1dLayer::backward(const Tensor& X, const Tensor& d) 
    {
        if (X.getShape().size() < 3)
            throw std::length_error("Invalid size of X in Conv1d backward");
        if (d.getShape().size() < 3)
            throw std::length_error("Invalid size of d in Conv1d backward");

        size_t dim = X.getShape().end()[-3];
        for (size_t i = 0; i + 3 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in Conv1d farward");

        size_t conv_length = 0;
        if (2*padding + X.getShape().back() + stride >= (kernels.getShape().back()-1)*dilation + 1)
            conv_length = (2*padding + X.getShape().back() - (kernels.getShape().back()-1)*dilation - 1) / stride + 1;
            
        if (d.getShape().end()[-1] != conv_length)
            throw std::length_error("Invalid size of d in Conv1d backward");

        Tensor kgrad = zeros_like(kernels);

        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < kernels.getShape()[0]; j++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                    kgrad.getRow({j, k}) += X.getRow({i, k}).crossCorrelation1d(d.getRow({i, j}), padding, dilation, stride, pm);
        
        Tensor bgrad = d.sum(0).sum(2);

        Tensor xgrad = zeros_like(X);
        Tensor dilated({d.getShape().end()[-3], d.getShape().end()[-2], (d.getShape().back()-1)*stride + 1 + 2*kernels.getShape().back() - 2 - 2*padding});
        int64_t newpad = (int64_t)kernels.getShape().back() - 1 - padding;

        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < kernels.getShape()[0]; j++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                {

                    auto drow = d.getRow({i, j});
                    for (size_t c = 0; c < d.getShape().back(); c++)
                    {
                        if (newpad + (int64_t)c*stride >= 0)
                            dilated(newpad + c*stride) = drow(c);
                    }
                    

                    Tensor pregrad = dilated.convolution1d(kernels.getRow({j, k}));

                    auto xgradrow = xgrad.getRow({i, k});
                    for (size_t c = 0; c < xgradrow.getShape().back(); c++)
                    {
                        xgradrow(c) += pregrad(c);
                    }
                    
                }

        optimizer->updateParameter(k_id, kernels, kgrad);
        optimizer->updateParameter(b_id, bias, bgrad);

        return xgrad;
    }

    uint64_t Conv1dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv1d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(stride) + sizeof(padding)
                      + sizeof(dilation) + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
        file.write((char*)&pm, sizeof(pm));
 
        size += kernels.save(file);
        size += bias.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    const Tensor &Conv1dLayer::getKernels() const
    {
        return kernels;
    }

    const Tensor &Conv1dLayer::getBiases() const
    {
        return bias;
    }

    Conv2dLayer::Conv2dLayer(size_t in_channels, size_t out_channels, Tuple2d kernel_size, Optimizer* optimizer, Tuple2d stride, Tuple2d padding, Tuple2d dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size.h, kernel_size.w}), bias({out_channels, 1, 1}), stride(stride), padding(padding), dilation(dilation), pm(pm), optimizer(optimizer)
    {
        kernels.randUniform(-.5, .5);
        bias.randUniform(-.5, .5);
        k_id = optimizer->allocateParameter(kernels);
        b_id = optimizer->allocateParameter(bias);
    }

    Conv2dLayer::Conv2dLayer(std::ifstream &file, Optimizer* optimizer)
        : optimizer(optimizer)
    {
        const std::string name = "Layer::Conv2d";
        char rname[sizeof("Layer::Conv2d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Conv2dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&k_id, sizeof(k_id));
        file.read((char*)&b_id, sizeof(b_id));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    Tensor Conv2dLayer::farward(const Tensor& X)
    {
        if (X.getShape().size() < 4)
            throw std::length_error("Invalid size of X in Conv2d farward");

        size_t dim = X.getShape().end()[-4];
        for (size_t i = 0; i + 4 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in Conv2d farward");

        Tuple2d conv_length = 0;
        if (2*padding.x + X.getShape().back() + stride.x >= (kernels.getShape().back()-1)*dilation.x + 1) conv_length.x = (2*padding.x + X.getShape().back() - (kernels.getShape().back()-1)*dilation.x + stride.x - 1) / stride.x;
        else conv_length.x = 0;

        if (2*padding.y + X.getShape().end()[-2] + stride.y >= (kernels.getShape().end()[-2]-1)*dilation.y + 1) conv_length.y = (2*padding.y + X.getShape().end()[-2] - (kernels.getShape().end()[-2]-1)*dilation.y + stride.y - 1) / stride.y;
        else conv_length.y = 0;


        Tensor conv({dim, kernels.getShape()[0], conv_length.y, conv_length.x});
        conv.zero();
        #pragma omp parallel for
        for (size_t k = 0; k < kernels.getShape()[1]; k++)
            for (size_t i = 0; i < dim; i++)
                for (size_t j = 0; j < kernels.getShape()[0]; j++)
                    conv.getMatrix({i, j}) += X.getMatrix({i, k}).crossCorrelation2d(kernels.getMatrix({j, k}), padding, stride, dilation, pm);
                
        conv += bias;
        return conv;
    }

    Tensor Conv2dLayer::backward(const Tensor& X, const Tensor& d) 
    {
        const float64 lambda = .001;
        if (X.getShape().size() < 4)
            throw std::length_error("Invalid size of X in Conv2d backward");
        if (d.getShape().size() < 4)
            throw std::length_error("Invalid size of d in Conv2d backward");

        size_t dim = X.getShape().end()[-4];
        for (size_t i = 0; i + 4 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in Conv2d farward");

        Tuple2d conv_length = 0;
        if (2*padding.x + X.getShape().back() + stride.x >= (kernels.getShape().back()-1)*dilation.x + 1) conv_length.x = (2*padding.x + X.getShape().back() - (kernels.getShape().back()-1)*dilation.x + stride.x - 1) / stride.x;
        else conv_length.x = 0;

        if (2*padding.y + X.getShape().end()[-2] + stride.y >= (kernels.getShape().end()[-2]-1)*dilation.y + 1) conv_length.y = (2*padding.y + X.getShape().end()[-2] - (kernels.getShape().end()[-2]-1)*dilation.y + stride.y - 1) / stride.y;
        else conv_length.y = 0;

            
        if (d.getShape().end()[-1] != conv_length.x)
            throw std::length_error("Invalid size of d in Conv2d backward");

        if (d.getShape().end()[-2] != conv_length.y)
            throw std::length_error("Invalid size of d in Conv2d backward");

        Tensor kgrad = zeros_like(kernels);

        #pragma omp parallel for
        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < kernels.getShape()[0]; j++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                    kgrad.getMatrix({j, k}) += X.getMatrix({i, k}).crossCorrelation2d(d.getMatrix({i, j}), padding, dilation, stride, pm);
        
        /* kgrad += kernels * lambda; */

        Tensor bgrad = d.sum(0).sum(1).sum(3) /* + bias * lambda */;

        Tensor xgrad = zeros_like(X);
        Tensor dilated({d.getShape().end()[-4], d.getShape().end()[-3], (d.getShape().end()[-2]-1)*stride.y + 1 + 2*kernels.getShape().end()[-2] - 2 - 2*padding.y, (d.getShape().back()-1)*stride.x + 1 + 2*kernels.getShape().back() - 2 - 2*padding.x});
        int newpady = (int64_t)kernels.getShape().end()[-2] - 1 - padding.y;
        int newpadx = (int64_t)kernels.getShape().back() - 1 - padding.x;

        #pragma omp parallel for
        for (size_t j = 0; j < kernels.getShape()[0]; j++)
            for (size_t i = 0; i < dim; i++)
                for (size_t k = 0; k < kernels.getShape()[1]; k++)
                {

                    auto drow = d.getMatrix({i, j});
                    for (size_t r = 0; r < d.getShape().end()[-2]; r++)
                        if (newpady + (int64_t)r*stride.y >= 0)
                        for (size_t c = 0; c < d.getShape().back(); c++)
                        {   
                            if (newpadx + (int64_t)c*stride.x >= 0)
                                dilated(newpady + r*stride.y, newpadx + c*stride.x) = drow(r, c);
                        }
                    

                    Tensor pregrad = dilated.convolution2d(kernels.getMatrix({j, k}));

                    auto xgradrow = xgrad.getMatrix({i, k});
                    for (size_t r = 0; r < xgradrow.getShape().end()[-2]; r++)
                    for (size_t c = 0; c < xgradrow.getShape().back(); c++)
                    {
                        xgradrow(r,c) = pregrad(r,c);
                    }
                    
                }

        optimizer->updateParameter(k_id, kernels, kgrad);
        optimizer->updateParameter(b_id, bias, bgrad);

        //std::cout << d;

        return xgrad;
    }

    uint64_t Conv2dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv2d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(stride) + sizeof(padding)
                      + sizeof(dilation) + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
        file.write((char*)&pm, sizeof(pm));
 
        size += kernels.save(file);
        size += bias.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    const Tensor &Conv2dLayer::getKernels() const
    {
        return kernels;
    }

    const Tensor &Conv2dLayer::getBiases() const
    {
        return bias;
    }

    Conv3dLayer::Conv3dLayer(size_t in_channels, size_t out_channels, Tuple3d kernel_size, Optimizer *optimizer, Tuple3d stride, Tuple3d padding, Tuple3d dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size.d,kernel_size.h, kernel_size.w}), bias({out_channels, 1, 1, 1}), stride(stride), padding(padding), dilation(dilation), pm(pm), optimizer(optimizer)
    {
    }

    Conv3dLayer::Conv3dLayer(std::ifstream &file, Optimizer* optimizer)
        : optimizer(optimizer)
    {
        const std::string name = "Layer::Conv3d";
        char rname[sizeof("Layer::Conv3d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Conv3dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&k_id, sizeof(k_id));
        file.read((char*)&b_id, sizeof(b_id));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    Tensor Conv3dLayer::farward(const Tensor &X)
    {
        return Tensor();
    }
    
    Tensor Conv3dLayer::backward(const Tensor &X, const Tensor &d)
    {
        return Tensor();
    }

    uint64_t Conv3dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv3d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(stride) + sizeof(padding)
                      + sizeof(dilation) + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
        file.write((char*)&pm, sizeof(pm));
 
        size += kernels.save(file);
        size += bias.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    const Tensor &Conv3dLayer::getKernels() const
    {
        return kernels;
    }

    const Tensor &Conv3dLayer::getBiases() const
    {
        return bias;
    }
}

/* 

------------ xn ---- 
k1 k2 k3 ... kn
--- xnkn --- xnk3 xnk2 xnk1

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
