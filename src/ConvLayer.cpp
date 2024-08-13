#include "ConvLayer.h"

namespace RedFish {

    /* 1D */

    Conv1dLayer::Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, int64_t padding, size_t dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size}), bias({out_channels, 1}), in_ch(in_channels), out_ch(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pm(pm)
    {
        float64 stdv = 1. / std::sqrt(in_channels*kernel_size);
        kernels.randUniform(-stdv, stdv);
        bias.zero();
    }

    Conv1dLayer::Conv1dLayer(std::ifstream &file)
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
        file.read((char*)&in_ch, sizeof(in_ch));
        file.read((char*)&out_ch, sizeof(out_ch));
        file.read((char*)&kernel_size, sizeof(kernel_size));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    void Conv1dLayer::useOptimizer(Optimizer &optimizer)
    {
        if (this->optimizer)
        {
            this->optimizer->deleteParameters(k_id);
            this->optimizer->deleteParameters(b_id);
        }
        k_id = optimizer.allocateParameters(kernels);
        b_id = optimizer.allocateParameters(bias);
        this->optimizer = &optimizer;
    }

    Tensor Conv1dLayer::forward(const Tensor& X)
    {
        if (training) this->X = X;
        Tensor result = X.asShapeOneInsert(2).correlation1d(kernels, padding, stride, dilation, ZERO, 1, true);
        result += bias;

        return result;
    }

    Tensor Conv1dLayer::backward(const Tensor& d) 
    {
        Tensor d_unstrided;
        if (stride > 1)
        {
            auto L = X.colSize();
            auto ushape = d.getShape();
            if (ushape.size()) ushape.back() = L - dilation*(kernel_size - 1) + 2*padding;
            d_unstrided = d.dilated(ushape);
        }

        auto dd = stride == 1 ? d.asShapeOneInsert(1) : d_unstrided.asShapeOneInsert(1);
        auto XX = X.asShapeOneInsert(2);

        Tensor grad_X = dd.convolution1d(kernels, dilation*(kernel_size - 1) - padding, 1, dilation, ZERO, 2, true);
        Tensor grad_k = XX.correlation1d(dd, padding, 1, 1, ZERO, 3, true);
        Tensor grad_b = d.sum(0).sum(2, true);

        if (dilation > 1) grad_k.shrink(kernels.getShape(), true);

        optimizer->grad(k_id) += grad_k;
        optimizer->grad(b_id) += grad_b;
        
        return grad_X;
    }

    uint64_t Conv1dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv1d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(in_ch) + sizeof(out_ch) 
                      + sizeof(kernel_size)
                      + sizeof(stride) + sizeof(padding)
                      + sizeof(dilation) + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&in_ch, sizeof(in_ch));
        file.write((char*)&out_ch, sizeof(out_ch));
        file.write((char*)&kernel_size, sizeof(kernel_size));
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

    /* 2D */

    Conv2dLayer::Conv2dLayer(size_t in_channels, size_t out_channels, TupleNd<2> kernel_size, TupleNd<2> stride, TupleNd<2, int64_t> padding, TupleNd<2> dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size.h, kernel_size.w}), bias({out_channels, 1, 1}), in_ch(in_channels), out_ch(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pm(pm)
    {
        float64 stdv = 1. / std::sqrt(in_channels*kernel_size.x*kernel_size.y);
        kernels.randUniform(-stdv, stdv);
        bias.zero();
    }

    Conv2dLayer::Conv2dLayer(std::ifstream &file)
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
        file.read((char*)&in_ch, sizeof(in_ch));
        file.read((char*)&out_ch, sizeof(out_ch));
        file.read((char*)&kernel_size.x, sizeof(kernel_size.x));
        file.read((char*)&kernel_size.y, sizeof(kernel_size.y));
        file.read((char*)&stride.x, sizeof(stride.x));
        file.read((char*)&stride.y, sizeof(stride.y));
        file.read((char*)&padding.x, sizeof(padding.x));
        file.read((char*)&padding.y, sizeof(padding.y));
        file.read((char*)&dilation.x, sizeof(dilation.x));
        file.read((char*)&dilation.y, sizeof(dilation.y));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    void Conv2dLayer::useOptimizer(Optimizer &optimizer)
    {
        if (this->optimizer)
        {
            this->optimizer->deleteParameters(k_id);
            this->optimizer->deleteParameters(b_id);
        }
        k_id = optimizer.allocateParameters(kernels);
        b_id = optimizer.allocateParameters(bias);
        this->optimizer = &optimizer;
    }

    Tensor Conv2dLayer::forward(const Tensor& X)
    {
        if (training) this->X = X;
        Tensor result = X.asShapeOneInsert(3).correlation2d(kernels, padding, stride, dilation, ZERO, 2, true);
        result += bias;

        return result;
    }

    Tensor Conv2dLayer::backward(const Tensor& d) 
    {
        Tensor d_unstrided;
        if (stride.x > 1 || stride.y > 1)
        {
            TupleNd<2> L(X.rowSize(), X.colSize());
            TupleNd<2> Lo = L-dilation*(kernel_size - 1) + padding*2;
            auto ushape = d.getShape();
            if (ushape.size() > 0) ushape.end()[-1] = Lo.x;
            if (ushape.size() > 1) ushape.end()[-2] = Lo.y;
            d_unstrided = d.dilated(ushape);
        }

        auto dd = stride.x == 1 && stride.y == 1 ? d.asShapeOneInsert(2) : d_unstrided.asShapeOneInsert(2);
        auto XX = X.asShapeOneInsert(3);

        Tensor grad_X = dd.convolution2d(kernels, dilation*(kernel_size - 1) - padding, 1, dilation, ZERO, 3, true);
        Tensor grad_k = XX.correlation2d(dd, padding, 1, 1, ZERO, 4, true);
        Tensor grad_b = d.sum(0).sum(1).sum(3, true);

        if (dilation.x > 1 || dilation.y > 1) grad_k.shrink(kernels.getShape(), true);

        optimizer->grad(k_id) += grad_k;
        optimizer->grad(b_id) += grad_b;
        
        return grad_X;
    }

    uint64_t Conv2dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv2d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(in_ch) + sizeof(out_ch) 
                      + sizeof(kernel_size.x) + sizeof(kernel_size.y)
                      + sizeof(stride.x)      + sizeof(stride.y)
                      + sizeof(padding.x)     + sizeof(padding.y)
                      + sizeof(dilation.x)    + sizeof(dilation.y)
                      + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&in_ch, sizeof(in_ch));
        file.write((char*)&out_ch, sizeof(out_ch));
        file.write((char*)&kernel_size.x, sizeof(kernel_size.x));
        file.write((char*)&kernel_size.y, sizeof(kernel_size.y));
        file.write((char*)&stride.x, sizeof(stride.x));
        file.write((char*)&stride.y, sizeof(stride.y));
        file.write((char*)&padding.x, sizeof(padding.x));
        file.write((char*)&padding.y, sizeof(padding.y));
        file.write((char*)&dilation.x, sizeof(dilation.x));
        file.write((char*)&dilation.y, sizeof(dilation.y));
        file.write((char*)&pm, sizeof(pm));
 
        size += kernels.save(file);
        size += bias.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    /* 3D */

    Conv3dLayer::Conv3dLayer(size_t in_channels, size_t out_channels, TupleNd<3> kernel_size, TupleNd<3> stride, TupleNd<3, int64_t> padding, TupleNd<3> dilation, PaddingMode pm)
        : kernels({out_channels,in_channels,kernel_size.d,kernel_size.h, kernel_size.w}), bias({out_channels, 1, 1, 1}), in_ch(in_channels), out_ch(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pm(pm)
    {
        float64 stdv = 1. / std::sqrt(in_channels*kernel_size.x*kernel_size.y*kernel_size.z);
        kernels.randUniform(-stdv, stdv);
        bias.zero();
    }

    Conv3dLayer::Conv3dLayer(std::ifstream &file)
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
        file.read((char*)&in_ch, sizeof(in_ch));
        file.read((char*)&out_ch, sizeof(out_ch));
        file.read((char*)&kernel_size.x, sizeof(kernel_size.x));
        file.read((char*)&kernel_size.y, sizeof(kernel_size.y));
        file.read((char*)&kernel_size.z, sizeof(kernel_size.z));
        file.read((char*)&stride.x, sizeof(stride.x));
        file.read((char*)&stride.y, sizeof(stride.y));
        file.read((char*)&stride.z, sizeof(stride.z));
        file.read((char*)&padding.x, sizeof(padding.x));
        file.read((char*)&padding.y, sizeof(padding.y));
        file.read((char*)&padding.z, sizeof(padding.z));
        file.read((char*)&dilation.x, sizeof(dilation.x));
        file.read((char*)&dilation.y, sizeof(dilation.y));
        file.read((char*)&dilation.z, sizeof(dilation.z));
        file.read((char*)&pm, sizeof(pm));
 
        kernels = Tensor(file);
        bias    = Tensor(file);
    }

    void Conv3dLayer::useOptimizer(Optimizer &optimizer)
    {
        if (this->optimizer)
        {
            this->optimizer->deleteParameters(k_id);
            this->optimizer->deleteParameters(b_id);
        }
        k_id = optimizer.allocateParameters(kernels);
        b_id = optimizer.allocateParameters(bias);
        this->optimizer = &optimizer;
    }

    Tensor Conv3dLayer::forward(const Tensor &X)
    {
        if (training) this->X = X;
        Tensor result = X.asShapeOneInsert(4).correlation3d(kernels, padding, stride, dilation, ZERO, 3, true);
        result += bias;

        return result;
    }
    
    Tensor Conv3dLayer::backward(const Tensor &d)
    {
        Tensor d_unstrided;
        if (stride.x > 1 || stride.y > 1 || stride.z > 1)
        {
            TupleNd<3> L(X.NthSize(2), X.rowSize(), X.colSize());
            TupleNd<3> Lo = L-dilation*(kernel_size - 1) + padding*2;
            auto ushape = d.getShape();
            if (ushape.size() > 0) ushape.end()[-1] = Lo.x;
            if (ushape.size() > 1) ushape.end()[-2] = Lo.y;
            if (ushape.size() > 2) ushape.end()[-3] = Lo.z;
            d_unstrided = d.dilated(ushape);
        }

        auto dd = stride.x == 1 && stride.y == 1 && stride.z == 1 ? d.asShapeOneInsert(3) : d_unstrided.asShapeOneInsert(3);
        auto XX = X.asShapeOneInsert(4);

        Tensor grad_X = dd.convolution3d(kernels, dilation*(kernel_size - 1) - padding, 1, dilation, ZERO, 4, true);
        Tensor grad_k = XX.correlation3d(dd, padding, 1, 1, ZERO, 5, true);
        Tensor grad_b = d.sum(0).sum(1).sum(2).sum(4, true);

        if (dilation.x > 1 || dilation.y > 1 || dilation.z > 1) grad_k.shrink(kernels.getShape(), true);

        optimizer->grad(k_id) += grad_k;
        optimizer->grad(b_id) += grad_b;
        
        return grad_X;
    }

    uint64_t Conv3dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Conv3d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(k_id) + sizeof(b_id)
                      + sizeof(in_ch) + sizeof(out_ch) 
                      + sizeof(kernel_size.x) + sizeof(kernel_size.y) + sizeof(kernel_size.z)
                      + sizeof(stride.x)      + sizeof(stride.y)      + sizeof(stride.z)
                      + sizeof(padding.x)     + sizeof(padding.y)     + sizeof(padding.z)
                      + sizeof(dilation.x)    + sizeof(dilation.y)    + sizeof(dilation.z)
                      + sizeof(pm);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&k_id, sizeof(k_id));
        file.write((char*)&b_id, sizeof(b_id));
        file.write((char*)&in_ch, sizeof(in_ch));
        file.write((char*)&out_ch, sizeof(out_ch));
        file.write((char*)&kernel_size.x, sizeof(kernel_size.x));
        file.write((char*)&kernel_size.y, sizeof(kernel_size.y));
        file.write((char*)&kernel_size.z, sizeof(kernel_size.z));
        file.write((char*)&stride.x, sizeof(stride.x));
        file.write((char*)&stride.y, sizeof(stride.y));
        file.write((char*)&stride.z, sizeof(stride.z));
        file.write((char*)&padding.x, sizeof(padding.x));
        file.write((char*)&padding.y, sizeof(padding.y));
        file.write((char*)&padding.z, sizeof(padding.z));
        file.write((char*)&dilation.x, sizeof(dilation.x));
        file.write((char*)&dilation.y, sizeof(dilation.y));
        file.write((char*)&dilation.z, sizeof(dilation.z));
        file.write((char*)&pm, sizeof(pm));
 
        size += kernels.save(file);
        size += bias.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

}
