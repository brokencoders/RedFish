#include "MaxPoolLayer.h"
#include <limits>
#include <numeric>

namespace RedFish {
    MaxPool1dLayer::MaxPool1dLayer(size_t kernel_size, size_t stride, size_t padding, size_t dilation) 
        :kernel_size(kernel_size), stride(stride ? stride : kernel_size), padding(padding), dilation(dilation) {  }

    MaxPool1dLayer::MaxPool1dLayer(std::ifstream &file)
    {
        const std::string name = "Layer::MaxPool1d";
        char rname[sizeof("Layer::MaxPool1d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in MaxPool1dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&kernel_size, sizeof(kernel_size));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
    }

    Tensor MaxPool1dLayer::forward(const Tensor &X)
    {
        if (training) this->X = X;
        auto shape = X.getShape();
        if (shape.size() == 0) shape.push_back(1);

        size_t L_size = shape.back();
        size_t L_out_size = (L_size + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        shape.back() = L_out_size;

        Tensor max_pool(shape);
        size_t size = 1;
        for (size_t i = 0; i < shape.size() - 1; i++) size *= shape[i];
        

        for (size_t i = 0; i < size; i++)
            for (size_t k = 0; k < L_out_size; k++)
            {
                float64 max = X(i*L_size + stride * k);
                for (size_t m = 1; m < kernel_size; m++)
                    if (max < X(i*L_size + stride * k + m * dilation))
                        max = X(i*L_size + stride * k + m * dilation);
                max_pool(i*L_out_size + k) = max;
            }

        return max_pool;
    }

    Tensor MaxPool1dLayer::backward(const Tensor &d)
    {
        auto shape = X.getShape();
        if (shape.size() == 0) shape.push_back(1);

        size_t L_size = shape.back();
        size_t L_out_size = (L_size + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        
        size_t size = 1;
        for (size_t i = 0; i < shape.size() - 1; i++) size *= shape[i];
        
        Tensor grad_X = Tensor::zeros_like(X);

        for (size_t i = 0; i < size; i++)
            for (size_t k = 0; k < L_out_size; k++)
            {
                float64 max = X(i*L_size + stride * k);
                size_t max_idx = 0;
                for (size_t m = 1; m < kernel_size; m++)
                    if (max < X(i*L_size + stride * k + m * dilation))
                        max = X(i*L_size + stride * k + m * dilation),
                        max_idx = m;

                grad_X(i*L_size + stride * k + max_idx * dilation) += d(i*L_out_size + k);
            }

        return grad_X;
    }

    uint64_t MaxPool1dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::MaxPool1d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(kernel_size) + sizeof(stride)
                      + sizeof(padding) + sizeof(dilation);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&kernel_size, sizeof(kernel_size));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }

    MaxPool2dLayer::MaxPool2dLayer(TupleNd<2> kernel_size, TupleNd<2> stride, TupleNd<2> padding, TupleNd<2> dilation) 
        :kernel_size(kernel_size), stride({stride.y ? stride.y : kernel_size.y, stride.x ? stride.x : kernel_size.x}), padding(padding), dilation(dilation) {  }

    MaxPool2dLayer::MaxPool2dLayer(std::ifstream &file)
    {
        const std::string name = "Layer::MaxPool2d";
        char rname[sizeof("Layer::MaxPool2d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in MaxPool2dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&kernel_size, sizeof(kernel_size));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
    }

    Tensor MaxPool2dLayer::forward(const Tensor &X)
    {
        if (training) this->X = X;
        auto shape = X.getShape();
        if (shape.size() < 2) shape.insert(shape.begin(), 2 - shape.size(), 1);

        size_t H_size = shape.end()[-2];
        size_t W_size = shape.end()[-1];
    
        size_t H_out_size = (H_size + 2 * padding.y - dilation.y * (kernel_size.y - 1) - 1) / stride.y + 1;
        size_t W_out_size = (W_size + 2 * padding.x - dilation.x * (kernel_size.x - 1) - 1) / stride.x + 1;

        shape.end()[-2] = H_out_size;
        shape.end()[-1] = W_out_size;
    
        Tensor max_pool(shape);
        size_t size = 1;
        for (size_t i = 0; i < shape.size() - 2; i++) size *= shape[i];

        for (size_t i = 0; i < size; i++)
            for (size_t h = 0; h < H_out_size; h++)
                for (size_t w = 0; w < W_out_size; w++)
                {
                    float64 max = X((i*H_size + stride.y * h)*W_size + stride.x * w);
                    for (size_t m = 0; m < kernel_size.y; m++)
                        for (size_t n = 0; n < kernel_size.x; n++)
                            if (max < X((i*H_size + stride.y * h + m * dilation.y)*W_size + stride.x * w + n * dilation.x))
                                max = X((i*H_size + stride.y * h + m * dilation.y)*W_size + stride.x * w + n * dilation.x);

                    max_pool((i*H_out_size + h)*W_out_size + w) = max;
                }

        return max_pool;
    }

    Tensor MaxPool2dLayer::backward(const Tensor &d)
    {
        auto shape = X.getShape();
        if (shape.size() < 2) shape.insert(shape.begin(), 2 - shape.size(), 1);

        size_t H_size = shape.end()[-2];
        size_t W_size = shape.end()[-1];
    
        size_t H_out_size = (H_size + 2 * padding.y - dilation.y * (kernel_size.y - 1) - 1) / stride.y + 1;
        size_t W_out_size = (W_size + 2 * padding.x - dilation.x * (kernel_size.x - 1) - 1) / stride.x + 1;

        shape.end()[-2] = H_out_size;
        shape.end()[-1] = W_out_size;
    
        size_t size = 1;
        for (size_t i = 0; i < shape.size() - 2; i++) size *= shape[i];

        Tensor grad_X = Tensor::zeros_like(X);

        for (size_t i = 0; i < size; i++)
            for (size_t h = 0; h < H_out_size; h++)
                for (size_t w = 0; w < W_out_size; w++)
                {
                    float64 max = X((i*H_size + stride.y * h)*W_size + stride.x * w);
                    size_t max_m = 0, max_n = 0;
                    for (size_t m = 0; m < kernel_size.y; m++)
                        for (size_t n = 0; n < kernel_size.x; n++)
                            if (max < X((i*H_size + stride.y * h + m * dilation.y)*W_size + stride.x * w + n * dilation.x))
                                max = X((i*H_size + stride.y * h + m * dilation.y)*W_size + stride.x * w + n * dilation.x),
                                max_m = m, max_n = n;

                    grad_X((i*H_size + stride.y * h + max_m * dilation.y)*W_size + stride.x * w + max_n * dilation.x) += d((i*H_out_size + h)*W_out_size + w);
                }

        return grad_X;
    }

    uint64_t MaxPool2dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::MaxPool2d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(kernel_size) + sizeof(stride)
                      + sizeof(padding) + sizeof(dilation);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&kernel_size, sizeof(kernel_size));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }

    MaxPool3dLayer::MaxPool3dLayer(TupleNd<3> kernel_size, TupleNd<3> stride, TupleNd<3> padding, TupleNd<3> dilation)
        :kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation)
    {
    }

    MaxPool3dLayer::MaxPool3dLayer(std::ifstream &file)
    {
        const std::string name = "Layer::MaxPool3d";
        char rname[sizeof("Layer::MaxPool3d")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in MaxPool3dLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&kernel_size, sizeof(kernel_size));
        file.read((char*)&stride, sizeof(stride));
        file.read((char*)&padding, sizeof(padding));
        file.read((char*)&dilation, sizeof(dilation));
    }

    Tensor MaxPool3dLayer::forward(const Tensor &X)
    {
        if (training) this->X = X;
        return Tensor();
    }
    
    Tensor MaxPool3dLayer::backward(const Tensor &d)
    {
        return Tensor();
    }

    uint64_t MaxPool3dLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::MaxPool3d";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(kernel_size) + sizeof(stride)
                      + sizeof(padding) + sizeof(dilation);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&kernel_size, sizeof(kernel_size));
        file.write((char*)&stride, sizeof(stride));
        file.write((char*)&padding, sizeof(padding));
        file.write((char*)&dilation, sizeof(dilation));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }
}