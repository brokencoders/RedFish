#include "MaxPoolLayer.h"
#include <limits>

namespace RedFish {
    MaxPool1dLayer::MaxPool1dLayer(size_t kernel_size, size_t stride, size_t padding, size_t dilation) 
        :kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation) {  }

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

    // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    Tensor MaxPool1dLayer::farward(const Tensor &X)
    {
        if (X.getShape().size() < 3)
            throw std::length_error("Invalid size of X in MaxPool1d backward");
            
        for (size_t i = 0; i + 3 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in MaxPool1d farward");

        size_t N_size = X.getShape().end()[-3];
        size_t C_size = X.getShape().end()[-2];
        size_t L_size = X.getShape().end()[-1];

        size_t L_out_size = (L_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        Tensor max_pool({N_size, C_size, L_out_size});

        for (size_t i = 0; i < N_size; i++)
            for (size_t j = 0; j < C_size; j++)
                for (size_t k = 0; k < L_out_size; k++)
                {
                    float64 max = -std::numeric_limits<float64>::infinity();
                    for (size_t m = 0; m < kernel_size; m++)
                        if (max < X(i, j, stride * k + m * dilation))
                            max = X(i, j, stride * k + m * dilation);
                    max_pool(i, j, k) = max;
                }

        return max_pool;
    }

    Tensor MaxPool1dLayer::backward(const Tensor &X, const Tensor &d)
    {
        if (X.getShape().size() < 3)
            throw std::length_error("Invalid size of X in MaxPool1d backward");
        if (d.getShape().size() < 3)
            throw std::length_error("Invalid size of d in MaxPool1d backward");

        for (size_t i = 1; i < 3; i++)
            if (X.getShape().end()[-i] != d.getShape().end()[-i])
                throw std::length_error("Invalid size of X or d in MaxPool1d backward");

        size_t N_size = X.getShape().end()[-3];
        size_t C_size = X.getShape().end()[-2];
        size_t L_size = X.getShape().end()[-1];

        size_t L_out_size = (L_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        Tensor grad = Tensor::empty_like(X);
        grad.zero();

        for (size_t i = 0; i < N_size; i++)
            for (size_t j = 0; j < C_size; j++)
                for (size_t k = 0; k < L_out_size; k++)
                {
                    float64 max = -std::numeric_limits<float64>::infinity();
                    size_t index = 0;
                    for (size_t m = 0; m < kernel_size; m++)
                    {
                        if (max < X(i, j, stride * k + m * dilation))
                        {
                            max = X(i, j, stride * k + m * dilation);
                            index = stride * k + m * dilation;
                        }
                    }
                    grad(i, j, index) += d(i, j, k);
                }
        return grad;
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

    MaxPool2dLayer::MaxPool2dLayer(Tuple2d kernel_size, Tuple2d stride, Tuple2d padding, Tuple2d dilation) 
        :kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation) {  }

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

    Tensor MaxPool2dLayer::farward(const Tensor &X)
    {
        if (X.getShape().size() < 4)
            throw std::length_error("Invalid size of X in MaxPool2d backward");
            

        for (size_t i = 0; i + 4 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in MaxPool1d farward");


        size_t N_size = X.getShape().end()[-4];
        size_t C_size = X.getShape().end()[-3];
        size_t H_size = X.getShape().end()[-2];
        size_t W_size = X.getShape().end()[-1];
    
        size_t H_out_size = (H_size + 2 * padding.y - dilation.y * (kernel_size.y - 1) - 1) / stride.y + 1;
        size_t W_out_size = (W_size + 2 * padding.x - dilation.x * (kernel_size.x - 1) - 1) / stride.x + 1;
    
        Tensor max_pool({N_size, C_size, H_out_size, W_out_size});

        for (size_t i = 0; i < N_size; i++)
            for (size_t j = 0; j < C_size; j++)
                for (size_t h = 0; h < H_out_size; h++)
                    for (size_t w = 0; w < W_out_size; w++)
                    {
                        float64 max = -std::numeric_limits<float64>::infinity();
                        for (size_t m = 0; m < kernel_size.y; m++)
                            for (size_t n = 0; n < kernel_size.x; n++)
                                if (max < X(i, j, stride.y * h + m * dilation.y, stride.x * w + n * dilation.x))
                                    max = X(i, j, stride.y * h + m * dilation.y, stride.x * w + n * dilation.x);
                        max_pool(i, j, h, w) = max;
                    }        
        return max_pool;
    }

    Tensor MaxPool2dLayer::backward(const Tensor &X, const Tensor &d)
    {
        if (X.getShape().size() < 4)
            throw std::length_error("Invalid size of X in MaxPool2d backward");
        if (d.getShape().size() < 4)
            throw std::length_error("Invalid size of X in MaxPool2d backward");
            

        for (size_t i = 0; i + 4 < X.getShape().size(); i++)
            if (X.getShape()[i] != 1)
                throw std::length_error("Invalid batch size in MaxPool1d farward");


        size_t N_size = X.getShape().end()[-4];
        size_t C_size = X.getShape().end()[-3];
        size_t H_size = X.getShape().end()[-2];
        size_t W_size = X.getShape().end()[-1];
    
        size_t H_out_size = (H_size + 2 * padding.y - dilation.y * (kernel_size.y - 1) - 1) / stride.y + 1;
        size_t W_out_size = (W_size + 2 * padding.x - dilation.x * (kernel_size.x - 1) - 1) / stride.x + 1;
    
        Tensor grad = Tensor::empty_like(X);
        grad.zero();

        for (size_t i = 0; i < N_size; i++)
            for (size_t j = 0; j < C_size; j++)
                for (size_t h = 0; h < H_out_size; h++)
                    for (size_t w = 0; w < W_out_size; w++)
                    {
                        float64 max = -std::numeric_limits<float64>::infinity();
                        Tuple2d index = 0;
                        for (size_t m = 0; m < kernel_size.y; m++)
                            for (size_t n = 0; n < kernel_size.x; n++)
                            {
                                if (max < X(i, j, stride.y * h + m * dilation.y, stride.x * w + n * dilation.x))
                                {
                                    max = X(i, j, stride.y * h + m * dilation.y, stride.x * w + n * dilation.x);
                                    index.y = stride.y * h + m * dilation.y;
                                    index.x = stride.x * w + n * dilation.x;
                                }
                            }
                        grad(i, j, index.y, index.x) += max;
                    }        
        return grad; 
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

    MaxPool3dLayer::MaxPool3dLayer(Tuple3d kernel_size, Tuple3d stride, Tuple3d padding, Tuple3d dilation)
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

    Tensor MaxPool3dLayer::farward(const Tensor &X)
    {
        return Tensor();
    }
    
    Tensor MaxPool3dLayer::backward(const Tensor &X, const Tensor &d)
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