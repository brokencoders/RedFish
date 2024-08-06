#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class Conv1dLayer : public Layer {
    public:
        Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, Optimizer* optimizer, size_t stride = 1, size_t padding = 0, size_t dilation = 1, PaddingMode pm = ZERO);
        Conv1dLayer(std::ifstream& file, Optimizer* optimizer);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    public:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch, kernel_size;
        size_t stride;
        size_t padding;
        size_t dilation;
        PaddingMode pm;
        Optimizer* optimizer;

    };
    
    class Conv2dLayer : public Layer {
    public:
        Conv2dLayer(size_t in_channels, size_t out_channels, Tuple2d kernel_size, Optimizer* optimizer, Tuple2d stride = 1, Tuple2d padding = 0, Tuple2d dilation = 1, PaddingMode pm = ZERO);
        Conv2dLayer(std::ifstream& file, Optimizer* optimizer);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    public:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch;
        Tuple2d kernel_size;
        Tuple2d stride;
        Tuple2d padding;
        Tuple2d dilation;
        PaddingMode pm;
        Optimizer* optimizer;

    };
    
    class Conv3dLayer : public Layer {
    public:
        Conv3dLayer(size_t in_channels, size_t out_channels, Tuple3d kernel_size, Optimizer* optimizer, Tuple3d stride = 1, Tuple3d padding = 0, Tuple3d dilation = 1, PaddingMode pm = ZERO);
        Conv3dLayer(std::ifstream& file, Optimizer* optimizer);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    public:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch;
        Tuple3d kernel_size;
        Tuple3d stride;
        Tuple3d padding;
        Tuple3d dilation;
        PaddingMode pm;
        Optimizer* optimizer;

    };
}