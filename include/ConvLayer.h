#pragma once 

#include "Tensor.h"
#include "Layer.h"

namespace RedFish {

    class Conv1dLayer : public Layer {
    public:
        Conv1dLayer(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, int64_t padding = 0, size_t dilation = 1, PaddingMode pm = ZERO);
        Conv1dLayer(std::ifstream& file);
        void useOptimizer(Optimizer& optimizer) override;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    private:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch, kernel_size;
        size_t stride;
        int64_t padding;
        size_t dilation;
        PaddingMode pm;

    };
    
    class Conv2dLayer : public Layer {
    public:
        Conv2dLayer(size_t in_channels, size_t out_channels, TupleNd<2> kernel_size, TupleNd<2> stride = 1, TupleNd<2, int64_t> padding = 0, TupleNd<2> dilation = 1, PaddingMode pm = ZERO);
        Conv2dLayer(std::ifstream& file);
        void useOptimizer(Optimizer& optimizer) override;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    private:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch;
        TupleNd<2> kernel_size;
        TupleNd<2> stride;
        TupleNd<2, int64_t> padding;
        TupleNd<2> dilation;
        PaddingMode pm;

    };
    
    class Conv3dLayer : public Layer {
    public:
        Conv3dLayer(size_t in_channels, size_t out_channels, TupleNd<3> kernel_size, TupleNd<3> stride = 1, TupleNd<3, int64_t> padding = 0, TupleNd<3> dilation = 1, PaddingMode pm = ZERO);
        Conv3dLayer(std::ifstream& file);
        void useOptimizer(Optimizer& optimizer) override;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& d) override;

        uint64_t save(std::ofstream& file) const override;

    private:
        Tensor kernels, bias;
        size_t k_id, b_id;
        size_t in_ch, out_ch;
        TupleNd<3> kernel_size;
        TupleNd<3> stride;
        TupleNd<3, int64_t> padding;
        TupleNd<3> dilation;
        PaddingMode pm;

    };
}