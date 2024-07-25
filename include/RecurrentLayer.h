#pragma once 

#include "Layer.h"
#include "Optimizer.h"
#include "Tensor.h"

#include "ActivationLayer.h"

namespace RedFish {
    
    template<typename Act>
    class RecurrentLayer : public Layer  {
    public:
        RecurrentLayer(size_t input_size, size_t output_size, Optimizer* optimizer);
        RecurrentLayer(std::ifstream& file, Optimizer* optimizer);

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& X, const Tensor& d) override;
        uint64_t save(std::ofstream& file) const override;

        void print();

    public:
        Tensor Wh, Wi, b, Y, h;
        size_t Wh_id, Wi_id, b_id;
        Optimizer* optimizer;
        Act f;
    };

}
