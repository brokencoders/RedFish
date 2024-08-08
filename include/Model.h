#pragma once 

#include "LinearLayer.h"
#include "RecurrentLayer.h"
#include "ConvLayer.h"
#include "ActivationLayer.h"
#include "FlattenLayer.h"
#include "DropoutLayer.h"
#include "MaxPoolLayer.h"
#include "Optimizer.h"
#include "Loss.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include "Tensor.h"

namespace RedFish {

    class Model {
    public:
        Model(const std::vector<Layer::Descriptor>& layers, uint32_t loss = SQUARE_LOSS, uint32_t optimizer = ADAM_OPT);
        Model(const std::vector<Layer*>& layers, Loss* loss, Optimizer* optimizer);
        Model(const std::string& file_path);
        ~Model();

        void train(const Tensor& in, const Tensor& out, uint32_t epochs = 100, double learning_rate = .01, size_t mini_batch_size = 3);
        double test(const Tensor& in, const Tensor& out, std::function<double(const Tensor&, const Tensor&)> accuracy);
        Tensor estimate(const Tensor& in);

        uint64_t save(const std::string& file_path, bool release);

    // private:
        std::vector<Layer*> layers;
        Optimizer* optimizer;
        Loss* loss;
    };

}

/**
 * 4 Byte: Input Size
 * 4 Byte: Layer Count
 * 
 * for each layer:
 *  4 Byte: Input Size 
 *  4 Byte: Neuron Count 
 *  4 Byte: Activation Function -> (See enum number)
 *  for each Neuron:
 *   8 Byte: Bias
 *   8 Byte x Input: Size Weights
 * 
 */