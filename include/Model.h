#pragma once 

#include "LinearLayer.h"
#include "Loss.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include "Tensor.h"

namespace RedFish {

    class Model {
    public:
        Model(int input_size, const std::vector<LayerDesc>& layers, const Loss* loss, const Optimizer* optimizer);
        Model(const std::string& file_path, const Loss* loss, const Optimizer* optimizer);

        void train(const Tensor& in, const Tensor& out, uint32_t epochs = 100, double learning_rate = .01, int mini_batch_size = 3);
        double test(const Tensor& in, const Tensor& out, std::function<double(const Tensor&, const Tensor&)> accuracy);
        Tensor estimate(const Tensor& in);

        int save(const std::string& file_path);

    private:
        uint32_t input_size;
        std::vector<LinearLayer> layers;
        const Loss* loss;
    };

    inline Model::Model(int input_size, const std::vector<LayerDesc> &l, const Loss* loss, const Optimizer* optimizer) 
        :input_size(input_size), loss(loss)
    {
        layers.reserve(l.size());
        for (auto& layer : l)
            layers.emplace_back(input_size, 
                                layer.neuron_count, 
                                layer.activation_func,
                                optimizer), 
            input_size = layer.neuron_count; 
        
    }

    inline Model::Model(const std::string &file_path, const Loss* loss, const Optimizer* optimizer)
        : loss(loss)
    {
        std::ifstream file(file_path, std::ios::binary);

        file.read((char*)&input_size, 4);

        uint32_t layer_count;
        file.read((char*)&layer_count, 4);
        layers.reserve(layer_count);

        uint32_t layer_input_size = input_size;

        for (size_t i = 0; i < layer_count; i++)
        {
            file.read((char*)&layer_input_size, 4);

            uint32_t neuron_count;
            file.read((char*)&neuron_count, 4);

            layers.emplace_back(layer_input_size, neuron_count, optimizer);

            for (auto& neuron : layers.back().neurons)
            {
                file.read((char*)&neuron.bias, 8);
                file.read((char*)&neuron.weights(0UL), 8 * layer_input_size);
            }

            layer_input_size = layers.back().neurons.size();
        }

    }

    inline void Model::train(const Tensor &in, const Tensor &out, uint32_t epochs, double learning_rate, int mini_batch_size)
    {
        for (int i = 0; i < epochs; i++)
        {
            Tensor mini_batch_in(0, in.colSize());
            Tensor mini_batch_out(0, out.colSize());
            for (int j = 0; j < mini_batch_size; j++)
            {
                int n = rand() % in.rowSize();
                mini_batch_in.vstack(in.getRow(n));
                mini_batch_out.vstack(out.getRow(n));
            }

            std::vector<Tensor> fw_res;

            fw_res.reserve(layers.size());

            fw_res.push_back(layers.front().farward(mini_batch_in));
            for (size_t j = 1; j < layers.size(); j++)
                fw_res.push_back(layers[j].farward(fw_res[j - 1]));

            double lossV = loss->farward(fw_res.back(), mini_batch_out);
            std::cout << "Epoch " << i << " - loss: " << lossV << "\n";

            auto bw_res = layers.back().backward(fw_res[fw_res.size()-2], loss->backward(fw_res[fw_res.size()-1], mini_batch_out));
            for (size_t j = layers.size() - 2; j > 0; j--)
                bw_res = layers[j].backward(fw_res[j-1], bw_res);
            
            bw_res = layers.front().backward(mini_batch_in, bw_res);
        }
    }

    inline double Model::test(const Tensor &in, const Tensor &out, std::function<double(const Tensor&, const Tensor&)> accuracy)
    {
        Tensor ris = this->estimate(in);
        double sum = 0;
        for (size_t i = 0; i < in.rowSize(); i++)
        {
            sum += accuracy(ris.getRow(i), out.getRow(i));
        }

        return sum / in.rowSize();
    }

    inline Tensor Model::estimate(const Tensor &in)
    {
        auto fw_res = layers.front().farward(in);
        for (size_t j = 1; j < layers.size(); j++)
            fw_res = layers[j].farward(fw_res);

        return fw_res;
    }

    inline int Model::save(const std::string &file_path)
    {
        std::ofstream file(file_path, std::ios::binary);

        file.write((char*)&input_size, 4);

        uint32_t layer_count = layers.size();
        file.write((char*)&layer_count, 4);

        uint32_t layer_input_size = input_size;

        for (auto& layer : layers)
        {
            file.write((char*)&layer_input_size, 4);

            uint32_t neuron_count = layer.biases.getSize();
            file.write((char*)&neuron_count, 8);

            file.write((char*)&layer.biases(0UL),  8 * layer.biases.getSize());
            file.write((char*)&layer.weights(0UL), 8 * layer.weights.getSize());
            
            layer_input_size = layer.weights.rowSize();
        }

        return 0;
    }
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