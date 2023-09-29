#pragma once 

#include "Activation.h"
#include "LinearLayer.h"
#include <iostream>
#include <fstream>

namespace RedFish {

    struct Layer {
        int neuron_count;
        Activation::AF activation_func;
    };

    class Model {
    public:
        Model(int input_size, const std::vector<Layer>& layers);
        Model(const std::string& file_path);

        void train(const Algebra::Matrix& in, const Algebra::Matrix& out, uint epochs = 100, double learning_rate = .01, int mini_batch_size = 3);
        double test(const Algebra::Matrix& in, const Algebra::Matrix& out, std::function<double(const Algebra::Matrix&, const Algebra::Matrix&)> accuracy);
        Algebra::Matrix estimate(const Algebra::Matrix& in);

        int save(const std::string& file_path);

    private:
        uint32_t input_size;
        std::vector<LinearLayer> layers;
    };

    RedFish::Model::Model(int input_size, const std::vector<Layer> &l) 
        :input_size(input_size)
    {
        layers.reserve(l.size());
        for (auto& layer : l)
            layers.emplace_back(input_size, 
                                layer.neuron_count, 
                                layer.activation_func), 
            input_size = layer.neuron_count; 
        
    }

    RedFish::Model::Model(const std::string &file_path)
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
            Activation::AF af;
            file.read((char*)&neuron_count, 4);
            file.read((char*)&af, 4);

            layers.emplace_back(layer_input_size, neuron_count, af);

            for (auto& neuron : layers.back().neurons)
            {
                file.read((char*)&neuron.bias, 8);
                file.read((char*)&neuron.weights(0), 8 * layer_input_size);
            }

            layer_input_size = layers.back().neurons.size();
        }

    }

    inline void Model::train(const Algebra::Matrix &in, const Algebra::Matrix &out, uint epochs, double learning_rate, int mini_batch_size)
    {
        for (int i = 0; i < epochs; i++)
        {
            Algebra::Matrix mini_batch_in(0, in.cols());
            Algebra::Matrix mini_batch_out(0, out.cols());
            for (int j = 0; j < mini_batch_size; j++)
            {
                int n = random() % in.rows();
                mini_batch_in.vstack(in.getRow(n));
                mini_batch_out.vstack(out.getRow(n));
            }   

            std::vector<Algebra::Matrix> fw_res;

            fw_res.reserve(layers.size());

            fw_res.push_back(layers.front().farward(mini_batch_in));
            for (size_t j = 1; j < layers.size(); j++)
                fw_res.push_back(layers[j].farward(fw_res[j - 1]));

            double loss = (fw_res.back() - mini_batch_out).normSquare() / mini_batch_out.rows();
            std::cout << "Epoch " << i << " - loss: " << loss << "\n";

            auto bw_res = layers.back().backward(fw_res[fw_res.size()-2], fw_res[fw_res.size()-1] - mini_batch_out, learning_rate);
            for (size_t j = layers.size() - 2; j > 0; j--)
                bw_res = layers[j].backward(fw_res[j-1], bw_res, learning_rate);
            
            bw_res = layers.front().backward(mini_batch_in, bw_res, learning_rate);
        }
    }

    inline double Model::test(const Algebra::Matrix &in, const Algebra::Matrix &out, std::function<double(const Algebra::Matrix&, const Algebra::Matrix&)> accuracy)
    {
        Algebra::Matrix ris = this->estimate(in);
        double sum = 0;
        for (size_t i = 0; i < in.rows(); i++)
        {
            sum += accuracy(ris.getRow(i), out.getRow(i));
        }

        return sum / in.rows();
    }

    inline Algebra::Matrix Model::estimate(const Algebra::Matrix &in)
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

            uint32_t neuron_count = layer.neurons.size();
            file.write((char*)&neuron_count, 4);

            file.write((char*)&layer.af, 4);

            for (auto& neuron : layer.neurons)
            {
                file.write((char*)&neuron.bias, 8);
                file.write((char*)&neuron.weights(0), 8 * layer_input_size);
            }

            layer_input_size = layer.neurons.size();
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