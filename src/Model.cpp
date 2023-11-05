#include "Model.h"
#include <chrono>

namespace RedFish {

    Model::Model(const std::vector<Layer::Descriptor>& layers, uint32_t loss, uint32_t optimizer) 
        : optimizer(make_optimizer(optimizer)), loss(make_loss(loss))
    {
        for (auto& layer : layers)
            this->layers.push_back(make_layer(layer, this->optimizer));
    }

    /* Model::Model(const std::string &file_path, const Loss* loss, const Optimizer* optimizer)
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

    } */

    void Model::train(const Tensor &in, const Tensor &out, uint32_t epochs, double learning_rate, size_t mini_batch_size)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        size_t train_time = 0, batching_time = 0;
        optimizer->setLearningRate(learning_rate);
        for (size_t i = 0; i < epochs; i++)
        {
            auto batch_begin = std::chrono::high_resolution_clock::now();
            Tensor mini_batch_in( {mini_batch_size,  in.colSize()});
            Tensor mini_batch_out({mini_batch_size, out.colSize()});
            for (size_t j = 0; j < mini_batch_size; j++)
            {
                size_t n = rand() % in.rowSize();
                for (size_t i = 0; i < in.colSize(); i++)
                    mini_batch_in(j,i)  = in(n,i);

                for (size_t i = 0; i < out.colSize(); i++)
                    mini_batch_out(j,i) = out(n,i);
            }
            auto train_begin = std::chrono::high_resolution_clock::now();

            std::vector<Tensor> fw_res;

            fw_res.reserve(layers.size());

            fw_res.emplace_back(layers.front()->farward(mini_batch_in));
            for (size_t j = 1; j < layers.size(); j++)
                fw_res.emplace_back(layers[j]->farward(fw_res[j - 1]));

            double lossV = loss->farward(fw_res.back(), mini_batch_out);
            std::cout << "Epoch " << i << " - loss: " << lossV << std::endl;

            Tensor bw_res = layers.back()->backward(fw_res[fw_res.size()-2], loss->backward(fw_res.back(), mini_batch_out));
            for (size_t j = layers.size() - 2; j > 0; j--)
                bw_res = layers[j]->backward(fw_res[j-1], bw_res);
            
            layers.front()->backward(mini_batch_in, bw_res);

            optimizer->step();
            auto train_end = std::chrono::high_resolution_clock::now();
            train_time += (train_end - train_begin).count();
            batching_time += (train_begin - batch_begin).count();
        }
        auto end = std::chrono::high_resolution_clock::now();
        size_t tot_time = (end - begin).count();
        printf("Training time: %fs (%f%)\nBatch generation time: %fs (%f%)\nTotal time: %fs\n",
                train_time    * 1e-9, 100. * train_time / tot_time,
                batching_time * 1e-9, 100. * batching_time / tot_time,
                tot_time      * 1e-9);
    }

    double Model::test(const Tensor &in, const Tensor &out, std::function<double(const Tensor&, const Tensor&)> accuracy)
    {
        Tensor ris = this->estimate(in);
        double sum = 0;
        for (size_t i = 0; i < in.rowSize(); i++)
        {
            Tensor ris_i({1,ris.colSize()}), out_i({1, out.colSize()});
            for (size_t j = 0; j < ris.colSize(); j++) ris_i(j) = ris(i,j);
            for (size_t j = 0; j < out.colSize(); j++) out_i(j) = out(i,j);
            
            sum += accuracy(ris_i, out_i);
        }

        return sum / in.rowSize();
    }

    Tensor Model::estimate(const Tensor &in)
    {
        Tensor fw_res = layers.front()->farward(in);
        for (size_t j = 1; j < layers.size(); j++)
            fw_res = layers[j]->farward(fw_res);

        return fw_res;
    }

    /* int Model::save(const std::string &file_path)
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
    } */
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