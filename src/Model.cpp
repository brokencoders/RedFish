#include "Model.h"
#include <chrono>

namespace RedFish
{

    Model::Model(const std::vector<Layer::Descriptor> &layers, uint32_t loss, uint32_t optimizer)
        : optimizer(make_optimizer(optimizer)), loss(make_loss(loss))
    {
        for (auto &layer : layers)
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
        auto mbsi = in.getShape();
        auto mbso = out.getShape();
        mbsi[0] = mbso[0] = mini_batch_size;
        for (size_t i = 0; i < epochs; i++)
        {
            auto batch_begin = std::chrono::high_resolution_clock::now();
            Tensor mini_batch_in(mbsi);
            Tensor mini_batch_out(mbso);
            for (size_t j = 0; j < mini_batch_size; j++)
            {
                size_t n = rand() % in.getShape()[0];
                mini_batch_in.sliceLastNDims({j}, in.getShape().size() - 1) = in.sliceLastNDims({n}, in.getShape().size() - 1);
                mini_batch_out.sliceLastNDims({j}, out.getShape().size() - 1) = out.sliceLastNDims({n}, out.getShape().size() - 1);
            }
            auto train_begin = std::chrono::high_resolution_clock::now();

            std::vector<Tensor> fw_res;

            fw_res.reserve(layers.size());

            fw_res.emplace_back(layers.front()->farward(mini_batch_in));
            for (size_t j = 1; j < layers.size(); j++)
                fw_res.emplace_back(layers[j]->farward(fw_res[j - 1]));

            double lossV = loss->farward(fw_res.back(), mini_batch_out);

            std::cout << "Epoch " << i << " - loss: " << lossV << std::endl;

            Tensor bw_res = layers.back()->backward(fw_res[fw_res.size() - 2], loss->backward(fw_res.back(), mini_batch_out));
            for (size_t j = layers.size() - 2; j > 0; j--)
                bw_res = layers[j]->backward(fw_res[j - 1], bw_res);

            layers.front()->backward(mini_batch_in, bw_res);

            optimizer->step();
            auto train_end = std::chrono::high_resolution_clock::now();
            train_time += (train_end - train_begin).count();
            batching_time += (train_begin - batch_begin).count();
        }
        auto end = std::chrono::high_resolution_clock::now();
        size_t tot_time = (end - begin).count();
        printf("Training time: %fs (%f%)\nBatch generation time: %fs (%f%)\nTotal time: %fs\n",
               train_time * 1e-9, 100. * train_time / tot_time,
               batching_time * 1e-9, 100. * batching_time / tot_time,
               tot_time * 1e-9);
    }

    double Model::test(const Tensor &in, const Tensor &out, std::function<double(const Tensor &, const Tensor &)> accuracy)
    {
        Tensor ris = this->estimate(in);
        double sum = 0;
        for (size_t i = 0; i < in.getShape()[0]; i++)
        {
            sum += accuracy(ris.sliceLastNDims({i}, ris.getShape().size() - 1), out.sliceLastNDims({i}, out.getShape().size() - 1));
        }

        return sum / in.getShape()[0];
    }

    Tensor Model::estimate(const Tensor &in)
    {
        Tensor fw_res = layers.front()->farward(in);
        for (size_t j = 1; j < layers.size(); j++)
            fw_res = layers[j]->farward(fw_res);

        return fw_res;
    }

    uint64_t Model::save(const std::string &file_path, bool optimizer)
    {
        std::ofstream file(file_path, std::ios::binary);

        const char name[] = "RedFishDeepModel";
        file.write(name, sizeof(name));

        auto spos = file.tellp();
        uint64_t size = sizeof(uint64_t);
        file.write((char*)&size, sizeof(size));
        
        uint64_t layer_count = layers.size();
        file.write((char*)&layer_count, sizeof(layer_count));

        for (size_t i = 0; i < layers.size(); i++)
            size += layers[i]->save(file);

        size += loss->save(file);
        if(optimizer)
            size += this->optimizer->save(file);

        file.seekp(spos);
        file.write((char*)&size, sizeof(size));

        return size + sizeof(name) + sizeof(size);
    }
}

/**
 * Model
 *
 */

/**
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

 */