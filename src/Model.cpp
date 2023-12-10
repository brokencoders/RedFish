#include "Model.h"
#include <chrono>
#include <string>

namespace RedFish
{

    Model::Model(const std::vector<Layer::Descriptor> &layers, uint32_t loss, uint32_t optimizer)
        : optimizer(make_optimizer(optimizer)), loss(make_loss(loss))
    {
        for (auto &layer : layers)
            this->layers.push_back(make_layer(layer, this->optimizer));
    }

    Model::Model(const std::string &file_path)
        : optimizer(nullptr), loss(nullptr)
    {
        std::ifstream file(file_path, std::ios::binary);

        const std::string name = "RedFishDeepModel";
        char rname[sizeof("RedFishDeepModel")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Model(const std::string&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        bool release = true;
        file.read((char*)&release, sizeof(release));

        if (!release)
        {
            this->optimizer = make_optimizer(file);
            this->loss = make_loss(file);
        }

        uint64_t layer_count = 0;
        file.read((char*)&layer_count, sizeof(layer_count));

        layers.reserve(layer_count);
        for (size_t i = 0; i < layer_count; i++)
            layers.push_back(make_layer(file, this->optimizer));

    }

    Model::~Model()
    {
        if (optimizer) delete optimizer;
        if (loss) delete loss;
        for (auto l : layers)
            if (l) delete l;
    }

    void Model::train(const Tensor &in, const Tensor &out, uint32_t epochs, double learning_rate, size_t mini_batch_size)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        size_t train_time = 0, batching_time = 0;
        optimizer->setLearningRate(learning_rate);
        auto mbsi = in.getShape();
        auto mbso = out.getShape();
        mbsi[0] = mbso[0] = mini_batch_size;
        std::cout << "Epoch - - loss: -" << std::endl;
        float64 last_losses[5] = {0,0,0,0,0};
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

            float64 lossV = loss->farward(fw_res.back(), mini_batch_out);
            last_losses[i % 5] = lossV;
            float64 avg_loss = 0;
            for (size_t i = 0; i < 5; i++) avg_loss += last_losses[i] / 5.;

            std::cout << "\r\033[1F\x1b[2KEpoch " << i << " - loss: " << lossV << " - avg. loss (5 samples): " << avg_loss << std::endl;

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

    uint64_t Model::save(const std::string &file_path, bool release)
    {
        std::ofstream file(file_path, std::ios::binary);

        const char name[] = "RedFishDeepModel";
        file.write(name, sizeof(name));

        auto spos = file.tellp();
        uint64_t size = sizeof(uint64_t) + sizeof(release);
        file.write((char*)&size, sizeof(size));

        file.write((char*)&release, sizeof(release));
        
        if (!release)
        {
            size += this->optimizer->save(file);
            size += this->loss->save(file);
        }

        uint64_t layer_count = layers.size();
        file.write((char*)&layer_count, sizeof(layer_count));

        for (size_t i = 0; i < layers.size(); i++)
            size += layers[i]->save(file);

        file.seekp(spos);
        file.write((char*)&size, sizeof(size));

        return size + sizeof(name) + sizeof(size);
    }
}
