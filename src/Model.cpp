#include "Model.h"
#include <chrono>
#include <string>
#include <matplot/matplot.h>

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
        using namespace matplot;
        size_t train_time = 0, batching_time = 0;
        float64 last_losses5[5] = {0,0,0,0,0};
        float64 last_losses10[10] = {0,0,0,0,0,0,0,0,0,0};
        std::vector<double> loss1, loss5, loss10, iter;

        auto begin = std::chrono::high_resolution_clock::now();

        optimizer->setLearningRate(learning_rate);

        size_t training_samples_count = in.getShape()[0];
        size_t batch_count = training_samples_count / mini_batch_size;
        auto mbsi = in.getShape();
        auto mbso = out.getShape();
        mbsi[0] = mbso[0] = mini_batch_size;

        Tensor mini_batch_out(mbso);
        std::vector<Tensor> fw_res(layers.size() + 1);
        fw_res[0].resize(mbsi);
        
        std::cout << "Epoch - - loss: -" << std::endl;

        for (size_t i = 0; i < epochs; i++)
        {
            for (size_t k = 0; k < batch_count; k++)
            {
                for (size_t j = 0; j < mini_batch_size; j++)
                {
                    fw_res[0]     .sliceLastNDims({j},  in.getShape().size() - 1) =  in.sliceLastNDims({j+k*mini_batch_size},  in.getShape().size() - 1);
                    mini_batch_out.sliceLastNDims({j}, out.getShape().size() - 1) = out.sliceLastNDims({j+k*mini_batch_size}, out.getShape().size() - 1);
                }

                for (size_t j = 0; j < layers.size(); j++)
                    fw_res[j + 1] = layers[j]->farward(fw_res[j]);

                float64 lossV = loss->farward(fw_res.back(), mini_batch_out);

                Tensor grad = loss->backward(fw_res.back(), mini_batch_out);
                for (size_t j = 0; j < layers.size(); j++)
                    grad = layers.end()[-j - 1]->backward(fw_res.end()[-j - 2], grad);

                optimizer->step();


                last_losses5[(i*batch_count + k) % 5] = lossV;
                last_losses10[(i*batch_count + k) % 10] = lossV;
                float64 avg_loss5 = 0, avg_loss10 = 0;
                for (size_t id = 0; id <= std::min<size_t>(4, i*batch_count + k); id++) avg_loss5  += last_losses5[id]  / std::min<float64>(5,  i*batch_count + k +1);
                for (size_t id = 0; id <= std::min<size_t>(9, i*batch_count + k); id++) avg_loss10 += last_losses10[id] / std::min<float64>(10, i*batch_count + k +1);
                iter.push_back(i*batch_count + k);
                loss1.push_back(lossV);
                loss5.push_back(avg_loss5);
                loss10.push_back(avg_loss10);

                std::cout << "\r\033[1F\x1b[2KEpoch " << i << "." << k << " - loss: " << lossV << " - avg. loss (5 samples): " << avg_loss5 << std::endl;
                plot(iter, loss1);
                hold(on);
                plot(iter, loss5);
                plot(iter, loss10);
                hold(off);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Trainig time: " << (end - begin).count() * 1e-9 << "s\n";
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
