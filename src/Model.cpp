#include "Model.h"
#include "With.h"
#include <chrono>
#include <string>
#include <matplot/matplot.h>

namespace RedFish
{
    Model::Model(const std::vector<Layer::Descriptor> &layers, uint32_t loss, uint32_t optimizer)
        : optimizer(make_optimizer(optimizer)), loss(make_loss(loss))
    {
        for (auto &layer : layers)
        {
            this->layers.push_back(make_layer(layer));
            this->layers.back()->useOptimizer(*this->optimizer);
        }
    }

    Model::Model(const std::vector<Layer *> &layers, Loss* loss, Optimizer *optimizer)
        : layers(layers), optimizer(optimizer), loss(loss)
    {
        for (auto layer : this->layers)
            layer->useOptimizer(*this->optimizer);
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
        {
            layers.push_back(make_layer(file));
            /* layers.back()->useOptimizer(*this->optimizer); */
        }

    }

    Model::~Model()
    {
        if (optimizer) delete optimizer;
        if (loss) delete loss;
        for (auto l : layers)
            if (l) delete l;
    }

    void Model::train(const Tensor &in, const Tensor &out, uint32_t epochs, double learning_rate, size_t mini_batch_size, bool smooth_lr)
    {
        using namespace matplot;
        With w(Layer::training, true);
        size_t train_time = 0, batching_time = 0;
        float64 /* avg_loss = 0., */ a = .5, b = .5;
        std::vector<float64> avg_loss(1,0);

        long long ttime = 0, btime = 0;

        float64 lrstep = learning_rate/5, lrdecay = .99, lr = learning_rate/5;
        if (smooth_lr)
            optimizer->setLearningRate(lr);
        else
            optimizer->setLearningRate(learning_rate);

        size_t training_samples_count = in.getShape()[0];
        size_t batch_count = training_samples_count / mini_batch_size;
        auto mbsi = in.getShape();
        auto mbso = out.getShape();
        mbsi[0] = mbso[0] = mini_batch_size;

        Tensor mini_batch_out(mbso);
        Tensor fw_res(mbsi);
        
        std::cout << "Epoch - - loss: -" << std::endl;

        for (size_t i = 0; i < epochs; i++)
        {
            for (size_t k = 0; k < batch_count; k++)
            {
                auto begin = std::chrono::high_resolution_clock::now();
                fw_res.resize(mbsi);
                for (size_t j = 0; j < mini_batch_size; j++)
                {
                    fw_res        .sliceLastNDims({j},  in.getShape().size() - 1) =  in.sliceLastNDims({j+k*mini_batch_size},  in.getShape().size() - 1);
                    mini_batch_out.sliceLastNDims({j}, out.getShape().size() - 1) = out.sliceLastNDims({j+k*mini_batch_size}, out.getShape().size() - 1);
                }
                auto end = std::chrono::high_resolution_clock::now();
                btime += (end - begin).count();

                begin = std::chrono::high_resolution_clock::now();

                for (size_t j = 0; j < layers.size(); j++)
                    fw_res = layers[j]->forward(fw_res);

                float64 lossV = loss->forward(fw_res, mini_batch_out);
                if (avg_loss.size() == 0) avg_loss.push_back(lossV);
                else avg_loss.push_back(a*avg_loss.back() + b*lossV);

                Tensor grad = loss->backward(fw_res, mini_batch_out);
                for (size_t j = 0; j < layers.size(); j++)
                    grad = layers.end()[-j - 1]->backward(grad);

                optimizer->step();
                if (smooth_lr) optimizer->setLearningRate(lr);
                /* std::cout << "lr: " << lr << std::endl; */

                if (i*batch_count+k < 4) lr += lrstep;
                /* else lr *= lrdecay; */

                end = std::chrono::high_resolution_clock::now();
                ttime += (end - begin).count();

                std::cout << "\r\033[1F\x1b[2KEpoch " << i << "." << k << " - loss: " << lossV << " - filt. loss: " << avg_loss.back() << std::endl;
                semilogy(avg_loss);
            }
        }


        std::cout << "Trainig time: " << ttime * 1e-9 << "s\n";
        std::cout << "Mini-batch generation time: " << btime * 1e-9 << "s\n";
    }

    double Model::test(const Tensor &in, const Tensor &out, std::function<double(const Tensor &, const Tensor &)> accuracy)
    {
        With w(Layer::training, false);
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
        With w(Layer::training, false);
        Tensor fw_res = layers.front()->forward(in);
        for (size_t j = 1; j < layers.size(); j++)
            fw_res = layers[j]->forward(fw_res);

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
