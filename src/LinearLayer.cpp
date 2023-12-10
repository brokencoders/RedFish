#include "LinearLayer.h"

namespace RedFish {
    
    LinearLayer::LinearLayer(size_t input_size, size_t neuron_count, Optimizer* optimizer) 
        : weights({input_size, neuron_count}), biases({neuron_count}), optimizer(optimizer)
    {
        weights.rand(-.5, .5);
        biases.rand(-.5, .5);
        w_id = optimizer->allocateParameter(weights);
        b_id = optimizer->allocateParameter(biases);
    }

    LinearLayer::LinearLayer(std::ifstream &file, Optimizer* optimizer)
        : optimizer(optimizer)
    {
        const std::string name = "Layer::Linear";
        char rname[sizeof("Layer::Linear")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in LinearLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&w_id, sizeof(w_id));
        file.read((char*)&b_id, sizeof(b_id));
 
        weights = Tensor(file);
        biases  = Tensor(file);
    }

    Tensor LinearLayer::farward(const Tensor &X)
    {
        return X.matmul(weights) + biases;
    }

    Tensor LinearLayer::backward(const Tensor &X, const Tensor &d)
    {
        const float64 lambda = .001;
        Tensor dX = d.matmul(weights, Transpose::RIGHT);
        Tensor grad = X.matmul(d, Transpose::LEFT) /* + weights * lambda */;
        Tensor bias_grad = d.sum((size_t)1) /* + biases * lambda */;

        optimizer->updateParameter(w_id, weights, grad);
        optimizer->updateParameter(b_id, biases, bias_grad);

        return dX;
    }

    uint64_t LinearLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Linear";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(w_id) + sizeof(b_id);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&w_id, sizeof(w_id));
        file.write((char*)&b_id, sizeof(b_id));
 
        size += weights.save(file);
        size += biases.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    void LinearLayer::print()
    {
        std::cout << "w = \n" << weights << "b = " << biases << "\n";
    }

}
