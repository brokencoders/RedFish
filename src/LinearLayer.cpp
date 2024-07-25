#include "LinearLayer.h"

namespace RedFish {
    
    LinearLayer::LinearLayer(size_t input_size, size_t output_size, Optimizer* optimizer) 
        : W({input_size, output_size}), b({output_size}), optimizer(optimizer)
    {
        W.randUniform(-.5, .5);
        b.randUniform(-.5, .5);
        W_id = optimizer->allocateParameter(W);
        b_id = optimizer->allocateParameter(b);
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

        file.read((char*)&W_id, sizeof(W_id));
        file.read((char*)&b_id, sizeof(b_id));
 
        W = Tensor(file);
        b = Tensor(file);
    }

    Tensor LinearLayer::forward(const Tensor &X)
    {
        return X.matmul(W) + b;
    }

    Tensor LinearLayer::backward(const Tensor &X, const Tensor &d)
    {
        Tensor grad_X = d.matmul(W, Transpose::RIGHT);
        Tensor grad_W = X.matmul(d, Transpose::LEFT);
        Tensor grad_b = d.sum(1);

        optimizer->updateParameter(W_id, W, grad_W);
        optimizer->updateParameter(b_id, b, grad_b);

        return grad_X;
    }

    uint64_t LinearLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Linear";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(W_id) + sizeof(b_id);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&W_id, sizeof(W_id));
        file.write((char*)&b_id, sizeof(b_id));
 
        size += W.save(file);
        size += b.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    void LinearLayer::print()
    {
        std::cout << "W = \n" << W << "b = " << b << "\n";
    }

}
