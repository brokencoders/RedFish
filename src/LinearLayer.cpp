#include "LinearLayer.h"

namespace RedFish {
    
    LinearLayer::LinearLayer(size_t input_size, size_t output_size) 
        : W({input_size, output_size}), b({output_size})
    {
        float64 stdv = 1. / std::sqrt(input_size);
        W.randUniform(-stdv, stdv);
        b.zero();
    }

    LinearLayer::LinearLayer(std::ifstream &file)
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

    void LinearLayer::useOptimizer(Optimizer &optimizer)
    {
        if (this->optimizer)
        {
            this->optimizer->deleteParameters(W_id);
            this->optimizer->deleteParameters(b_id);
        }
        W_id = optimizer.allocateParameters(W);
        b_id = optimizer.allocateParameters(b);
        this->optimizer = &optimizer;
    }

    Tensor LinearLayer::forward(const Tensor &X)
    {
        if (training) this->X = X;
        return X.matmul(W) + b;
    }

    Tensor LinearLayer::backward(const Tensor &d)
    {
        Tensor grad_X = d.matmul(W, Transpose::RIGHT);
        Tensor grad_W = X.matmul(d, Transpose::LEFT);
        Tensor grad_b = d.sum(1);
        optimizer->grad(W_id) += grad_W;
        optimizer->grad(b_id) += grad_b;

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

}
