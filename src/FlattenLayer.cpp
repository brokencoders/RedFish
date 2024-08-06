#include "FlattenLayer.h"

namespace RedFish {

    FlattenLayer::FlattenLayer(size_t start_dim, size_t end_dim)
        :start_dim(start_dim), end_dim(end_dim) { }

    FlattenLayer::FlattenLayer(std::ifstream &file)
    {
        const std::string name = "Layer::Flatten";
        char rname[sizeof("Layer::Flatten")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in FlattenLayer(std::ifstream&)");
        
        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&start_dim, sizeof(start_dim));
        file.read((char*)&end_dim, sizeof(end_dim));
    }

    Tensor FlattenLayer::forward(const Tensor& X)
    {
        Tensor flatten(X);
        auto new_shape = X.getShape();
        size_t start = start_dim >= new_shape.size() ? 0 : new_shape.size() - start_dim - 1;
        size_t end   =   end_dim >= new_shape.size() ? 0 : new_shape.size() -   end_dim - 1;
        if (end_dim == (size_t)-1) end = new_shape.size() - 1;
        
        for (size_t i = start; i < end; i++)
            new_shape[end] *= new_shape[i];

        new_shape.erase(new_shape.begin() + start, new_shape.begin() + end);

        flatten.reshape(new_shape);
        return flatten;
    }

    Tensor FlattenLayer::backward(const Tensor& X, const Tensor& d)
    {
        Tensor grad(d);
        grad.reshape(X.getShape());
        return grad;
    }

    uint64_t FlattenLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Flatten";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(start_dim) + sizeof(end_dim);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&start_dim, sizeof(start_dim));
        file.write((char*)&end_dim, sizeof(end_dim));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }
}