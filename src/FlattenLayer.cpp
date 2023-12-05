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

    Tensor FlattenLayer::farward(const Tensor& X)
    {
        Tensor flatten(X);
        std::vector<size_t> new_dim;
        
        size_t dim = 1;
        for (size_t i = 0; i < X.getShape().size(); i++)
        {
            if (i >= start_dim && i <= end_dim )
            {
                dim *= X.getShape()[i];
                if(i == end_dim || i == X.getShape().size() - 1)
                    new_dim.push_back(dim);
            }
            else 
                new_dim.push_back(X.getShape()[i]);
        }
        

        flatten.reshape(new_dim);
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