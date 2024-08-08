#include "DropoutLayer.h"

namespace RedFish 
{
    DropoutLayer::DropoutLayer(float64 rate)
        :rate(rate)
    {
    }

    DropoutLayer::DropoutLayer(std::ifstream &file)
    {
        const std::string name = "Layer::Dropout";
        char rname[sizeof("Layer::Dropout")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in DropoutLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&rate, sizeof(rate));
    }

    Tensor DropoutLayer::forward(const Tensor& X)
    {
        if (training)
        {
            this->X = X;
            mask.resize(X.getShape());
            mask.randBernulli(1.-rate);
            mask *= 1./(1.-rate);
            return X*mask;
        }
        else return X;
    }

    Tensor DropoutLayer::backward(const Tensor& d)
    {
        return d*mask;
    }

    uint64_t DropoutLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Dropout";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(rate);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&rate, sizeof(rate));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }
}