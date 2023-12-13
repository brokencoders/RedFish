#include "DropoutLayer.h"

namespace RedFish 
{
    DropoutLayer::DropoutLayer(float64 rate)
        :rate(rate) { }

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
        file.read((char*)&skip_size, sizeof(skip_size));
        file.read((char*)&batch_size, sizeof(batch_size));
        file.read((char*)&factor, sizeof(factor));
    }

    Tensor DropoutLayer::farward(const Tensor& X)
    {
        if(output.getShape().size() == 0)
        {
            output = Tensor::empty_like(X);

            batch_size = X.getShape()[0];
            
            skip_size = 1;
            for (size_t i = 1; i < X.getShape().size(); i++)
                skip_size *= X.getShape()[i];

            factor = 1 / (1 - rate);
        }
        std::mt19937 gen(Tensor::getRandomDevice()());
        std::bernoulli_distribution d(rate);

        for (size_t i = 0; i < skip_size; i++)
        {
            float64 t = 1;
            if(d(gen) == true)
                t = 0;
            
            for (size_t j = 0; j < batch_size; j++) 
                output(i + j * skip_size) = t * factor * X(i + j * skip_size);
        }

        return output;
    }

    Tensor DropoutLayer::backward(const Tensor& X, const Tensor& d)
    {
        Tensor grad = Tensor::empty_like(X);
        
        for(size_t i = 0; i < batch_size * skip_size; i++)
        {
            if (output(i) == 0) grad(i) = 0;
            else grad(i) = d(i) * factor;
        }

        return grad;
    }

    uint64_t DropoutLayer::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Dropout";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(rate) + sizeof(skip_size)
                      + sizeof(batch_size) + sizeof(factor);

        file.write((char*)&size, sizeof(size));

        file.write((char*)&rate, sizeof(rate));
        file.write((char*)&skip_size, sizeof(skip_size));
        file.write((char*)&batch_size, sizeof(batch_size));
        file.write((char*)&factor, sizeof(factor));
 
        return size + sizeof(uint64_t) + sizeof(name);
    }
}