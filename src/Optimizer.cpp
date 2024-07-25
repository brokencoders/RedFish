#include "Optimizer.h"
#include <string>

namespace RedFish
{
    
    SGD::SGD(float64 weight_decay, float64 momentum, float64 dampening, bool nesterov)
        : b(), weight_decay(weight_decay), momentum(momentum), dampening(dampening), learning_rate(.01), nesterov(nesterov), t(1) {}

    SGD::SGD(std::ifstream &file)
    {
        const std::string name = "Optimizer::SGD";
        char rname[sizeof("Optimizer::SGD")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in SGD(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&weight_decay,  sizeof(weight_decay));
        file.read((char*)&momentum,      sizeof(momentum));
        file.read((char*)&dampening,     sizeof(dampening));
        file.read((char*)&learning_rate, sizeof(learning_rate));
        file.read((char*)&t,             sizeof(t));
 
        uint64_t b_size = b.size();
        file.read((char*)&b_size, sizeof(b_size));

        b.reserve(b_size);

        for (size_t i = 0; i < b_size; i++)
            b.emplace_back(file);
    }

    size_t SGD::allocateParameter(const Tensor& t)
    {
        if (momentum) b.emplace_back(Tensor::empty_like(t));
        return b.size() - 1;
    }

    void SGD::updateParameter(size_t i, Tensor& value, const Tensor& grad)
    {
        if (weight_decay || momentum)
        {
            Tensor mgrad(grad);
            if (weight_decay)
                mgrad += weight_decay*value;
            if (momentum)
            {
                if (t) b[i] = momentum*b[i] + (1-dampening)*mgrad;
                else   b[i] = mgrad;
                if (nesterov) mgrad += momentum*b[i];
                else mgrad = b[i];
            }
            value -= learning_rate*mgrad;
        }
        else value -= learning_rate*grad;
    }

    void SGD::step()
    {
        t++;
    }
    
    void SGD::setLearningRate(float64 lr)
    {
        learning_rate = lr;
    }

    uint64_t SGD::save(std::ofstream &file) const
    {
        const char name[] = "Optimizer::SGD";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(weight_decay) + sizeof(momentum) + sizeof(dampening) + sizeof(learning_rate) + sizeof(t);

        size += sizeof(uint64_t); 

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&weight_decay,  sizeof(weight_decay));
        file.write((char*)&momentum,      sizeof(momentum));
        file.write((char*)&dampening,     sizeof(dampening));
        file.write((char*)&learning_rate, sizeof(learning_rate));
        file.write((char*)&t,             sizeof(t));
 
        uint64_t b_size = b.size();
        file.write((char*)&b_size, sizeof(b_size));

        for (size_t i = 0; i < b.size(); i++)
            size += b[i].save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    Adam::Adam() : mw(), vw(), im1(1 / (1 - b1)), im2(1 / (1 - b2)), learning_rate(.01), t(1) {}

    Adam::Adam(std::ifstream &file)
    {
        const std::string name = "Optimizer::Adam";
        char rname[sizeof("Optimizer::Adam")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Adam(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&im1, sizeof(im1));
        file.read((char*)&im2, sizeof(im2));
        file.read((char*)&learning_rate, sizeof(learning_rate));
        file.read((char*)&t, sizeof(t));
 
        uint64_t mw_size = mw.size();
        file.read((char*)&mw_size, sizeof(mw_size));

        mw.reserve(mw_size);
        vw.reserve(mw_size);

        for (size_t i = 0; i < mw_size; i++)
        {
            mw.emplace_back(file);
            vw.emplace_back(file);
        }
    }

    size_t Adam::allocateParameter(const Tensor& t)
    {
        mw.emplace_back(Tensor::zeros_like(t));
        vw.emplace_back(Tensor::zeros_like(t));
        return mw.size() - 1;
    }

    void Adam::updateParameter(size_t i, Tensor& value, const Tensor& grad)
    {
        mw[i] *= b1;
        vw[i] *= b2;
        mw[i] += grad      * one_minus_b1;
        vw[i] += grad*grad * one_minus_b2; 

        Tensor m_hat = mw[i] * im1; 
        Tensor v_hat = vw[i] * im2;

        value -= learning_rate * m_hat / (std::sqrt(v_hat) - epsilon);
    }

    void Adam::step()
    {
        t++;
        im1 = 1 / (1 - std::pow(b1, t));
        im2 = 1 / (1 - std::pow(b2, t));
    }
    
    void Adam::setLearningRate(float64 lr)
    {
        learning_rate = lr;
    }

    // im1
    // im2
    // Lr
    // t
    // dim [mw[0], vw[0] ... ]

    uint64_t Adam::save(std::ofstream &file) const
    {
        const char name[] = "Optimizer::Adam";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(im1) + sizeof(im2) + sizeof(learning_rate) + sizeof(t);

        size += sizeof(uint64_t); 

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&im1, sizeof(im1));
        file.write((char*)&im2, sizeof(im2));
        file.write((char*)&learning_rate, sizeof(learning_rate));
        file.write((char*)&t, sizeof(t));
 
        uint64_t mw_size = mw.size();
        file.write((char*)&mw_size, sizeof(mw_size));

        for (size_t i = 0; i < mw.size(); i++)
        {
            size += mw[i].save(file);
            size += vw[i].save(file);
        }

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    Optimizer* make_optimizer(const uint32_t o)
    {
        switch (o)
        {
        case OPTIMIZER::ADAM_OPT: return new Adam();
        case OPTIMIZER::SGD_OPT:  return new SGD();
                
        default: return nullptr;
        }
    }

    Optimizer* make_optimizer(std::ifstream &file)
    {
        auto spos = file.tellg();
        char name[128] = "", c = 1;
        for (size_t i = 0; i < sizeof(name) && c != '\0'; name[i] = c, i++)
            file.get(c);

        file.seekg(spos);
        
        const std::string opt = "Optimizer::";
        if (std::string(name).find_first_of(opt) != 0)
            throw std::runtime_error("Invalid file content in make_optimizer(std::ifstram&)");

        const std::string opt_name = name + sizeof("Optimizer::") - 1;

        if (opt_name == "Adam") return new Adam(file);

        return nullptr;
    }
}