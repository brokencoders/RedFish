#include "Optimizer.h"

namespace RedFish
{

    Adam::Adam() : mw(), vw(), im1(1 / (1 - b1)), im2(1 / (1 - b2)), learning_rate(.01), t(1) {}

    size_t Adam::allocateParameter(const Tensor& t)
    {
        mw.emplace_back(zeros_like(t));
        vw.emplace_back(zeros_like(t));
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

    Optimizer* make_optimizer(uint32_t o)
    {
        switch (o)
        {
        case ADAM_OPTIMIZER: return new Adam();
                
        default: return nullptr;
        }
    }

}