#include "Optimizer.h"

namespace RedFish
{

    Adam::Adam() : mw(), vw(), im1(1 / (1 - b1)), im2(1 / (1 - b2)), learning_rate(.01), t(1) {}

    size_t Adam::allocateParameter(const Tensor& t)
    {
        mw.emplace_back(empty_like(t));
        vw.emplace_back(empty_like(t));
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



    Optimizer* make_optimizer(uint32_t o)
    {
        switch (o)
        {
        case ADAM_OPTIMIZER: return new Adam();
                
        default: return nullptr;
        }
    }

}