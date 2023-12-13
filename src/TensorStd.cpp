#include "Tensor.h" 

namespace std
{

    RedFish::Tensor sqrt(const RedFish::Tensor &t)
    {
        return RedFish::forEach<std::sqrt>(t);
    }

    RedFish::Tensor exp(const RedFish::Tensor &t)
    {
        return RedFish::forEach<std::exp>(t);
    }

    RedFish::Tensor log(const RedFish::Tensor &t)
    {
        return RedFish::forEach<std::log>(t);
    }

    RedFish::Tensor pow(const RedFish::Tensor &t, RedFish::float64 power)
    {
        RedFish::Tensor ret = t.empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = std::pow(t.b[i], power);

        return ret;
    }

    RedFish::Tensor pow(const RedFish::Tensor &t, const RedFish::Tensor &power)
    {
        if (!t.sizeMatch(t.shape, power.shape))
            throw std::length_error("Tensor sizes not matching in std::pow operation");

        RedFish::Tensor ret = t.empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = std::pow(t.b[i], power.b[i]);

        return ret;
    }

}
