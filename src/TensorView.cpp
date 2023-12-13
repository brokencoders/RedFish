#include "Tensor.h"

namespace RedFish
{
    DirectTensorView::DirectTensorView(const std::vector<size_t>& new_shape, float64* ptr)
        : Tensor({0})
    {
        shape = new_shape;
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];
        b = ptr;
    }

    DirectTensorView &DirectTensorView::operator=(const Tensor &t)
    {
        if (!sizeMatch(this->shape, t.shape))
            throw std::length_error("Tensor sizes not matching in assignment operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

        return *this;
    }

    DirectTensorView &DirectTensorView::operator=(const DirectTensorView &t)
    {
        if (!sizeMatch(this->shape, t.shape))
            throw std::length_error("Tensor sizes not matching in assignment operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

        return *this;
    }

    Tensor &DirectTensorView::operator+=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    Tensor &DirectTensorView::operator-=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    Tensor &DirectTensorView::operator*=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  *  n2; };
        ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    Tensor &DirectTensorView::operator/=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }
}