#include "Tensor.h"
#include "OpenCLManager.h"

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

    DirectTensorView::~DirectTensorView()
    {
        b = nullptr;
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

    DirectTensorView &DirectTensorView::operator=(Tensor &&t)
    {
        if (!sizeMatch(this->shape, t.shape))
            throw std::length_error("Tensor sizes not matching in assignment operation");
            
        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

        return *this;
    }

    DirectTensorView &DirectTensorView::operator+=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_left_broadcast_assign<fn, Kernel::T_TENSOR_ADD>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    DirectTensorView &DirectTensorView::operator-=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        ew_or_left_broadcast_assign<fn, Kernel::T_TENSOR_SUB>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    DirectTensorView &DirectTensorView::operator*=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  *  n2; };
        ew_or_left_broadcast_assign<fn, Kernel::T_TENSOR_MUL>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    DirectTensorView &DirectTensorView::operator/=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_left_broadcast_assign<fn, Kernel::T_TENSOR_DIV>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }
    
    /**
     * @brief executes fn() on two broadcastable shape tensors element wise
     *        and the result is stored on the first tensor
     *
     * @tparam fn function to be executed
     * @param t1 source and destination tensor
     * @param t2 source tensor
     * @param err_msg error message to display in case of non broadcastable shapes
     */
    template <float64 (*fn)(float64, float64), size_t fn_device>
    void Tensor::ew_or_left_broadcast_assign(Tensor &t1, const Tensor &t2, const char *err_msg)
    {
        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            for (size_t i = 0; i < t1.size; i++)
                t1.b[i] = fn(t1.b[i], t2.b[i]);
        }
        else if (t1.broadcastable(t1.shape, t2.shape))
        {
            auto shapeT1 = t1.shape;
            auto shapeT2 = t2.shape;

            if (shapeT1.size() > shapeT2.size())
                for (size_t i = 0; shapeT1.size() != shapeT2.size(); i++)
                    shapeT2.insert(shapeT2.begin(), 1);
            else
                for (size_t i = 0; shapeT1.size() != shapeT2.size(); i++)
                    shapeT1.insert(shapeT1.begin(), 1);

            std::vector<size_t> shapeDst(shapeT1.size());
            bool self_assign = true;
            for (size_t i = 0; i < shapeT1.size(); i++)
            {
                shapeDst[i] = shapeT1[i] == 1 ? shapeT2[i] : shapeT1[i];
                if (shapeDst[i] != shapeT1[i])
                    self_assign = false;
            }

            if (self_assign)
            {
                if (t1.size)
                    broadcast_ew_assign<fn>(t1.b, t1.b, t2.b, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
            }
            else
                throw std::length_error(err_msg);
        }
        else
            throw std::length_error(err_msg);
    }
    
    template <float64 (*fn)(float64, float64)>
    void Tensor::broadcast_ew_assign(float64* dst, const float64* src1, const float64* src2,
                                    const size_t *shape, const size_t *shape1, const size_t *shape2,
                                    size_t depth,
                                    size_t off, size_t off1, size_t off2)
    {
        if (depth > 1)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                broadcast_ew_assign<fn>(dst, src1, src2, shape + 1, shape1 + 1, shape2 + 1, depth - 1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
        else
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                dst[off * *shape + i] = fn(src1[off1 * *shape1 + (i & bdc1)], src2[off2 * *shape2 + (i & bdc2)]);
    }

}