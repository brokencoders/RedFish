#pragma once

#define CHECK_BOUNDS

#include <iostream>
#include <limits.h>
#include <vector>
#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>

namespace RedFish
{
    class Tensor;
    typedef double float64;
}

namespace std
{
    RedFish::Tensor sqrt(const RedFish::Tensor &);
    RedFish::Tensor exp(const RedFish::Tensor &);
    RedFish::Tensor log(const RedFish::Tensor &);
    RedFish::Tensor pow(const RedFish::Tensor &, RedFish::float64);
    RedFish::Tensor pow(const RedFish::Tensor &, const RedFish::Tensor &);
}

namespace RedFish
{
    using Buffer = size_t;
    typedef double float64;
    class DirectTensorView;

    enum Transpose : int8_t
    {
        LEFT,
        RIGHT,
        NONE
    };
    enum PaddingMode : int8_t
    {
        ZERO,
        REFLECT,
        REPLICATE,
        CIRCULAR
    };

    struct Tuple2d
    {
        Tuple2d(size_t y, size_t x) : y(y), x(x) {}
        Tuple2d(size_t n) : y(n), x(n) {}
        Tuple2d() : y(0), x(0) {}
        union { size_t y, h; };
        union { size_t x, w; };
    };

    struct Tuple3d
    {
        Tuple3d(size_t z, size_t y, size_t x) : z(z), y(y), x(x) {}
        Tuple3d(size_t n) : z(n), y(n), x(n) {}
        Tuple3d() : y(0), x(0) {}
        union { size_t z, d; };
        union { size_t y, h; };
        union { size_t x, w; };
    };

    template <void (*fn)(float64 &, float64)>
    Tensor op_along_axes(const Tensor &, size_t, const float64);
    template <void (*fn)(float64 &, float64)>
    float64 op_along_all_axes(const Tensor &, const float64);
    template <float64 (*fn)(float64, float64)>
    void broadcast_ew_assign(Tensor &, const Tensor &, const Tensor &, const size_t *, const size_t *, const size_t *, size_t, size_t = 0, size_t = 0, size_t = 0);
    template <auto fn, typename... Args>
    void broadcast_op(float64 *, const float64 *, const float64 *, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, Args...);
    template <float64 (*fn)(float64, float64)>
    Tensor ew_or_broadcast(const Tensor &, const Tensor &, const char *);
    template <float64 (*fn)(float64, float64)>
    void ew_or_broadcast_assign(Tensor &, const Tensor &, const char *);
    template <float64 (*fn)(float64, float64)>
    void ew_or_left_broadcast_assign(Tensor &, const Tensor &, const char *);
    void copy_2d(float64 *, float64 *, Tuple2d, size_t, size_t);
    Tensor operator+(const float64 val, const Tensor &t);
    Tensor operator-(const float64 val, const Tensor &t);
    Tensor operator*(const float64 val, const Tensor &t);
    Tensor operator/(const float64 val, const Tensor &t);


    class Tensor
    {
    private:
        static std::random_device rd;
        static std::default_random_engine gen;

    public:
        Tensor(const std::vector<size_t> &shape = {});
        Tensor(const size_t *shape, size_t len);
        Tensor(const std::vector<size_t> &shape, float64 *buff, bool copy = true);
        Tensor(const Tensor &t);  // Copy Constructor
        Tensor(Tensor &&t);       // Move Constructor
        Tensor(const std::vector<size_t> &shape, std::initializer_list<float64> data);
        Tensor(std::ifstream& file);

        // Operetors
        Tensor &operator=(const Tensor &t);
        Tensor &operator=(Tensor &&t);
        Tensor  operator+(const Tensor &t) const;
        Tensor  operator+(const float64 val) const;
        Tensor& operator+=(const Tensor &t);
        Tensor& operator+=(const float64 val);
        Tensor  operator-(const Tensor &t) const;
        Tensor  operator-(const float64 val) const;
        Tensor  operator-() const;
        Tensor& operator-=(const Tensor &t);
        Tensor& operator-=(const float64 val);
        Tensor  operator*(const Tensor &t) const;
        Tensor  operator*(const float64 val) const;
        Tensor& operator*=(const Tensor &t);
        Tensor& operator*=(const float64 val);
        Tensor  operator/(const Tensor &t) const;
        Tensor  operator/(const float64 val) const;
        Tensor& operator/=(const Tensor &t);
        Tensor& operator/=(const float64 val);
        Tensor  operator==(const Tensor &other) const;
        Tensor  operator>=(const Tensor &other) const;
        Tensor  operator<=(const Tensor &other) const;
        Tensor  operator>(const Tensor &other) const;
        Tensor  operator<(const Tensor &other) const;

        float64& operator()(const std::vector<size_t> &index);
        float64  operator()(const std::vector<size_t> &index) const;

        template <typename... Args>
        float64& operator()(Args... indices);
        template <typename... Args>
        float64  operator()(Args... indices) const;

        DirectTensorView getRow(const std::vector<size_t> &index);
        const DirectTensorView getRow(const std::vector<size_t> &index) const;
        DirectTensorView getMatrix(const std::vector<size_t> &index);
        const DirectTensorView getMatrix(const std::vector<size_t> &index) const;
        template <size_t N>
        DirectTensorView sliceLastNDims(const std::vector<size_t> &index);
        template <size_t N>
        const DirectTensorView sliceLastNDims(const std::vector<size_t> &index) const;
        DirectTensorView sliceLastNDims(const std::vector<size_t> &index, size_t N);
        const DirectTensorView sliceLastNDims(const std::vector<size_t> &index, size_t N) const;

        void resize(const std::vector<size_t> &new_shape);
        void reshape(const std::vector<size_t> &new_shape);

        Tensor T() const;
        Tensor T(size_t dimension1, size_t dimension2);

        void zero();
        void ones();
        void costant(float64 val);
        void randUniform(float64 a = 0.0, float64 b = 1.0);
        void randNormal(float64 mean = 0.0, float64 std = 1.0);

        float64 squareSum() const;
        Tensor  squareSum(size_t dimension) const;
        float64 max() const;
        Tensor  max(size_t dimension) const;
        float64 min() const;
        Tensor  min(size_t dimension) const;
        float64 sum() const;
        Tensor  sum(size_t dimension) const;
        
        Tensor matmul(const Tensor &t, const Transpose transpose = NONE) const;
        Tensor crossCorrelation1d(const Tensor &kernel, size_t  padding = 0, size_t  stride = 1, size_t  dilation = 1, PaddingMode pm = ZERO) const;
        Tensor crossCorrelation2d(const Tensor &kernel, Tuple2d padding = 0, Tuple2d stride = 1, Tuple2d dilation = 1, PaddingMode pm = ZERO) const;
        Tensor crossCorrelation3d(const Tensor &kernel, Tuple3d padding = 0, Tuple3d stride = 1, Tuple3d dilation = 1, PaddingMode pm = ZERO) const;
        Tensor convolution1d(const Tensor &kernel, size_t  padding = 0, size_t  stride = 1, size_t  dilation = 1, PaddingMode pm = ZERO) const;
        Tensor convolution2d(const Tensor &kernel, Tuple2d padding = 0, Tuple2d stride = 1, Tuple2d dilation = 1, PaddingMode pm = ZERO) const;
        Tensor convolution3d(const Tensor &kernel, Tuple3d padding = 0, Tuple3d stride = 1, Tuple2d dilation = 1, PaddingMode pm = ZERO) const;

        friend void reprint(std::ostream &, const Tensor &, size_t, std::vector<size_t> &);
        friend std::ostream &operator<<(std::ostream &, const Tensor &);
        
        uint64_t save(std::ofstream &file) const;
        static bool sizeMatch(const std::vector<size_t> &s1, const std::vector<size_t> &s2);
        static bool broadcastable(const std::vector<size_t>& s1, const std::vector<size_t>& s2);

        static Tensor empty_like(const Tensor& t);
        static Tensor zeros_like(const Tensor& t);
        static Tensor ones_like(const Tensor& t);
        
        friend Tensor operator-(const float64, const Tensor &);
        friend Tensor operator/(const float64, const Tensor &);
        
        template <float64 (*fn)(float64)>
        friend Tensor forEach(const Tensor &);
        friend Tensor forEach(const Tensor &, std::function<float64(float64)>);
        
        template <float64 (*fn)(float64)>
        friend Tensor &forEachInPlace(Tensor &);
        friend Tensor &forEachInPlace(Tensor &, std::function<float64(float64)>);
        
        template <void (*fn)(float64 &, float64)>
        friend Tensor op_along_axes(const Tensor &, size_t, const float64);
        
        template <void (*fn)(float64 &, float64)>
        friend float64 op_along_all_axes(const Tensor &, const float64);
        
        template <float64 (*fn)(float64, float64)>
        friend void broadcast_ew_assign(Tensor &, const Tensor &, const Tensor &, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t);
        
        template <float64 (*fn)(float64, float64)>
        friend Tensor ew_or_broadcast(const Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64)>
        friend void ew_or_broadcast_assign(Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64)>
        friend void ew_or_left_broadcast_assign(Tensor &, const Tensor &, const char *);
        
        friend Tensor stack(const Tensor &t1, const Tensor &t2, size_t dim);
        
        friend Tensor std::sqrt(const Tensor &);
        friend Tensor std::exp(const Tensor &);
        friend Tensor std::log(const Tensor &);
        friend Tensor std::pow(const Tensor &, RedFish::float64);
        friend Tensor std::pow(const Tensor &, const Tensor &);
        
        // For Debug only 
        friend bool debug(const Tensor &t1, const Tensor &t2, float64 delta);


        /* Get */
        size_t colSize() const { return this->shape.back(); }
        size_t rowSize() const { return *(this->shape.end() - 2); }
        size_t getSize() const { return size; }
        const std::vector<size_t> &getShape() const { return shape; }
        
        static std::random_device &getRandomDevice() { return rd; }

    protected:
        float64 *b;
        size_t size;
        std::vector<size_t> shape;

        // CPU 
        std::unique_ptr<float64[]> b_mem;

        // GPU 
        Buffer buffer;

        friend class DirectTensorView;
    };

    inline std::random_device Tensor::rd;
    inline std::default_random_engine Tensor::gen(rd());

    class DirectTensorView : public Tensor
    {
    public:
        DirectTensorView(const std::vector<size_t> &new_shape, float64 *ptr);
        DirectTensorView &operator=(const Tensor &t);
        DirectTensorView &operator=(const DirectTensorView &t);
        Tensor &operator=(Tensor &&t) = delete;
        void resize(const std::vector<size_t> &shape) = delete;
        Tensor &operator+=(const Tensor &t);
        Tensor &operator-=(const Tensor &t);
        Tensor &operator*=(const Tensor &t);
        Tensor &operator/=(const Tensor &t);
    };

    template <typename... Args>
    inline float64 &Tensor::operator()(Args... indices)
    {
        const size_t idx[] = {indices...};
        constexpr size_t isize = sizeof...(indices);
        size_t nsize = isize;
#ifdef CHECK_BOUNDS
        if constexpr (isize > 1)
            if (shape.size() < isize)
                throw new std::range_error("Out of bound in Tensor () operetor");

        if constexpr (isize == 0)
        {
            if (size == 0)
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 1)
        {
            if (size <= *idx)
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 2)
        {
            if (idx[0] >= *(shape.end() - 2) || idx[1] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 3)
        {
            if (idx[0] >= *(shape.end() - 3) || idx[1] >= *(shape.end() - 2) || idx[2] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 4)
        {
            if (idx[0] >= *(shape.end() - 4) || idx[1] >= *(shape.end() - 3) || idx[2] >= *(shape.end() - 2) || idx[3] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else
            for (size_t i = 0; i < isize; i++)
            {
                if (idx[i] >= *(shape.end() - isize + i - 1))
                    throw new std::range_error("Out of bound in Tensor () operetor");
            }
#endif

        if constexpr (isize == 0)
            return this->b[0];
        else if constexpr (isize == 1)
            return this->b[*idx];
        else if constexpr (isize == 2)
            return this->b[idx[0] * shape.back() + idx[1]];
        else if constexpr (isize == 3)
            return this->b[(idx[0] * *(shape.end() - 2) + idx[1]) * shape.back() + idx[2]];
        else if constexpr (isize == 4)
            return this->b[((idx[0] * *(shape.end() - 3) + idx[1]) * *(shape.end() - 2) + idx[2]) * shape.back() + idx[3]];
        else
        {
            size_t n = 0;
            for (size_t i = 0; i < this->shape.size() - 1; i++)
                n = (n + idx[i]) * this->shape[i + 1];

            return this->b[n + idx[this->shape.size() - 1]];
        }
    }


    template <typename... Args>
    inline float64 Tensor::operator()(Args... indices) const
    {
        const size_t idx[] = {indices...};
        constexpr size_t isize = sizeof...(indices);
        size_t nsize = isize;
#ifdef CHECK_BOUNDS
        if constexpr (isize > 1)
            if (shape.size() < isize)
                throw new std::range_error("Out of bound in Tensor () operetor");

        if constexpr (isize == 0)
        {
            if (size == 0)
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 1)
        {
            if (size <= *idx)
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 2)
        {
            if (idx[0] >= *(shape.end() - 2) || idx[1] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 3)
        {
            if (idx[0] >= *(shape.end() - 3) || idx[1] >= *(shape.end() - 2) || idx[2] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else if constexpr (isize == 4)
        {
            if (idx[0] >= *(shape.end() - 4) || idx[1] >= *(shape.end() - 3) || idx[2] >= *(shape.end() - 2) || idx[3] >= *(shape.end() - 1))
                throw new std::range_error("Out of bound in Tensor () operetor");
        }
        else
            for (size_t i = 0; i < isize; i++)
            {
                if (idx[i] >= *(shape.end() - isize + i - 1))
                    throw new std::range_error("Out of bound in Tensor () operetor");
            }
#endif

        if constexpr (isize == 0)
            return this->b[0];
        else if constexpr (isize == 1)
            return this->b[*idx];
        else if constexpr (isize == 2)
            return this->b[idx[0] * shape.back() + idx[1]];
        else if constexpr (isize == 3)
            return this->b[(idx[0] * *(shape.end() - 2) + idx[1]) * shape.back() + idx[2]];
        else if constexpr (isize == 4)
            return this->b[((idx[0] * *(shape.end() - 3) + idx[1]) * *(shape.end() - 2) + idx[2]) * shape.back() + idx[3]];
        else
        {
            size_t n = 0;
            for (size_t i = 0; i < this->shape.size() - 1; i++)
                n = (n + idx[i]) * this->shape[i + 1];

            return this->b[n + idx[this->shape.size() - 1]];
        }
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @tparam N
     * @param index shape: (x,...,x,d1,..,dN) -> index: (x,...,x)
     * @return DirectTensorView shape: (d1,..,dN)
     */
    template <size_t N>
    inline DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index)
    {
        static_assert(N > 0);
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end() - index.size() + i - N + 1);
            if (index[i] >= *(shape.end() - index.size() + i - N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i < N - 1; i++)
            off *= *(shape.end() + i - N + 1);

        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end() - N + i);

        return DirectTensorView({new_shape, new_shape + N}, b + off);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @tparam N
     * @param index shape: (x,...,x,d1,..,dN) -> index: (x,...,x)
     * @return DirectTensorView shape: (d1,..,dN)
     */
    template <size_t N>
    inline const DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index) const
    {
        static_assert(N > 0);
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end() - index.size() + i - N + 1);
            if (index[i] >= *(shape.end() - index.size() + i - N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i < N - 1; i++)
            off *= *(shape.end() + i - N + 1);

        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end() - N + i);

        return DirectTensorView({new_shape, new_shape + N}, b + off);
    }


    /* ---------- Functions ---------- */

    template <float64 (*fn)(float64)>
    inline Tensor forEach(const Tensor &t)
    {
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = fn(t.b[i]);
        return ret;
    }

    template <float64 (*fn)(float64)>
    inline Tensor &forEachInPlace(Tensor &t)
    {
        for (size_t i = 0; i < t.size; i++)
            t.b[i] = fn(t.b[i]);
        return t;
    }

    template <void (*fn)(float64 &, float64)>
    inline Tensor op_along_axes(const Tensor &t, size_t d, const float64 init_val)
    {
        d = t.shape.size() - d - 1;
        auto shape = t.shape;
        shape[d] = std::min((size_t)1, shape[d]);
        Tensor ret(shape);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++)
            tot *= shape[i];
        for (size_t i = d + 1; i < shape.size(); i++)
            stride *= shape[i];

        if (ret.size)
            for (size_t k = 0; k < tot; k++)
                for (size_t i = 0; i < stride; i++)
                {
                    float64 value = init_val;
                    for (size_t j = 0; j < t.shape[d]; j++)
                        fn(value, t.b[j * stride + i + k * stride * t.shape[d]]);

                    ret.b[i + k * stride] = value;
                }

        return ret;
    }

    template <void (*fn)(float64 &, float64)>
    inline float64 op_along_all_axes(const Tensor &t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
    }

    template <float64 (*fn)(float64, float64)>
    inline void broadcast_ew_assign(Tensor &dst, const Tensor &src1, const Tensor &src2,
                                    const size_t *shape, const size_t *shape1, const size_t *shape2,
                                    size_t depth,
                                    size_t off, size_t off1, size_t off2)
    {
        if (depth > 1)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                broadcast_ew_assign<fn>(dst, src1, src2, shape + 1, shape1 + 1, shape2 + 1, depth - 1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
        else
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                dst.b[off * *shape + i] = fn(src1.b[off1 * *shape1 + (i & bdc1)], src2.b[off2 * *shape2 + (i & bdc2)]);
    }

    /**
     * @brief executes fn() on two broadcastable shape tensors element wise
     *        and the result is returned with a new tensor
     *
     * @tparam fn function to be executed
     * @param t1 source and destination tensor
     * @param t2 source tensor
     * @param err_msg error message to display in case of non broadcastable shapes
     */
    template <float64 (*fn)(float64, float64)>
    inline Tensor ew_or_broadcast(const Tensor &t1, const Tensor &t2, const char *err_msg)
    {
        Tensor result;
        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            result.resize(t1.shape);
            for (size_t i = 0; i < t1.size; i++)
                result.b[i] = fn(t1.b[i], t2.b[i]);
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
            for (size_t i = 0; i < shapeT1.size(); i++)
                shapeDst[i] = shapeT1[i] == 1 ? shapeT2[i] : shapeT1[i];

            result.resize(shapeDst);
            if (result.size)
                broadcast_ew_assign<fn>(result, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
        }
        else
            throw std::length_error(err_msg);

        return result;
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
    template <float64 (*fn)(float64, float64)>
    inline void ew_or_broadcast_assign(Tensor &t1, const Tensor &t2, const char *err_msg)
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
                    broadcast_ew_assign<fn>(t1, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
            }
            else
            {
                Tensor result(shapeDst);
                if (result.size)
                    broadcast_ew_assign<fn>(result, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
                t1 = std::move(result);
            }
        }
        else
            throw std::length_error(err_msg);
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
    template <float64 (*fn)(float64, float64)>
    inline void ew_or_left_broadcast_assign(Tensor &t1, const Tensor &t2, const char *err_msg)
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
                    broadcast_ew_assign<fn>(t1, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
            }
            else
                throw std::length_error(err_msg);
        }
        else
            throw std::length_error(err_msg);
    }

    template <auto fn, typename... Args>
    inline void broadcast_op_impl(float64 *dst, const float64 *src1, const float64 *src2,
                                  const size_t *shape, const size_t *shape1, const size_t *shape2,
                                  size_t depth,
                                  size_t foff, size_t foff1, size_t foff2,
                                  size_t off, size_t off1, size_t off2,
                                  Args... args)
    {
        size_t bdc1 = (*shape1 == *shape) * ((size_t)-1);
        size_t bdc2 = (*shape2 == *shape) * ((size_t)-1);
        if (depth > 1)
            for (size_t i = 0; i < *shape; i++)
                broadcast_op_impl<fn, Args...>(
                    dst, src1, src2,
                    shape + 1, shape1 + 1, shape2 + 1,
                    depth - 1,
                    foff, foff1, foff2,
                    off * *shape + i,
                    off1 * *shape1 + (i & bdc1),
                    off2 * *shape2 + (i & bdc2),
                    args...);
        else
            for (size_t i = 0; i < *shape; i++)
                fn(dst + (off * *shape + i) * foff,
                   src1 + (off1 * *shape1 + (i & bdc1)) * foff1,
                   src2 + (off2 * *shape2 + (i & bdc2)) * foff2,
                   args...);
    }

    /**
     * @brief executes fn() on two broadcastable shape tensors "element wise"
     *
     * @param dst    buffer of the result of the operation
     * @param src1   buffer of the first tensor
     * @param src2   buffer of the second tensor
     * @param shape  shape of the resulting tensor
     * @param shape1 shape of the first tensor
     * @param shape2 shape of the first tensor
     * @param depth  recusion depth (usually the length of the shape)
     * @param foff   final offset for the destination tensor
     * @param foff1  final offset for the first tensor
     * @param foff2  final offset for the second tensor
     * @param args   additional arguments to pass to fn
     * @return void
     */
    template <auto fn, typename... Args>
    inline void broadcast_op(float64 *dst, const float64 *src1, const float64 *src2,
                             const size_t *shape, const size_t *shape1, const size_t *shape2,
                             size_t depth,
                             size_t foff, size_t foff1, size_t foff2,
                             Args... args)
    {
        broadcast_op_impl<fn, Args...>(dst, src1, src2, shape, shape1, shape2, depth, foff, foff1, foff2, (size_t)0, (size_t)0, (size_t)0, args...);
    }

    inline RedFish::Tensor stack(const RedFish::Tensor &t1, const RedFish::Tensor &t2, size_t dim)
    {
        if (t1.shape.size() <= dim)
            throw std::length_error("Tensor has not that many dimensions");

        std::vector<size_t> t1_shape = t1.shape;
        std::vector<size_t> t2_shape = t2.shape;

        int t1_1 = 0;
        for (size_t i = 0; i < (int64_t)t1.shape.size() - dim; i++)
            if (t1_shape[i] == 1)
                t1_shape.erase(t1_shape.begin()), t1_1++;
            else
                break;

        int t2_1 = 0;
        for (size_t i = 0; i < (int64_t)t2.shape.size() - dim; i++)
            if (t2_shape[i] == 1)
                t2_shape.erase(t2_shape.begin()), t2_1++;
            else
                break;

        if (t1_shape.size() != t2_shape.size())
            throw std::length_error("Tensor has not same dimmensions");

        for (size_t i = 0; i < t1_shape.size(); i++)
            if (t1_shape[i] != t2_shape[i] && i != t2.shape.size() - dim - 1)
                throw std::length_error("Tensor has not same dimmensions");

        std::vector<size_t> t3_shape;

        t1_1 = std::max(t1_1, t2_1);
        t3_shape.reserve(t1_shape.size() + t1_1);
        for (size_t i = 0; i < t1_1; i++)
            t3_shape.push_back(1);

        for (size_t i = 0; i < t1_shape.size(); i++)
            if (i == t1_shape.size() - dim - 1)
                t3_shape.push_back(t1_shape[i] + t2_shape[i]);
            else
                t3_shape.push_back(t1_shape[i]);

        Tensor t3(t3_shape);

        size_t n1 = 1;
        size_t n2 = 1;
        for (size_t i = t3_shape.size() - dim - 1; i < t3_shape.size(); i++)
        {
            n1 *= t1_shape[i];
            n2 *= t2_shape[i];
        }
        size_t n3 = n1 + n2;

        size_t p = 1;
        for (size_t i = 0; i < t3_shape.size() - dim - 1; i++)
            p *= t3_shape[i];

        for (size_t i = 0; i < p; i++)
        {
            size_t in1 = i * n1;
            size_t in2 = i * n2;
            size_t in3 = i * n3;
            for (size_t j = 0; j < n1; j++)
                t3.b[in3 + j] = t1.b[in1 + j];
            for (size_t k = 0; k < n2; k++)
                t3.b[in3 + n1 + k] = t2.b[in2 + k];
        }

        return t3;
    }

    inline bool debug(const RedFish::Tensor &t, const RedFish::Tensor &result, float64 delta)
    {
        if (!t.sizeMatch(result.getShape(), t.getShape()))
            return false;

        size_t size = result.getSize();

        for (size_t i = 0; i < size; i++)
            if (std::abs(t.b[i] - result.b[i]) > delta)
                return false;
        return true;
    }

    inline void copy_2d(float64 *src, float64 *dst, Tuple2d size, size_t stride_out, size_t stride_in)
    {
        constexpr size_t block_size = 8;
        size_t endw = size.w - size.w % block_size;
        size_t endh = size.h - size.h % block_size;

        for (size_t rb = 0; rb < endh; rb += block_size)
        {
            for (size_t cb = 0; cb < endw; cb += block_size)
                for (size_t r = rb; r < rb + block_size; r++)
                    for (size_t c = cb; c < cb + block_size; c++)
                        dst[r * stride_out + c] = src[r * stride_in + c];

            for (size_t r = rb; r < rb + block_size; r++)
                for (size_t c = endw; c < size.w; c++)
                    dst[r * stride_out + c] = src[r * stride_in + c];
        }
        for (size_t cb = 0; cb < endw; cb += block_size)
            for (size_t r = endh; r < size.h; r++)
                for (size_t c = cb; c < cb + block_size; c++)
                    dst[r * stride_out + c] = src[r * stride_in + c];

        for (size_t r = endh; r < size.h; r++)
            for (size_t c = endw; c < size.w; c++)
                dst[r * stride_out + c] = src[r * stride_in + c];
    }

} // namespace RedFish


