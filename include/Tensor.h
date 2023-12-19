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
        Tensor(const std::vector<size_t> &shape = {0}, bool onCPU = true);
        Tensor(const size_t *shape, size_t len);
        Tensor(const std::vector<size_t> &shape, float64 *buff, bool copy = true);
        Tensor(const Tensor &t);  // Copy Constructor
        Tensor(Tensor &&t);       // Move Constructor
        Tensor(const std::vector<size_t> &shape, std::initializer_list<float64> data);
        Tensor(std::ifstream& file);
        ~Tensor();

        void toDevice();
        void fromDevice();

        /* Get */
        size_t colSize() const { return this->shape.back(); }
        size_t rowSize() const { return *(this->shape.end() - 2); }
        size_t getSize() const { return size; }
        const std::vector<size_t>& getShape() const { return shape; }

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
        Tensor  operator==(const Tensor &t) const;
        Tensor  operator==(const float64 val) const;
        Tensor  operator>=(const Tensor &t) const;
        Tensor  operator>=(const float64 val) const;
        Tensor  operator<=(const Tensor &t) const;
        Tensor  operator<=(const float64 val) const;
        Tensor  operator>(const Tensor &t) const;
        Tensor  operator>(const float64 val) const;
        Tensor  operator<(const Tensor &t) const;
        Tensor  operator<(const float64 val) const;

        float64& operator()(const std::vector<size_t> &index);
        float64  operator()(const std::vector<size_t> &index) const;

        template <typename... Args>
        float64& operator()(Args... indices);
        template <typename... Args>
        float64  operator()(Args... indices) const;

        DirectTensorView getRow(const std::vector<size_t> &index);
        DirectTensorView getMatrix(const std::vector<size_t> &index);
        DirectTensorView sliceLastNDims(const std::vector<size_t> &index, size_t N);
        const DirectTensorView getRow(const std::vector<size_t> &index) const;
        const DirectTensorView getMatrix(const std::vector<size_t> &index) const;
        const DirectTensorView sliceLastNDims(const std::vector<size_t> &index, size_t N) const;
        template <size_t N>
        DirectTensorView sliceLastNDims(const std::vector<size_t> &index);
        template <size_t N>
        const DirectTensorView sliceLastNDims(const std::vector<size_t> &index) const;

        void resize(const std::vector<size_t> &new_shape);
        void reshape(const std::vector<size_t> &new_shape);

        Tensor T() const;
        Tensor T(size_t dimension1, size_t dimension2);

        void zero();
        void ones();
        void constant(float64 val);
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

        
        uint64_t save(std::ofstream &file) const;

        static Tensor empty_like(const Tensor& t);
        static Tensor zeros_like(const Tensor& t);
        static Tensor ones_like(const Tensor& t);
        
        static Tensor stack(const Tensor &t1, const Tensor &t2, size_t dim);
        
    private:
        static bool sizeMatch(const std::vector<size_t> &s1, const std::vector<size_t> &s2);
        static bool broadcastable(const std::vector<size_t>& s1, const std::vector<size_t>& s2);

        template <void (*fn)(float64 &, float64)>
        static Tensor axes_reduction(const Tensor &, size_t, const float64);
        
        template <void (*fn)(float64 &, float64)>
        static float64 full_reduction(const Tensor &, const float64);
        
        template <float64 (*fn)(float64, float64)>
        static void broadcast_ew_assign(float64*, const float64*, const float64*, const size_t *, const size_t *, const size_t *, size_t, size_t=0, size_t=0, size_t=0);

        static void broadcast_ew_device(const size_t, Buffer, Buffer, Buffer, const size_t *, const size_t *, const size_t *, size_t, size_t=0, size_t=0, size_t=0);
        
        template <float64 (*fn)(float64, float64), size_t fn_device, size_t fn_device_brdc>
        static Tensor ew_or_broadcast(const Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64), size_t fn_device>
        static void ew_or_broadcast_assign(Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64), size_t fn_device>
        static void ew_or_left_broadcast_assign(Tensor &, const Tensor &, const char *);

        friend Tensor operator-(const float64, const Tensor &);
        friend Tensor operator/(const float64, const Tensor &);
        
        friend void reprint(std::ostream &, const Tensor &, size_t, std::vector<size_t> &);
        friend std::ostream &operator<<(std::ostream &, const Tensor &);
        
        template <float64 (*fn)(float64)>
        friend Tensor forEach(const Tensor &);
        friend Tensor forEach(const Tensor &, std::function<float64(float64)>);
        
        template <float64 (*fn)(float64)>
        friend Tensor &forEachInPlace(Tensor &);
        friend Tensor &forEachInPlace(Tensor &, std::function<float64(float64)>);
        
        
        friend Tensor std::sqrt(const Tensor &);
        friend Tensor std::exp(const Tensor &);
        friend Tensor std::log(const Tensor &);
        friend Tensor std::pow(const Tensor &, RedFish::float64);
        friend Tensor std::pow(const Tensor &, const Tensor &);
        
        // For Debug only 
        friend bool debug(const Tensor &t1, const Tensor &t2, float64 delta);
        
    public:
        static std::random_device &getRandomDevice() { return rd; }

    protected:
        std::vector<size_t> shape;
        size_t size;
        bool onCPU;

        // CPU
        float64 *b;

        // GPU
        Buffer buffer;

        friend class DirectTensorView;
    };

    class DirectTensorView : public Tensor
    {
    public:
        DirectTensorView(const std::vector<size_t> &new_shape, float64 *ptr);
        ~DirectTensorView();
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

} // namespace RedFish


