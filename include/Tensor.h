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

    template<size_t N>
    struct TupleNd
    {
        TupleNd(size_t n) : { for (size_t i = 0; i < N; i++) c[i] = n; }
        TupleNd() : { for (size_t i = 0; i < N; i++) c[i] = 0; }
        TupleNd operator+(size_t n) { TuplaNd<N> tp; for (size_t i = 0; i < N; i++) tp.c[i] = c[i] + n;}
        TupleNd operator-(size_t n) { TuplaNd<N> tp; for (size_t i = 0; i < N; i++) tp.c[i] = c[i] - n;}
        size_t operator[](size_t i) { return c[i]; }
        size_t c[N];
    };

    template<>
    struct TupleNd<1>
    {
        TupleNd(size_t n) : x(n) {}
        TupleNd() : x(0) {}
        TupleNd operator+(size_t n) { return {x+n}; }
        TupleNd operator-(size_t n) { return {x-n}; }
        size_t operator[](size_t i) { return c[i]; }
        union {
            size_t c[1];
            union { size_t x, w; };
        };
    };

    template<>
    struct TupleNd<2>
    {
        TupleNd(size_t y, size_t x) : y(y), x(x) {}
        TupleNd(size_t n) : y(n), x(n) {}
        TupleNd() : y(0), x(0) {}
        TupleNd operator+(size_t n) { return {y+n, x+n}; }
        TupleNd operator-(size_t n) { return {y-n, x-n}; }
        size_t operator[](size_t i) { return c[i]; }
        union {
            size_t c[2];
            struct {
                union { size_t y, h; };
                union { size_t x, w; };
            };
        };
    };

    template<>
    struct TupleNd<3>
    {
        TupleNd(size_t z, size_t y, size_t x) : z(z), y(y), x(x) {}
        TupleNd(size_t n) : z(n), y(n), x(n) {}
        TupleNd() : y(0), x(0) {}
        TupleNd operator+(size_t n) { return {z+n, y+n, x+n}; }
        TupleNd operator-(size_t n) { return {z-n, y-n, x-n}; }
        size_t operator[](size_t i) { return c[i]; }
        union {
            size_t c[3];
            struct {
                union { size_t z, d; };
                union { size_t y, h; };
                union { size_t x, w; };
            };
        };
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
        Tensor(const std::vector<size_t> &shape, float64 *buff);
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

        void resize(const std::vector<size_t> &new_shape);
        void reshape(const std::vector<size_t> &new_shape);
        DirectTensorView asShape(const std::vector<size_t> &new_shape);
        const DirectTensorView asShape(const std::vector<size_t> &new_shape) const;
        DirectTensorView asShapeOneInsert(size_t where, size_t count = 1);
        const DirectTensorView asShapeOneInsert(size_t where, size_t count = 1) const;

        Tensor T() const;
        Tensor T(size_t dimension1, size_t dimension2);

        Tensor& zero();
        Tensor& ones();
        Tensor& constant(float64 val);
        Tensor& linspace(float64 start, float64 stop);
        Tensor& randBernulli(float64 p = 0.5);
        Tensor& randUniform(float64 a = 0.0, float64 b = 1.0);
        Tensor& randNormal(float64 mean = 0.0, float64 std = 1.0);

        float64 squareSum() const;
        Tensor  squareSum(size_t dimension, bool collapse_dimension = false) const;
        float64 max() const;
        Tensor  max(size_t dimension, bool collapse_dimension = false) const;
        float64 min() const;
        Tensor  min(size_t dimension, bool collapse_dimension = false) const;
        float64 sum() const;
        Tensor  sum(size_t dimension, bool collapse_dimension = false) const;
        Tensor  shift(size_t dimension, int direction, const float64 fill = 0.) const;
        Tensor  roundShift(size_t dimension, int direction) const;
        
        Tensor matmul(const Tensor &t, const Transpose transpose = NONE) const;
        Tensor correlation1d(const Tensor &kernel, TupleNd<1> padding = 0, TupleNd<1> stride = 1, TupleNd<1> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;
        Tensor correlation2d(const Tensor &kernel, TupleNd<2> padding = 0, TupleNd<2> stride = 1, TupleNd<2> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;
        Tensor correlation3d(const Tensor &kernel, TupleNd<3> padding = 0, TupleNd<3> stride = 1, TupleNd<3> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;
        Tensor convolution1d(const Tensor &kernel, TupleNd<1> padding = 0, TupleNd<1> stride = 1, TupleNd<1> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;
        Tensor convolution2d(const Tensor &kernel, TupleNd<2> padding = 0, TupleNd<2> stride = 1, TupleNd<2> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;
        Tensor convolution3d(const Tensor &kernel, TupleNd<3> padding = 0, TupleNd<3> stride = 1, TupleNd<3> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;

        
        uint64_t save(std::ofstream &file) const;

        static Tensor empty_like(const Tensor& t);
        static Tensor zeros_like(const Tensor& t);
        static Tensor ones_like(const Tensor& t);
        
        static Tensor stack(const Tensor &t1, const Tensor &t2, size_t dim);
        static Tensor stack(const std::vector<Tensor> &tensors, size_t dim);
        
    private:
        static bool sizeMatch(const std::vector<size_t> &s1, const std::vector<size_t> &s2);
        static bool broadcastable(const std::vector<size_t>& s1, const std::vector<size_t>& s2);

        template <void (*fn)(float64 &, float64)>
        static Tensor axes_reduction(const Tensor &, size_t, const float64, const bool);

        template <void (*fn)(float64 &, float64)>
        static Tensor axes_reduction(const Tensor &, const std::vector<size_t>&, const float64);
        
        template <void (*fn)(float64 &, float64)>
        static float64 full_reduction(const Tensor &, const float64);

        static std::vector<size_t> broadcast_shape(std::vector<size_t>&, std::vector<size_t>&);
        static void broadcast_operation(const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, const std::function<void(size_t,size_t,size_t)>&, size_t=0, size_t=0, size_t=0, size_t=0);
        
        template <float64 (*fn)(float64, float64)>
        static void broadcast_ew_assign(float64*, const float64*, const float64*, const size_t *, const size_t *, const size_t *, size_t, size_t=0, size_t=0, size_t=0);

        static void broadcast_ew_device(const size_t, Buffer, Buffer, Buffer, const size_t *, const size_t *, const size_t *, size_t, size_t=0, size_t=0, size_t=0);
        
        template <float64 (*fn)(float64, float64), size_t fn_device, size_t fn_device_brdc>
        static Tensor ew_or_broadcast(const Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64), size_t fn_device>
        static void ew_or_broadcast_assign(Tensor &, const Tensor &, const char *);
        
        template <float64 (*fn)(float64, float64), size_t fn_device>
        static void ew_or_left_broadcast_assign(Tensor &, const Tensor &, const char *);

        template <size_t N, bool conv>
        inline Tensor convcorr(const Tensor &kernel, TupleNd<N> padding = 0, TupleNd<N> stride = 1, TupleNd<N> dilation = 1, PaddingMode pm = ZERO, size_t sum_dimension = -1, bool collapse = false) const;

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
        DirectTensorView &operator=(Tensor &&t);
        void resize(const std::vector<size_t> &shape) = delete;
        DirectTensorView &operator+=(const Tensor &t);
        DirectTensorView &operator-=(const Tensor &t);
        DirectTensorView &operator*=(const Tensor &t);
        DirectTensorView &operator/=(const Tensor &t);
    };

    template <typename... Args>
    inline float64 &Tensor::operator()(Args... indices)
    {
        const size_t idx[] = {(size_t)indices...};
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
        const size_t idx[] = {(size_t)indices...};
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


