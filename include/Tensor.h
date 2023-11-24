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
#include <complex>
#include <random>

#ifdef USE_PROFILING
#include "Profiler.h"
#else
#define PROFILE
#endif

namespace RedFish
{
    class Tensor;
    typedef double float64;
}

namespace std
{
    RedFish::Tensor sqrt(const RedFish::Tensor&);
    RedFish::Tensor exp(const RedFish::Tensor&);
    RedFish::Tensor log(const RedFish::Tensor&);
    RedFish::Tensor pow(const RedFish::Tensor&, RedFish::float64);
    RedFish::Tensor pow(const RedFish::Tensor&, const RedFish::Tensor&);
}

namespace RedFish {

    typedef double float64;
    class DirectTensorView;

    enum Transpose { LEFT, RIGHT, NONE };
    enum PaddingMode { ZERO, REFLECT, REPLICATE, CIRCULAR };
    
    template <void(*fn)(float64&, float64)>\
    Tensor op_along_axes(const Tensor&, size_t, const float64);
    template <void(*fn)(float64&, float64)>
    float64 op_along_all_axes(const Tensor&, const float64);
    template <float64(*fn)(float64, float64)>
    void broadcast_ew_assign(Tensor&, const Tensor&, const Tensor&, const size_t*, const size_t*, const size_t*, size_t, size_t=0, size_t=0, size_t=0);
    template<auto fn, typename... Args>
    void broadcast_op(float64*, const float64*, const float64*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, Args...);
    template <float64(*fn)(float64, float64)>
    Tensor ew_or_broadcast(const Tensor&, const Tensor&, const char*);
    template <float64(*fn)(float64, float64)>
    void ew_or_broadcast_assign(Tensor&, const Tensor&, const char*);
    template <float64(*fn)(float64, float64)>
    void ew_or_left_broadcast_assign(Tensor&, const Tensor&, const char*);
    bool broadcastable(const std::vector<size_t>&, const std::vector<size_t>&);

    struct Tuple2d
    {
        Tuple2d(size_t y, size_t x) : y(y), x(x) {}
        Tuple2d(size_t n) : y(n), x(n) {}
        union { size_t y, h; };
        union { size_t x, w; };
    };

    struct Tuple3d
    {
        Tuple3d(size_t z, size_t y, size_t x) : z(z), y(y), x(x) {}
        Tuple3d(size_t n) : z(n), y(n), x(n) {}
        union { size_t z, d; };
        union { size_t y, h; };
        union { size_t x, w; };
    };

    class Tensor
    {
    private:
        static std::random_device rd;
        static std::default_random_engine gen;
    public:
        Tensor(const std::vector<size_t>& shape = {});
        Tensor(const size_t* shape, size_t len);
        Tensor(const std::vector<size_t>& shape, float64* buff, bool copy = true);
        Tensor(const Tensor& t);                    // Copy Constructor
        Tensor(const std::vector<size_t>& shape, std::initializer_list<float64> data);
        ~Tensor();

        Tensor& operator=(const Tensor& t);
        Tensor& operator=(Tensor&& t);
        void resize(const std::vector<size_t>& shape);
        void reshape(const std::vector<size_t>& shape);

        Tensor matmul(const Tensor& t, const Transpose transpose = NONE) const;
        Tensor T() const;
        Tensor T(size_t dimension1, size_t dimension2);
    
        // Operetors
        Tensor  operator+(const Tensor& t)  const;
        Tensor  operator+(const float64 val) const;
        Tensor& operator+=(const Tensor& t);
        Tensor& operator+=(const float64 val);
        Tensor  operator-(const Tensor& t)  const;
        Tensor  operator-(const float64 val) const;
        Tensor  operator-() const;
        Tensor& operator-=(const Tensor& t);
        Tensor& operator-=(const float64 val);
        Tensor  operator*(const Tensor& t)  const;
        Tensor  operator*(const float64 val) const;
        Tensor& operator*=(const Tensor& t);
        Tensor& operator*=(const float64 val);
        Tensor  operator/(const Tensor& t)  const;
        Tensor  operator/(const float64 val) const;
        Tensor& operator/=(const Tensor& t);
        Tensor& operator/=(const float64 val);

        Tensor crossCorrelation1d(const Tensor& kernel, size_t padding = 0, size_t stride = 1, size_t dilation = 1, PaddingMode pm = ZERO) const;
        Tensor crossCorrelation2d(const Tensor& kernel, Tuple2d padding = 0, Tuple2d stride = 1, Tuple2d dilation = 1, PaddingMode pm = ZERO) const;
        Tensor crossCorrelation3d(const Tensor& kernel, Tuple3d padding = 0, Tuple3d stride = 1, Tuple3d dilation = 1, PaddingMode pm = ZERO) const;

        float64 squareSum() const;
        Tensor  squareSum(size_t dimension) const;
        float64 max() const;
        Tensor  max(size_t dimension) const;
        float64 min() const;
        Tensor  min(size_t dimension) const;
        float64 sum() const;
        Tensor  sum(size_t dimension) const;

        float64& operator()(const std::vector<size_t>& index);
        float64  operator()(const std::vector<size_t>& index) const;

        template<typename... Args>
        float64& operator()(Args... indices);
        template<typename... Args>
        float64  operator()(Args... indices) const;

        float64& operator()(size_t);
        float64  operator()(size_t) const;

        DirectTensorView getRow(const std::vector<size_t>& index);
        const DirectTensorView getRow(const std::vector<size_t>& index) const;
        DirectTensorView getMatrix(const std::vector<size_t>& index);
        const DirectTensorView getMatrix(const std::vector<size_t>& index) const;
        template<size_t N>
        DirectTensorView sliceLastNDims(const std::vector<size_t>& index);
        template<size_t N>
        const DirectTensorView sliceLastNDims(const std::vector<size_t>& index) const;

        bool operator==(const Tensor& other) const;

        friend Tensor operator-(const float64, const Tensor&);
        friend Tensor operator/(const float64, const Tensor&);
        friend std::ostream& operator<<(std::ostream&, const Tensor&);
        friend void reprint(std::ostream&, const Tensor&, size_t, std::vector<size_t>&);
        friend Tensor empty_like(const Tensor&);
        friend Tensor zeros_like(const Tensor&);
        friend Tensor ones_like(const Tensor&);
        template<float64(*fn)(float64)>
        friend Tensor forEach(const Tensor&);
        friend Tensor forEach(const Tensor&, std::function<float64(float64)>);
        template<float64(*fn)(float64)>
        friend Tensor& forEachInPlace(Tensor&);
        friend Tensor& forEachInPlace(Tensor&, std::function<float64(float64)>);
        template <void(*fn)(float64&, float64)>
        friend Tensor op_along_axes(const Tensor&, size_t, const float64);
        template <void(*fn)(float64&, float64)>
        friend float64 op_along_all_axes(const Tensor&, const float64);
        template<float64(*fn)(float64, float64)>
        friend void broadcast_ew_assign(Tensor&, const Tensor&, const Tensor&, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t);
        template <float64(*fn)(float64, float64)>
        friend Tensor ew_or_broadcast(const Tensor&, const Tensor&, const char*);
        template <float64(*fn)(float64, float64)>
        friend void ew_or_broadcast_assign(Tensor&, const Tensor&, const char*);
        template <float64(*fn)(float64, float64)>
        friend void ew_or_left_broadcast_assign(Tensor&, const Tensor&, const char*);
        friend Tensor std::sqrt(const Tensor&);
        friend Tensor std::exp(const Tensor&);
        friend Tensor std::log(const Tensor&);
        friend Tensor std::pow(const Tensor&, RedFish::float64);
        friend Tensor std::pow(const Tensor&, const Tensor&);

        friend Tensor stack(const Tensor& t1, const Tensor& t2, size_t dim);

        // For Testing 
        friend bool debug(const Tensor &t1, const Tensor &t2, float64 delta);

        void zero();
        void ones();
        void rand();
        void rand(float64 start, float64 end);
        void randUniform(float64 a = 0.0, float64 b = 1.0);
        void randNormal(float64 mean = 0.0, float64 std = 1.0);
        void costant(float64 val);

        static std::random_device& getRandomDevice() { return rd; }
        
        static bool sizeMatch(const std::vector<size_t>& s1, const std::vector<size_t>& s2);

        size_t colSize() const { return this->shape.back(); }
        size_t rowSize() const { return *(this->shape.end()-2); }
        size_t getSize() const { return size; }
        std::vector<size_t> getShape() const { return shape; }
        void setShape(std::vector<size_t> new_shape) { shape = new_shape; }

    protected:
        std::unique_ptr<float64[]> b_mem;
        float64* b;
        size_t size;
        std::vector<size_t> shape;
    };

    inline std::random_device Tensor::rd;
    inline std::default_random_engine Tensor::gen(rd());

    class DirectTensorView : public Tensor
    {
    public:
        DirectTensorView(const std::vector<size_t>& new_shape, float64* ptr)
            : Tensor({0})
        {
            shape = new_shape;
            size = 1;
            for (size_t i = 0; i < shape.size(); i++)
                size *= shape[i];
            b = ptr;
        }
        Tensor& operator=(const Tensor& t) = delete;
        Tensor& operator=(Tensor&& t) = delete;
        void resize(const std::vector<size_t>& shape) = delete;
        Tensor& operator+=(const Tensor& t)
        {
            constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
            ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in sum operation");
            return *this;
        }
        Tensor& operator-=(const Tensor& t)
        {
            constexpr auto fn = [](float64 n1, float64 n2) {return n1-n2;};
            ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in subtruction operation");
            return *this;
        }
        Tensor& operator*=(const Tensor& t)
        {
            constexpr auto fn = [](float64 n1, float64 n2) {return n1*n2;};
            ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
            return *this;
        }
        Tensor& operator/=(const Tensor& t)
        {
            constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
            ew_or_left_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in division operation");
            return *this;
        }

    };


    /* -------- Constructors -------- */

    inline Tensor::Tensor(const std::vector<size_t>& shape)
        :shape(shape)
    {
        PROFILE
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (size)
            b = (b_mem = std::make_unique<float64[]>(size)).get();
    }

    inline Tensor::Tensor(const size_t* shape, size_t len)
        :shape(shape, shape + len)
    {
        PROFILE
        size = 1;
        for (size_t i = 0; i < len; i++)
            size *= shape[i];

        if (size)
            b = (b_mem = std::make_unique<float64[]>(size)).get();
    }

    inline Tensor::Tensor(const std::vector<size_t>& shape, float64 *buff, bool copy)
        :shape(shape)
    {
        PROFILE
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (copy)
        {
            if (size)
            {
                b_mem = std::make_unique<float64[]>(size);
                std::copy(buff, buff + size, b_mem.get());
            }
        }
        else
            b_mem.reset(buff);

        b = b_mem.get();
    }

    inline Tensor::Tensor(const Tensor& t)
    {
        PROFILE
        this->shape = t.shape;
        this->size = t.size;
        if (size) this->b_mem = std::make_unique<float64[]>(size);
        b = b_mem.get();

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }

    inline Tensor::Tensor(const std::vector<size_t>& shape, std::initializer_list<float64> data)
        :shape(shape)
    {
        if(shape.size() != 0)
        {
            size = 1;
            for (size_t i = 0; i < shape.size(); i++)
                size *= shape[i];
        }
        else 
        {
            this->shape.push_back(data.size());
            size = data.size();
        }

        if (size != data.size())
            throw std::length_error("Invalid number of data given to Tensor for this shape");

        if (size)
            b = (b_mem = std::make_unique<float64[]>(size)).get();

        for (size_t i = 0; i < size; i++)
            this->b[i] = data.begin()[i];        
    }

    inline Tensor& Tensor::operator=(const Tensor& t)
    {
        PROFILE
        this->shape = t.shape;
        this->size = t.size;
        if (size) this->b_mem = std::make_unique<float64[]>(size);
        else this->b_mem = nullptr;
        b = b_mem.get();

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator=(Tensor &&t)
    {
        PROFILE
        this->shape = t.shape;
        this->size = t.size;
        this->b_mem = std::move(t.b_mem);
        b = b_mem.get();
        t.shape = {0};
        t.size = 0;

        return *this;
    }

    inline Tensor::~Tensor() { }

    inline void Tensor::resize(const std::vector<size_t>& shape)
    {
        PROFILE
        this->shape = shape;
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        b_mem = std::make_unique<float64[]>(size);
        b = b_mem.get();
    }

    inline void Tensor::reshape(const std::vector<size_t> &shape)
    {
        PROFILE
        size_t new_size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            new_size *= shape[i];

        if (new_size != size)
            throw std::length_error("Invalid new shape in Tensor reshape");

        this->shape = shape;
    }

    inline void matmul_gotoblas(float64* dst, const float64* m1, const float64* m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        const size_t block_size = 8;
        size_t j_end = rows - rows % block_size;
        size_t i_end = cols - cols % block_size;
        size_t k_end = mid  - mid  % block_size;

        #pragma omp parallel for
        for (size_t jc = 0; jc < j_end; jc += block_size)
        {
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];
            }
                
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                    for (size_t k  = k_end; k < mid; k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k  = k_end; k  < mid;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];
        }
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];

                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];
            }
                
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                    for (size_t k  = k_end; k < mid; k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];

                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k  = k_end; k  < mid;  k++)
                        dst[jr*ld0 + ir] += m1[jr*ld1 + k] * m2[ir + k*ld2];
    }

    inline void matmul_left_T(float64* dst, const float64* m1, const float64* m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        const size_t block_size = 256;
        size_t j_end = rows - rows % block_size;
        size_t i_end = cols - cols % block_size;
        size_t k_end = mid  - mid  % block_size;

        #pragma omp parallel for
        for (size_t jc = 0; jc < j_end; jc += block_size)
        {
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];
            }
                
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t k  = k_end; k < mid; k++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t k  = k_end; k  < mid;  k++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];
        }
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];

                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t k  = kc; k  < kc + block_size;  k++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];
            }
                
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t k  = k_end; k < mid; k++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];

                    for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t k  = k_end; k  < mid;  k++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        dst[jr*ld0 + ir] += m1[jr + k*ld1] * m2[ir + k*ld2];
    }
    
    inline Tensor Tensor::matmul(const Tensor &t, const Transpose trsp) const
    {
        PROFILE
        Tensor result;
        std::vector<size_t> shapeT1, matShapeT1(  shape.begin() + std::max<int64_t>(0, (int64_t)  shape.size() - 2),   shape.end());
        std::vector<size_t> shapeT2, matShapeT2(t.shape.begin() + std::max<int64_t>(0, (int64_t)t.shape.size() - 2), t.shape.end());
        size_t size1 = 1, size2 = 1;

        for (size_t i = 0; i < (int64_t)  shape.size()-2; i++) shapeT1.push_back(  shape[i]), size1 *=   shape[i];
        for (size_t i = 0; i < (int64_t)t.shape.size()-2; i++) shapeT2.push_back(t.shape[i]), size2 *= t.shape[i];
        matShapeT1.insert(matShapeT1.begin(), std::max<int64_t>(0, (int64_t)2 -   shape.size()), 1);
        matShapeT2.insert(matShapeT2.begin(), std::max<int64_t>(0, (int64_t)2 - t.shape.size()), 1);
        if (t.shape.size() == 1) std::swap(matShapeT2[0], matShapeT2[1]);
        if (trsp == LEFT)  std::swap(matShapeT1[0], matShapeT1[1]); else
        if (trsp == RIGHT) std::swap(matShapeT2[0], matShapeT2[1]);

        if (shapeT1.size() > shapeT2.size())
            shapeT2.insert(shapeT2.begin(), shapeT1.size() - shapeT2.size(), 1);
        else
        if (shapeT1.size() < shapeT2.size())
            shapeT1.insert(shapeT1.begin(), shapeT2.size() - shapeT1.size(), 1);

        if (matShapeT1[1] != matShapeT2[0])
            throw std::length_error("Matrix size not matching in Tensor matmul");

        size_t rows = matShapeT1[0];
        size_t mid  = matShapeT1[1];
        size_t cols = matShapeT2[1];
        size_t matsize0 = rows * cols;
        size_t matsize1 = rows * mid;
        size_t matsize2 = mid  * cols;

        if (sizeMatch(shapeT1, shapeT2))
        {
            shapeT1.push_back(rows);
            shapeT1.push_back(cols);
            result.resize(shapeT1);
            result.zero();
            switch (trsp)
            {
            case LEFT:
                for (size_t i = 0; i < size1; i++)
                    matmul_left_T(result.b + i*matsize0, b + i*matsize1, t.b + i*matsize2, rows, mid, cols, cols, rows, cols);
                break;
            case RIGHT:
            {
                auto tt = t.T();
                for (size_t i = 0; i < size1; i++)
                    matmul_gotoblas(result.b + i*matsize0, b + i*matsize1, tt.b + i*matsize2, rows, mid, cols, cols, mid, mid);
                break;
            }
            case NONE:
            default:
                for (size_t i = 0; i < size1; i++)
                    matmul_gotoblas(result.b + i*matsize0, b + i*matsize1, t.b + i*matsize2, rows, mid, cols, cols, mid, cols);
                break;
            }
        }
        else if(broadcastable(shapeT1, shapeT2))
        {
            std::vector<size_t> shapeDst(shapeT1.size());
            for (size_t i = 0; i < shapeT1.size(); i++)
                shapeDst[i] = shapeT1[i] == 1 ? shapeT2[i] : shapeT1[i];

            shapeDst.push_back(rows);
            shapeDst.push_back(cols);
                
            result.resize(shapeDst);
            result.zero();
            if (result.size)
                switch (trsp)
                {
                case LEFT:
                    broadcast_op<matmul_left_T>(result.b, this->b, t.b,
                                                shapeDst.data(), shapeT1.data(), shapeT2.data(),
                                                shapeDst.size()-2,
                                                rows*cols, rows*mid, mid*cols,
                                                rows, mid, cols, cols, rows, cols);
                    break;
                case RIGHT:
                    broadcast_op<matmul_gotoblas>(result.b, this->b, t.T().b,
                                                  shapeDst.data(), shapeT1.data(), shapeT2.data(),
                                                  shapeDst.size()-2,
                                                  rows*cols, rows*mid, mid*cols,
                                                  rows, mid, cols, cols, mid, cols);
                    break;
                case NONE:
                default:
                    broadcast_op<matmul_gotoblas>(result.b, this->b, t.b,
                                                  shapeDst.data(), shapeT1.data(), shapeT2.data(),
                                                  shapeDst.size()-2,
                                                  rows*cols, rows*mid, mid*cols,
                                                  rows, mid, cols, cols, mid, cols);
                    break;
                }
        }
        else
            throw std::length_error("Shapes not matching in Tensor matmul");

        return result;
    }

    inline Tensor matmul(const Tensor& t1, const Tensor& t2, const Transpose transpose = NONE)
    {
        return t1.matmul(t2, transpose);
    }

    inline Tensor Tensor::T() const
    {
        PROFILE
        Tensor t(shape);

        if (shape.size() < 1) t.b[0] = this->b[0];
        else 
        {
            size_t end = 1;
            size_t cols = t.shape.back();
            size_t rows = shape.size() < 2 ? 1 : t.shape[t.shape.size()-2];
            std::swap(t.shape.back(), t.shape[t.shape.size()-2]);

            for (size_t i = 0; i < (int64_t)t.shape.size() - 2; i++)
                end *= t.shape[i];

            const size_t block_size = 8;
            const size_t k_end = rows - rows % block_size;
            const size_t j_end = cols - cols % block_size;

            for (size_t i = 0, stride = rows*cols; i < end; i++)
            {
                float64* tp = t.b + i * stride, *thisp = this->b + i * stride;

                for (size_t k = 0; k < k_end; k += block_size)
                {
                    for (size_t j = 0; j < j_end; j += block_size)
                        for (size_t r = k; r < k + block_size; r++)
                        for (size_t c = j; c < j + block_size; c++)
                            tp[c*rows + r] = thisp[r*cols + c];

                        for (size_t r = k; r < k + block_size; r++)
                        for (size_t c = j_end; c < cols; c++)
                            tp[c*rows + r] = thisp[r*cols + c];
                }
                    for (size_t j = 0; j < j_end; j += block_size)
                        for (size_t r = k_end; r < rows; r++)
                        for (size_t c = j; c < j + block_size; c++)
                            tp[c*rows + r] = thisp[r*cols + c];

                        for (size_t r = k_end; r < rows; r++)
                        for (size_t c = j_end; c < cols; c++)
                            tp[c*rows + r] = thisp[r*cols + c];
            }
        }

        return t;
    }

    inline Tensor Tensor::T(size_t d1, size_t d2)
    {
        if (d1 == d2 || d1 >= shape.size() || d2 >= shape.size())
            throw std::range_error("invalid dimensions in Tensor transposition T()");

        if (d1 < d2) std::swap(d1, d2);

        Tensor t(shape);
        d1 = t.shape.size()-1 - d1, d2 = t.shape.size()-1 - d2;
        size_t rows = t.shape[d1], cols = t.shape[d2], end = 1, stride = 1, step = 1;
        std::swap(t.shape[d1], t.shape[d2]);

        for (size_t i = 0; i < d1; i++) end *= t.shape[i];
        for (size_t i = d1+1; i < d2; i++) step *= t.shape[i];
        for (size_t i = d2+1; i < t.shape.size(); i++) stride *= t.shape[i];

        // To-Do

        return t;
    }

    /* Operetors */
    inline Tensor Tensor::operator+(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in sum operation");
    }

    inline Tensor Tensor::operator+(const float64 val) const
    {
        PROFILE
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] + val;

        return result;
    }

    inline Tensor operator+(const float64 val, const Tensor& t)
    {
        return t + val;
    }

    inline Tensor& Tensor::operator+=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    inline Tensor& Tensor::operator+=(const float64 val)
    {
        PROFILE
        for (size_t i = 0; i < size; i++)
            this->b[i] += val;

        return *this;
    }

    inline Tensor Tensor::operator-(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1-n2;};
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in subtraction operation");
    }

    inline Tensor Tensor::operator-(const float64 val) const
    {
        PROFILE
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - val;

        return result;
    }

    inline Tensor Tensor::operator-() const
    {
        PROFILE
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = -this->b[i];

        return result;
    }

    inline Tensor operator-(const float64 val, const Tensor& t)
    {
        PROFILE
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val - t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator-=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1-n2;};
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    inline Tensor& Tensor::operator-=(const float64 val)
    {
        PROFILE
        for (size_t i = 0; i < size; i++)
            this->b[i] -= val;

        return *this;
    }

    inline Tensor Tensor::operator*(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1*n2;};
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
    }

    inline Tensor Tensor::operator*(const float64 val) const
    {
        PROFILE
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] * val;

        return result;
    }

    inline Tensor operator*(const float64 val, const Tensor& t)
    {
        return t * val;
    }

    inline Tensor& Tensor::operator*=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1*n2;};
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    inline Tensor& Tensor::operator*=(const float64 val)
    {
        PROFILE
        for (size_t i = 0; i < size; i++)
            this->b[i] *= val;

        return *this;
    }

    inline Tensor Tensor::operator/(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1/n2;};
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in division operation");
    }

    inline Tensor Tensor::operator/(const float64 val) const
    {
        PROFILE
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / val;

        return result;
    }

    inline Tensor operator/(const float64 val, const Tensor& t)
    {
        PROFILE
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val / t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator/=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }

    inline Tensor& Tensor::operator/=(const float64 val)
    {
        PROFILE
        for (size_t i = 0; i < size; i++)
            this->b[i] /= val;

        return *this;
    }

    inline uint8_t reverse(uint8_t b) {
        b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
        b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
        b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
        return b;
    }

    inline size_t reverse_bits(size_t i)
    {
        size_t ret;
        uint8_t* bt = (uint8_t*)&i;
        uint8_t* btr = (uint8_t*)&ret;
        if constexpr (sizeof(size_t)>0) btr[sizeof(size_t)-1] = reverse(bt[0]);
        if constexpr (sizeof(size_t)>1) btr[sizeof(size_t)-2] = reverse(bt[1]);
        if constexpr (sizeof(size_t)>2) btr[sizeof(size_t)-3] = reverse(bt[2]);
        if constexpr (sizeof(size_t)>3) btr[sizeof(size_t)-4] = reverse(bt[3]);
        if constexpr (sizeof(size_t)>4) btr[sizeof(size_t)-5] = reverse(bt[4]);
        if constexpr (sizeof(size_t)>5) btr[sizeof(size_t)-6] = reverse(bt[5]);
        if constexpr (sizeof(size_t)>6) btr[sizeof(size_t)-7] = reverse(bt[6]);
        if constexpr (sizeof(size_t)>7) btr[sizeof(size_t)-8] = reverse(bt[7]);
        if constexpr (sizeof(size_t)>8) btr[sizeof(size_t)-9] = reverse(bt[8]);
        if constexpr (sizeof(size_t)>9) btr[sizeof(size_t)-10] = reverse(bt[9]);
        return ret;
    }

    inline void fft_impl(std::complex<float64>* dst, const float64* src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2*pi;
        size_t shift = sizeof(size_t)*8 - std::log2(n);
        for (size_t k = 0; k < n; k++)
            dst[reverse_bits(k) >> shift] = src[k];

        for (size_t m = 2; m <= n; m *= 2)
        {
            using namespace std;
            complex<float64> wm = exp(-pi2/m*1i);
            for (size_t k = 0; k < n; k += m)
            {
                complex<float64> w = 1.;
                for (size_t j = 0; j < m/2; j++)
                {
                    auto t = dst[k+j+m/2]*w;
                    auto u = dst[k+j];
                    dst[k+j] = u+t;
                    dst[k+j + m/2] = u-t;
                    w *= wm;
                }
            }
        }
    }

    inline void fft_impl_reversed(std::complex<float64>* dst, const float64* src, size_t n, size_t stride = 1, size_t stride_in = 1)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2*pi;
        for (size_t k = 0; k < n; k++)
            dst[k*stride] = src[k*stride_in];

        for (size_t m = n; m > 1; m /= 2)
        {
            using namespace std;
            complex<float64> wm = exp(-pi2/m*1i);
            for (size_t k = 0; k < n; k += m)
            {
                complex<float64> w = 1.;
                for (size_t j = 0; j < m/2; j++)
                {
                    auto t = dst[(k+j+m/2)*stride];
                    auto u = dst[(k+j)*stride];
                    dst[(k+j)*stride] = u+t;
                    dst[(k+j + m/2)*stride] = (u-t)*w;
                    w *= wm;
                }
            }
        }
    }

    inline void fft2d_impl_reversed(std::complex<float64>* dst, const float64* src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2*pi;
        for (size_t k = 0; k < n*n; k++)
            dst[k] = src[k];

        for (size_t m = n; m > 1; m /= 2)
        {
            using namespace std;
            complex<float64> wm = exp(-pi2/m*1i);
            for (size_t l = 0; l < n; l += m)
            for (size_t k = 0; k < n; k += m)
            {
                complex<float64> wr = 1.;
                for (size_t i = 0; i < m/2; i++)
                {
                    complex<float64> wc = 1.;
                    for (size_t j = 0; j < m/2; j++)
                    {
                        auto s00 = dst[(l+i)*n + k+j];
                        auto s01 = dst[(l+i)*n + k+j+m/2];
                        auto s10 = dst[(l+i+m/2)*n + k+j];
                        auto s11 = dst[(l+i+m/2)*n + k+j+m/2];
                        dst[(l+i)*n + k+j] = s00 + s01 + s10 + s11;
                        dst[(l+i)*n + k+j+m/2] = (s00 - s01 + s10 - s11)*wc;
                        dst[(l+i+m/2)*n + k+j] = (s00 + s01 - s10 - s11)*wr;
                        dst[(l+i+m/2)*n + k+j+m/2] = (s00 - s01 - s10 + s11)*wc*wr;
                        wc *= wm;
                    }
                    wr *= wm;
                }
            }
        }
    }
    
    inline void cross_correlation_1d_impl(float64* dst, const float64* t, const float64* kernel, size_t t_size, size_t kernel_size, size_t padding, size_t stride, size_t dilation)
    {
        constexpr size_t block_size = 32;
        size_t end = (t_size - (kernel_size-1)*dilation + stride - 1) / stride;
        size_t end_b = end - end % block_size;

        for (size_t cb = 0; cb < end_b; cb += block_size)
        for (size_t ck = 0; ck < kernel_size; ck++)
        for (size_t c = cb; c < cb + block_size; c++)
                dst[c] += t[c*stride + ck*dilation] * kernel[ck];

        for (size_t ck = 0; ck < kernel_size; ck++)
        for (size_t c = end_b; c < end; c++)
                dst[c] += t[c*stride + ck*dilation] * kernel[ck];
    }

    inline void cross_correlation_2d_impl(float64* dst, const float64* t, const float64* kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation)
    {
        constexpr size_t block_size_r = 4;
        constexpr size_t block_size_c = 32;
        Tuple2d end = {(t_size.y - (kernel_size.y-1)*dilation.y + stride.y - 1) / stride.y,
                       (t_size.x - (kernel_size.x-1)*dilation.x + stride.x - 1) / stride.x};
        Tuple2d end_b = {end.y - end.y % block_size_r, end.x - end.x % block_size_c};

        const auto conv1d = [=](float64* dst, const float64* t, const float64* kernel) {
            for (size_t cb = 0; cb < end_b.x; cb += block_size_c)
            for (size_t ck = 0; ck < kernel_size.x; ck++)
            for (size_t c = cb; c < cb + block_size_c; c++)
                    dst[c] += t[c*stride.w + ck*dilation.w] * kernel[ck];

            for (size_t ck = 0; ck < kernel_size.x; ck++)
            for (size_t c = end_b.x; c < end.x; c++)
                    dst[c] += t[c*stride.w + ck*dilation.w] * kernel[ck];
        };

        #pragma omp parallel for
        for (size_t rb = 0; rb < end_b.y; rb += block_size_r)
        for (size_t rk = 0; rk < kernel_size.y; rk++)
        for (size_t r = rb; r < rb + block_size_r; r++)
            conv1d(dst + r*end.w, t + (r*stride.h + rk*dilation.h)*t_size.w, kernel + rk*kernel_size.w);

        for (size_t rk = 0; rk < kernel_size.y; rk++)
        for (size_t r = end_b.y; r < end.y; r++)
            conv1d(dst + r*end.w, t + (r*stride.h + rk*dilation.h)*t_size.w, kernel + rk*kernel_size.w);
    }

    inline void cross_correlation_3d_impl(float64* dst, const float64* t, const float64* kernel, Tuple3d t_size, Tuple3d kernel_size, Tuple3d stride, Tuple3d dilation)
    {
        constexpr size_t block_size_r = 4;
        constexpr size_t block_size_c = 32;
        Tuple3d end = {0, (t_size.y - (kernel_size.y-1)*dilation.y + stride.y - 1) / stride.y,
                          (t_size.x - (kernel_size.x-1)*dilation.x + stride.x - 1) / stride.x};
        Tuple3d end_b = {0, end.y - end.y % block_size_r, end.x - end.x % block_size_c};

        const auto conv1d = [=](float64* dst, const float64* t, const float64* kernel) {
            for (size_t cb = 0; cb < end_b.x; cb += block_size_c)
            for (size_t ck = 0; ck < kernel_size.x; ck++)
            for (size_t c = cb; c < cb + block_size_c; c++)
                    dst[c] += t[c*stride.w + ck*dilation.w] * kernel[ck];

            for (size_t ck = 0; ck < kernel_size.x; ck++)
            for (size_t c = end_b.x; c < end.x; c++)
                    dst[c] += t[c*stride.w + ck*dilation.w] * kernel[ck];
        };

        const auto conv2d = [=](float64* dst, const float64* t, const float64* kernel) {
            for (size_t rb = 0; rb < end_b.y; rb += block_size_r)
            for (size_t rk = 0; rk < kernel_size.y; rk++)
            for (size_t r = rb; r < rb + block_size_r; r++)
                conv1d(dst + r*end.w, t + (r*stride.h + rk*dilation.h)*t_size.w, kernel + rk*kernel_size.w);

            for (size_t rk = 0; rk < kernel_size.y; rk++)
            for (size_t r = end_b.y; r < end.y; r++)
                conv1d(dst + r*end.w, t + (r*stride.h + rk*dilation.h)*t_size.w, kernel + rk*kernel_size.w);
        };

        for (size_t dk = 0; dk < kernel_size.z; dk++)
        for (size_t d = 0; d < end.z; d++)
            conv2d(dst + d*end.h*end.w, t + (d*stride.d + dk*dilation.d)*t_size.h*t_size.w, kernel + dk*kernel_size.h*kernel_size.w);
    }

    inline Tensor Tensor::crossCorrelation1d(const Tensor& kernel, size_t padding, size_t stride, size_t dilation, PaddingMode pm) const
    {
        PROFILE
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 1) shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 1) shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 1 < shape_k.size(); i++)
            if (shape_k[i] != 1) 
                throw std::length_error("Invalid kernel shape in crossCorrelation1d");
        if (shape_k.back() == 0) 
            throw std::length_error("Invalid kernel shape in crossCorrelation1d");
        if (stride == 0) 
            throw std::length_error("Invalid stride in crossCorrelation1d");

        for (size_t i = 0; i + 1 < shape_t.size(); i++) size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2*padding + shape_t.back() + stride >= (shape_k.back()-1)*dilation + 1)
             shape_r.back() = (2*padding + shape_t.back() - (shape_k.back()-1)*dilation - 1) / stride + 1;
        else shape_r.back() = 0, size_batch = 0;
        size_t off_t = shape_t.back();
        size_t off_r = shape_r.back();

        Tensor result(shape_r);
        result.zero();

        #pragma omp parallel for
        for (size_t i = 0; i < size_batch; i++)
        {
            cross_correlation_1d_impl(
                    result.b + i*off_r,
                    this-> b + i*off_t,
                    kernel.b,
                    shape_t.back(),
                    shape_k.back(),
                    padding, stride, dilation);
        }
        
        return result;
    }

    inline Tensor Tensor::crossCorrelation2d(const Tensor& kernel, Tuple2d padding, Tuple2d stride, Tuple2d dilation, PaddingMode pm) const
    {
        PROFILE
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 2) shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 2) shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 2 < shape_k.size(); i++)
            if (shape_k[i] != 1) 
                throw std::length_error("Invalid kernel shape in crossCorrelation2d");
        if (shape_k.back() == 0 || shape_k.end()[-2] == 0) 
            throw std::length_error("Invalid kernel shape in crossCorrelation2d");
        if (stride.x == 0 || stride.y == 0) 
            throw std::length_error("Invalid stride in crossCorrelation2d");

        for (size_t i = 0; i + 2 < shape_t.size(); i++) size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2*padding.x + shape_t.back() + stride.x >= (shape_k.back()-1)*dilation.x + 1) shape_r.back() = (2*padding.x + shape_t.back() - (shape_k.back()-1)*dilation.x + stride.x - 1) / stride.x;
        else shape_r.back() = 0, size_batch = 0;
        if (2*padding.y + shape_t.end()[-2] + stride.y >= (shape_k.end()[-2]-1)*dilation.y + 1) shape_r.end()[-2] = (2*padding.y + shape_t.end()[-2] - (shape_k.end()[-2]-1)*dilation.y + stride.y - 1) / stride.y;
        else shape_r.end()[-2] = 0, size_batch = 0;
        size_t off_t = shape_t.end()[-2]*shape_t.back();
        size_t off_r = shape_r.end()[-2]*shape_r.back();
        size_t ph = shape_t.end()[-2]+2*padding.h, pw = shape_t.back()+2*padding.w;
        size_t off_p = ph*pw;

        Tensor result(shape_r);
        result.zero();

        std::unique_ptr<float64[]> padded = std::make_unique<float64[]>(off_p);
        //#pragma omp parallel for
        for (size_t i = 0; i < size_batch; i++)
        {
            const size_t block_size = 8;
            for (size_t rb = 0; rb < shape_t.end()[-2]; rb += block_size)
            for (size_t cb = 0; cb < shape_t.back(); cb += block_size)
                for (size_t r = rb; rb + r < block_size; r++)
                for (size_t c = 0; cb + c < block_size; c++)
                    padded[(r + padding.h)*pw + c + padding.w] = b[r*shape_t.back() + c];
                
            
            cross_correlation_2d_impl(
                    result.b + i*off_r,
                    padded.get() + i*off_p,
                    kernel.b,
                    {ph, pw},
                    {shape_k.end()[-2], shape_k.back()},
                    stride, dilation);
        }
        
        return result;
    }

    inline Tensor Tensor::crossCorrelation3d(const Tensor& kernel, Tuple3d padding, Tuple3d stride, Tuple3d dilation, PaddingMode pm) const
    {
        PROFILE
        return Tensor();
    }

    inline float64 Tensor::squareSum() const
    {
        constexpr auto fn = [](float64& sq_sum, float64 n) {sq_sum += n*n;};
        return op_along_all_axes<fn>(*this, 0.);
    }

    inline Tensor Tensor::squareSum(size_t d) const
    {
        constexpr auto fn = [](float64& sq_sum, float64 n) {sq_sum += n*n;};
        return op_along_axes<fn>(*this, d, 0.);
    }

    inline float64 Tensor::max() const
    {
        constexpr auto fn = [](float64& max, float64 n) {if (max < n) max = n;};
        return op_along_all_axes<fn>(*this, -std::numeric_limits<float64>::infinity());
    }

    inline Tensor Tensor::max(size_t d) const
    {
        constexpr auto fn = [](float64& max, float64 n) {if (max < n) max = n;};
        return op_along_axes<fn>(*this, d, -std::numeric_limits<float64>::infinity());
    }

    inline float64 Tensor::min() const
    {
        constexpr auto fn = [](float64& min, float64 n) {if (min > n) min = n;};
        return op_along_all_axes<fn>(*this, std::numeric_limits<float64>::infinity());
    }

    inline Tensor Tensor::min(size_t d) const
    {
        constexpr auto fn = [](float64& min, float64 n) {if (min > n) min = n;};
        return op_along_axes<fn>(*this, d, std::numeric_limits<float64>::infinity());
    }

    inline float64 Tensor::sum() const
    {
        constexpr auto fn = [](float64& sum, float64 n) {sum += n;};
        return op_along_all_axes<fn>(*this, 0.);
    }

    inline Tensor Tensor::sum(size_t d) const
    {
        constexpr auto fn = [](float64& sum, float64 n) {sum += n;};
        return op_along_axes<fn>(*this, d, 0.);
    }

    inline float64& Tensor::operator()(const std::vector<size_t>& index)
    {
        #ifdef CHECK_BOUNDS 
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (index[i] < 0 || index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tensor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (size_t i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    inline float64 Tensor::operator()(const std::vector<size_t>& index)  const 
    {
        #ifdef CHECK_BOUNDS 
            for (size_t i = 0; i < shape.size(); i++)
                if (index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tensor () operetor");
        #endif
        
        size_t n = 0;
        for (size_t i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    template<typename... Args>
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
            else if constexpr (isize == 1) { if (size <= *idx) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 2) { if (idx[0] >= *(shape.end()-2) || idx[1] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 3) { if (idx[0] >= *(shape.end()-3) || idx[1] >= *(shape.end()-2) || idx[2] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 4) { if (idx[0] >= *(shape.end()-4) || idx[1] >= *(shape.end()-3) || idx[2] >= *(shape.end()-2) || idx[3] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else
                for (size_t i = 0; i < isize; i++)
                {
                    if (idx[i] >= *(shape.end()-isize+i-1))
                        throw new std::range_error("Out of bound in Tensor () operetor");
                }
        #endif

        if constexpr (isize == 0) return this->b[0];
        else if constexpr (isize == 1) return this->b[*idx];
        else if constexpr (isize == 2) return this->b[idx[0]*shape.back() + idx[1]];
        else if constexpr (isize == 3) return this->b[(idx[0] * *(shape.end()-2) + idx[1])*shape.back() + idx[2]];
        else if constexpr (isize == 4) return this->b[((idx[0] * *(shape.end()-3) + idx[1])* *(shape.end()-2) + idx[2])*shape.back() + idx[3]];
        else
        {
            size_t n = 0;
            for (size_t i = 0; i < this->shape.size() - 1; i++)
                n = (n + idx[i]) * this->shape[i + 1];
                
            return this->b[n + idx[this->shape.size() - 1]];
        }
    }

    template<typename... Args>
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
            else if constexpr (isize == 1) { if (size <= *idx) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 2) { if (idx[0] >= *(shape.end()-2) || idx[1] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 3) { if (idx[0] >= *(shape.end()-3) || idx[1] >= *(shape.end()-2) || idx[2] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else if constexpr (isize == 4) { if (idx[0] >= *(shape.end()-4) || idx[1] >= *(shape.end()-3) || idx[2] >= *(shape.end()-2) || idx[3] >= *(shape.end()-1)) throw new std::range_error("Out of bound in Tensor () operetor"); }
            else
                for (size_t i = 0; i < isize; i++)
                {
                    if (idx[i] >= *(shape.end()-isize+i-1))
                        throw new std::range_error("Out of bound in Tensor () operetor");
                }
        #endif

        if constexpr (isize == 0) return this->b[0];
        else if constexpr (isize == 1) return this->b[*idx];
        else if constexpr (isize == 2) return this->b[idx[0]*shape.back() + idx[1]];
        else if constexpr (isize == 3) return this->b[(idx[0] * *(shape.end()-2) + idx[1])*shape.back() + idx[2]];
        else if constexpr (isize == 4) return this->b[((idx[0] * *(shape.end()-3) + idx[1])* *(shape.end()-2) + idx[2])*shape.back() + idx[3]];
        else
        {
            size_t n = 0;
            for (size_t i = 0; i < this->shape.size() - 1; i++)
                n = (n + idx[i]) * this->shape[i + 1];
                
            return this->b[n + idx[this->shape.size() - 1]];
        }
    }


    inline float64& Tensor::operator()(size_t index)
    {
        #ifdef CHECK_BOUNDS
            if(index > size)
                throw new std::range_error("Out of bound in Tensor () operetor");
        #endif 
        return this->b[index];
    }

    inline float64 Tensor::operator()(size_t index) const
    {
        #ifdef CHECK_BOUNDS
            if(index > size)
                throw new std::range_error("Out of bound in Tensor () operetor");
        #endif 
        return this->b[index];
    }

    template <size_t N>
    inline DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index)
    {
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end()-index.size()+i-N+1);
            if (index[i] >= *(shape.end()-index.size()+i-N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end()-N);

        return DirectTensorView({new_shape, new_shape+N}, b + off);
    }

    template <size_t N>
    inline const DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index) const
    {
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end()-index.size()+i-N+1);
            if (index[i] >= *(shape.end()-index.size()+i-N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end()-N);

        return DirectTensorView({new_shape, new_shape+N}, b + off);
    }

    inline       DirectTensorView Tensor::getRow(const std::vector<size_t> &index)          { return sliceLastNDims<1>(index); }
    inline const DirectTensorView Tensor::getRow(const std::vector<size_t> &index) const    { return sliceLastNDims<1>(index); }
    inline       DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index)       { return sliceLastNDims<2>(index); }
    inline const DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index) const { return sliceLastNDims<2>(index); }

    inline bool Tensor::operator==(const Tensor &t) const
    {
        PROFILE
        if(!sizeMatch(shape, t.shape)) return false;
        
        for (size_t i = 0; i < size; i++)
            if(b[i] != t.b[i]) return false;
        
        return true;
    }

    inline void reprint(std::ostream& os, const Tensor& t, size_t depth, std::vector<size_t>& index)
    {
        if (depth == 0) { if (t.size) os << t(index); return; }

        if (depth > 1)
        {
            os << "[\n";
            for (size_t i = 0; i < t.shape.size() - depth + 1; i++) os << "  ";
        }
        else
            os << "[";
        
        index.push_back(0);
        for (size_t i = 0; i + 1 < t.shape[t.shape.size() - depth]; i++)
        {
            index.back() = i;
            if (i == 4 && depth == 1 && t.shape.back() > 8) { os << "...";  i = t.shape.back() - 4 -1;}
            else if (i == 3 && depth == 2 && t.shape[t.shape.size()-depth] > 6) { os << "...";  i = t.shape[t.shape.size()-depth] - 3 -1;}
            else if (i == 2 && depth == 3 && t.shape[t.shape.size()-depth] > 4) { os << "...";  i = t.shape[t.shape.size()-depth] - 2 -1;}
            else if (i == 1 && depth >= 4 && t.shape[t.shape.size()-depth] > 2) { os << "...";  i = t.shape[t.shape.size()-depth] - 1 -1;}
            else
                reprint(os, t, depth-1, index);
            if (depth > 1)
            {
                os << ",\n";
                for (size_t i = 0; i < t.shape.size() - depth + 1; i++) os << "  ";
            }
            else
                os << ", ";

        }
        index.back() = t.shape[t.shape.size() - depth] - 1;
        reprint(os, t, depth-1, index);
        index.pop_back();
        
        if (depth > 1)
        {
            os << "\n";
            for (size_t i = 0; i < t.shape.size() - depth; i++) os << "  ";
        }
        os << "]";
    }

    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) 
    {
        os << "Tensor" << std::endl << "Shape: (";
        for (size_t i = 0; i < t.shape.size() - 1; i++)
            os << t.shape[i] << ", ";
        os << t.shape.back() << ")\n";

        std::vector<size_t> v;
        reprint(os, t, t.shape.size(), v);
        os << '\n';
        return os;
    }

    inline void Tensor::zero()
    {
        PROFILE
        for (size_t i = 0; i < size; i++)
            b[i] = 0;
    }  

    inline void Tensor::ones()
    {
        PROFILE
        for (size_t i = 0; i < size; i++) 
            b[i] = 1;
    }

    inline void Tensor::rand()
    {
        PROFILE
        float64 inv = 1. / RAND_MAX;
        for (size_t i = 0; i < size; i++) 
            b[i] = std::rand() * inv;
    }

    inline void Tensor::rand(float64 start, float64 end)
    {
        PROFILE
        float64 l = (end - start) / RAND_MAX;
        for (size_t i = 0; i < size; i++) 
            b[i] = std::rand() * l + start;
    }

    inline void Tensor::randUniform(float64 a, float64 b)
    {
        PROFILE
        std::uniform_real_distribution<> dis(a, b);
        for (size_t i = 0; i < size; i++) 
            this->b[i] = dis(gen);
    }

    inline void Tensor::randNormal(float64 mean, float64 std)
    {
        PROFILE
        std::normal_distribution<double> distribution(mean,std);
        for (size_t i = 0; i < size; i++) 
            b[i] = distribution(gen);
    }

    inline void Tensor::costant(float64 val)
    {
        PROFILE
        for (size_t i = 0; i < size; i++) 
            b[i] = val;
    }

    inline bool Tensor::sizeMatch(const std::vector<size_t>& s1, const std::vector<size_t>& s2)
    {
        PROFILE
        // if (size != t.size) return false;
        size_t end = std::min(s1.size(), s2.size());
        for (size_t i = 1; i <= end; i++)
            if (s1[s1.size() - i] != s2[s2.size() - i])
                return false;
        for (size_t i = 0; i < s1.size() - end; i++) if (s1[i] != 1) return false;
        for (size_t i = 0; i < s2.size() - end; i++) if (s2[i] != 1) return false;
        
        return true;
    }


    /* ---------- Functions ---------- */

    inline bool broadcastable(const std::vector<size_t>& i1, const std::vector<size_t>& i2)
    {
        PROFILE
        auto p1 = i1.end() - 1;
        auto p2 = i2.end() - 1;

        while (p1 != i1.begin() && p2 != i2.begin())
        {
            if (*p1 != *p2 && *p1 != 1 && *p2 != 1)
                return false;
            p1--, p2--;
        }
        return true;
    }

    inline Tensor empty_like(const Tensor& t)
    {
        return {t.shape};
    }

    inline Tensor zeros_like(const Tensor& t)
    {
        Tensor zl(t.shape);
        zl.zero();
        return zl;
    }

    inline Tensor ones_like(const Tensor& t)
    {
        Tensor zl(t.shape);
        zl.ones();
        return zl;
    }

    template<float64(*fn)(float64)>
    inline Tensor forEach(const Tensor& t)
    {
        PROFILE
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.size; i++) ret.b[i] = fn(t.b[i]);
        return ret;
    }

    inline Tensor forEach(const Tensor& t, std::function<float64(float64)> fn)
    {
        PROFILE
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.size; i++) ret.b[i] = fn(t.b[i]);
        return ret;
    }

    template<float64(*fn)(float64)>
    inline Tensor& forEachInPlace(Tensor& t)
    {
        PROFILE
        for (size_t i = 0; i < t.size; i++) t.b[i] = fn(t.b[i]);
        return t;
    }

    inline Tensor& forEachInPlace(Tensor& t, std::function<float64(float64)> fn)
    {
        PROFILE
        for (size_t i = 0; i < t.size; i++) t.b[i] = fn(t.b[i]);
        return t;
    }

    template <void(*fn)(float64&, float64)>
    inline Tensor op_along_axes(const Tensor& t, size_t d, const float64 init_val)
    {
        PROFILE
        d = t.shape.size() - d - 1;
        auto shape = t.shape;
        shape[d] = std::min((size_t)1, shape[d]);
        Tensor ret(shape);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= shape[i];
        for (size_t i = d+1; i < shape.size(); i++) stride *= shape[i];
        
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

    template <void(*fn)(float64&, float64)>
    inline float64 op_along_all_axes(const Tensor& t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
    }

    template<float64(*fn)(float64, float64)>
    inline void broadcast_ew_assign(Tensor& dst,         const Tensor& src1,   const Tensor& src2,
                                    const size_t* shape, const size_t* shape1, const size_t* shape2,
                                    size_t depth,
                                    size_t off, size_t off1, size_t off2)
    {
        if (depth > 1)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                broadcast_ew_assign<fn>(dst, src1, src2, shape+1, shape1+1, shape2+1, depth-1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
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
    template <float64(*fn)(float64, float64)>
    inline Tensor ew_or_broadcast(const Tensor& t1, const Tensor& t2, const char* err_msg)
    {
        PROFILE
        Tensor result;
        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            result.resize(t1.shape);
            for (size_t i = 0; i < t1.size; i++)
                result.b[i] = fn(t1.b[i], t2.b[i]);
        }
        else if(broadcastable(t1.shape, t2.shape))
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
    template <float64(*fn)(float64, float64)>
    inline void ew_or_broadcast_assign(Tensor& t1, const Tensor& t2, const char* err_msg)
    {
        PROFILE
        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            for (size_t i = 0; i < t1.size; i++)
                t1.b[i] = fn(t1.b[i], t2.b[i]);
        }
        else if(broadcastable(t1.shape, t2.shape))
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
    template <float64(*fn)(float64, float64)>
    inline void ew_or_left_broadcast_assign(Tensor& t1, const Tensor& t2, const char* err_msg)
    {
        PROFILE
        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            for (size_t i = 0; i < t1.size; i++)
                t1.b[i] = fn(t1.b[i], t2.b[i]);
        }
        else if(broadcastable(t1.shape, t2.shape))
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

    template<auto fn, typename... Args>
    inline void broadcast_op_impl(float64* dst,        const float64* src1,  const float64* src2,
                                  const size_t* shape, const size_t* shape1, const size_t* shape2,
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
                    dst,     src1,     src2,
                    shape+1, shape1+1, shape2+1,
                    depth-1,
                    foff, foff1, foff2,
                    off  * *shape  +  i,
                    off1 * *shape1 + (i & bdc1),
                    off2 * *shape2 + (i & bdc2),
                    args...);
        else
            for (size_t i = 0; i < *shape; i++)
                fn(dst  + (off  * *shape  +  i        )*foff,
                   src1 + (off1 * *shape1 + (i & bdc1))*foff1,
                   src2 + (off2 * *shape2 + (i & bdc2))*foff2,
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
    template<auto fn, typename... Args>
    inline void broadcast_op(float64* dst,        const float64* src1,  const float64* src2,
                             const size_t* shape, const size_t* shape1, const size_t* shape2,
                             size_t depth,
                             size_t foff, size_t foff1, size_t foff2,
                             Args... args)
    {
        PROFILE
        broadcast_op_impl<fn, Args...>(dst, src1, src2, shape, shape1, shape2, depth, foff, foff1, foff2, (size_t)0, (size_t)0, (size_t)0, args...);
    }
    
    inline RedFish::Tensor stack(const RedFish::Tensor& t1, const RedFish::Tensor& t2, size_t dim)
    {
        PROFILE
        if (t1.shape.size() <= dim)
            throw std::length_error("Tensor has not that many dimensions");
        
        std::vector<size_t> t1_shape = t1.shape;
        std::vector<size_t> t2_shape = t2.shape;

        int t1_1 = 0;
        for (size_t i = 0; i < (int64_t)t1.shape.size() - dim; i++)
            if (t1_shape[i] == 1) t1_shape.erase(t1_shape.begin()), t1_1++;
            else break;
        
        int t2_1 = 0;
        for (size_t i = 0; i < (int64_t)t2.shape.size() - dim; i++)
            if (t2_shape[i] == 1) t2_shape.erase(t2_shape.begin()), t2_1++;
            else break;
        
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
            if (i == t1_shape.size() - dim - 1) t3_shape.push_back(t1_shape[i] + t2_shape[i]);
            else t3_shape.push_back(t1_shape[i]);

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
            for (size_t j = 0; j < n1; j++) t3.b[in3 + j]      = t1.b[in1 + j];
            for (size_t k = 0; k < n2; k++) t3.b[in3 + n1 + k] = t2.b[in2 + k];
        }
        
        return t3;
    }

    inline bool debug(const RedFish::Tensor &t, const RedFish::Tensor &result, float64 delta) 
    {    
        if(!t.sizeMatch(result.getShape(), t.getShape())) return false;
    
        size_t size = 1;
        for (size_t i = 0; i < t.getShape().size(); i++)
            size += t.getShape()[i];
        

        for (size_t i = 0; i < size; i++)
            if(std::abs(t.b[i] - result.b[i]) > delta) return false;
        
        return true;
    }

} // namespace RedFish

namespace std
{

    inline RedFish::Tensor sqrt(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::sqrt>(t);
    }

    inline RedFish::Tensor exp(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::exp>(t);
    }

    inline RedFish::Tensor log(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::log>(t);
    }

    inline RedFish::Tensor pow(const RedFish::Tensor& t, RedFish::float64 power)
    {
        PROFILE
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = std::pow(t.b[i], power);
        
        return ret;
    }

    inline RedFish::Tensor pow(const RedFish::Tensor& t, const RedFish::Tensor& power)
    {
        PROFILE
        if (!t.sizeMatch(t.shape, power.shape))
            throw std::length_error("Tensor sizes not matching in std::pow operation");

        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = std::pow(t.b[i], power.b[i]);

        return ret;
    }

}
