#pragma once 

#define CHECK_BOUNDS

#include <iostream>
#include <limits.h>
#include <vector>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <cmath>

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
    
    template <void(*fn)(float64&, float64)>
    Tensor opAlongAxes(const Tensor&, size_t, const float64);
    template <void(*fn)(float64&, float64)>
    float64 opAlongAllAxes(const Tensor&, const float64);
    template <float64(*fn)(float64, float64)>
    void for_(Tensor&, const Tensor&, const Tensor&, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t);
    template <float64(*fn)(float64, float64)>
    Tensor ewOrBroadcast(const Tensor&, const Tensor&, const char*);
    template <float64(*fn)(float64, float64)>
    void ewOrBroadcastAssign(Tensor&, const Tensor&, const char*);
    bool broadcastable(const std::vector<size_t>&, const std::vector<size_t>&);

    class Tensor
    {
    public:
        Tensor(const std::vector<size_t>& shape = {});
        Tensor(const size_t* shape, size_t len);
        Tensor(const std::vector<size_t>& shape, float64* buff, bool copy = true);
        Tensor(const Tensor& t);                    // Copy Constructor
        ~Tensor();

        Tensor& operator=(const Tensor& t);
        Tensor& operator=(Tensor&& t);
        void resize(const std::vector<size_t>& shape);
        void reshape(const std::vector<size_t>& shape);

        Tensor matmul(const Tensor& t) const;
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

        float64 squareSum() const;
        Tensor  squareSum(size_t dimension) const;
        float64 max() const;
        Tensor  max(size_t dimension) const;
        float64 min() const;
        Tensor  min(size_t dimension) const;
        float64 sum() const;
        Tensor  sum(size_t dimension) const;

        float64& operator()(const size_t* index);
        float64  operator()(const size_t* index) const;

        float64& operator()();
        float64  operator()() const;

        float64& operator()(size_t x);
        float64  operator()(size_t x) const;

        float64& operator()(size_t x, size_t y);
        float64  operator()(size_t x, size_t y) const;

        float64& operator()(size_t x, size_t y, size_t z);
        float64  operator()(size_t x, size_t y, size_t z) const;

        bool operator==(const Tensor& other) const;

        friend Tensor operator-(const float64, const Tensor&);
        friend Tensor operator/(const float64, const Tensor&);
        friend std::ostream& operator<<(std::ostream&, const Tensor&);
        friend void reprint(std::ostream&, const Tensor&, size_t, std::vector<size_t>&);
        friend Tensor empty_like(const Tensor&);
        template<float64(*fn)(float64)>
        friend Tensor forEach(const Tensor&);
        friend Tensor forEach(const Tensor&, std::function<float64(float64)>);
        template <void(*fn)(float64&, float64)>
        friend Tensor opAlongAxes(const Tensor&, size_t, const float64);
        template <void(*fn)(float64&, float64)>
        friend float64 opAlongAllAxes(const Tensor&, const float64);
        template<float64(*fn)(float64, float64)>
        friend void for_(Tensor&, const Tensor&, const Tensor&, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t);
        template <float64(*fn)(float64, float64)>
        friend Tensor ewOrBroadcast(const Tensor&, const Tensor&, const char*);
        template <float64(*fn)(float64, float64)>
        friend void ewOrBroadcastAssign(Tensor&, const Tensor&, const char*);
        friend Tensor std::sqrt(const Tensor&);
        friend Tensor std::exp(const Tensor&);
        friend Tensor std::log(const Tensor&);
        friend Tensor std::pow(const Tensor&, RedFish::float64);
        friend Tensor std::pow(const Tensor&, const Tensor&);

        friend Tensor stack(const Tensor& t1, const Tensor& t2, size_t dim);

        void zero();
        void ones();
        void rand();
        void rand(float64 start, float64 end);
        
        static bool sizeMatch(const std::vector<size_t>& s1, const std::vector<size_t>& s2);

        size_t colSize() const { return this->shape.back(); }
        size_t rowSize() const { return *(this->shape.end()-2); }
        size_t getSize() const { return size; }
        std::vector<size_t> getShape() { return shape; }

    private:
        std::unique_ptr<float64[]> b;
        size_t size;
        std::vector<size_t> shape;
    };

    inline Tensor::Tensor(const std::vector<size_t>& shape)
        :shape(shape)
    {
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        b = std::make_unique<float64[]>(size);
    }

    inline Tensor::Tensor(const size_t* shape, size_t len)
        :shape(shape, shape + len)
    {
        size = 1;
        for (size_t i = 0; i < len; i++)
            size *= shape[i];

        b = std::make_unique<float64[]>(size);
    }

    inline Tensor::Tensor(const std::vector<size_t>& shape, float64 *buff, bool copy)
        :shape(shape)
    {
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (copy)
        {
            b = std::make_unique<float64[]>(size);
            std::copy(buff, buff + size, b.get());
        }
        else
            b.reset(buff);
    }

    inline Tensor::Tensor(const Tensor& t)
    {
        this->shape = t.shape;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }

    inline Tensor& Tensor::operator=(const Tensor& t)
    {
        this->shape = t.shape;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator=(Tensor &&t)
    {
        this->shape = t.shape;
        this->size = t.size;
        this->b = std::move(t.b);
        t.shape = {0};
        t.size = 0;

        return *this;
    }

    inline Tensor::~Tensor() { }

    inline void Tensor::resize(const std::vector<size_t>& shape)
    {
        this->shape = shape;
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        b = std::make_unique<float64[]>(size);
    }

    inline void Tensor::reshape(const std::vector<size_t> &shape)
    {
        size_t new_size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            new_size *= shape[i];

        if (new_size != size)
            throw std::length_error("Invalid new shape in Tensor reshape");

        this->shape = shape;
    }

    /* inline void matmul_impl(float64* dst, const float64* m1, const float64* m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        for (size_t j = 0; j < rows; j++)
            for (size_t k = 0; k < cols; k++)
                dst[j*ld0 + k] = m1[j*ld1] * m2[k]; 
        for (size_t i = 1; i < mid; i++)
            for (size_t j = 0; j < rows; j++)
                for (size_t k = 0; k < cols; k++)
                    dst[j*ld0 + k] += m1[j*ld1 + i] * m2[i*ld2 + k];
    } */

    inline void matmul_impl(float64* dst, const float64* m1, const float64* m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        for (size_t j = 0; j < rows; j++)
            for (size_t k = 0; k < cols; k++)
                dst[j*ld0 + k] = m1[j] * m2[k]; 
        for (size_t i = 1; i < mid; i++)
            for (size_t j = 0; j < rows; j++)
                for (size_t k = 0; k < cols; k++)
                    dst[j*ld0 + k] += m1[j + i*ld1] * m2[i*ld2 + k];
    }

    /* inline void matmul_gotoblas(float64* dst, const float64* m1, const float64* m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        for (size_t jc = 0; jc < N; jc += steps of NC)
            for (size_t kc = 0; kc < K; kc += steps of KC)
                //Pack KCxNC block of B
                for (size_t ic = 0; ic < M; ic += steps of MC)
                    //Pack MCxKC block of A
        ----------------Macro Kernel------------
                    for (size_t jr = 0; jr < NC; jr += steps of NR)
                        for (size_t ir = 0; ir < MC; ir += steps of MR)
        ----------------Micro Kernel------------
                            for (size_t k = 0; k < KC; k++)
                                //update MRxNR block of C matrix
    } */

    inline void for_matmul(float64* dst, const float64* src1, const float64* src2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2, 
                           const size_t* shape, const size_t* shape1, const size_t* shape2, size_t depth, size_t off, size_t off1, size_t off2)
    {
        if (depth > 2)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                for_matmul(dst, src1, src2, rows, mid, cols, ld0, ld1, ld2, shape+1, shape1+1, shape2+1, depth-1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
        else
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                matmul_impl(dst + off * *shape + i, src1 + off1 * *shape1 + (i & bdc1), src2 + off2 * *shape2 + (i & bdc2), rows, mid, cols, ld0, ld1, ld2);
    }

    inline Tensor Tensor::matmul(const Tensor &t) const
    {
        Tensor result;
        std::vector<size_t> shapeT1, matShapeT1(  shape.begin() + std::max<int64_t>(0, (int64_t)  shape.size() - 2),   shape.end());
        std::vector<size_t> shapeT2, matShapeT2(t.shape.begin() + std::max<int64_t>(0, (int64_t)t.shape.size() - 2), t.shape.end());
        size_t size1 = 1, size2 = 1;

        for (size_t i = 0; i <   shape.size()-2; i++) shapeT1.push_back(  shape[i]), size1 *=   shape[i];
        for (size_t i = 0; i < t.shape.size()-2; i++) shapeT2.push_back(t.shape[i]), size2 *= t.shape[i];
        matShapeT1.insert(matShapeT1.begin(), std::max<int64_t>(0, (int64_t)2 -   shape.size()), 1);
        matShapeT2.insert(matShapeT2.begin(), std::max<int64_t>(0, (int64_t)2 - t.shape.size()), 1);
        if (t.shape.size() == 1) std::swap(matShapeT2[0], matShapeT2[1]);

        if (shapeT1.size() > shapeT2.size())
            shapeT2.insert(shapeT2.begin(), shapeT1.size() - shapeT2.size(), 1);
        else if (shapeT1.size() < shapeT2.size())
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
            Tensor tt = this->T();
            for (size_t i = 0; i < size1; i++)
                matmul_impl(result.b.get() + i*matsize0, tt.b.get() + i*matsize1, t.b.get() + i*matsize2, rows, mid, cols, cols, rows, cols);
        }
        else if(broadcastable(shapeT1, shapeT2)) /* To Fix */
        {
            std::vector<size_t> shapeDst(shapeT1.size());
            for (size_t i = 0; i < shapeT1.size(); i++)
                shapeDst[i] = shapeT1[i] == 1 ? shapeT2[i] : shapeT1[i];

            shapeDst.push_back(rows);
            shapeDst.push_back(cols);
            shapeT1.push_back(rows);
            shapeT1.push_back(mid);
            shapeT2.push_back(mid);
            shapeT2.push_back(cols);
                
            result.resize(shapeDst);
            if (result.size)
                for_matmul(result.b.get(), this->b.get(), t.b.get(), rows, mid, cols, cols, mid, cols,
                           shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size(), (size_t)0,(size_t)0,(size_t)0);
        }
        else
            throw std::length_error("Shapes not matching in Tensor matmul");

        return result;
    }

    inline Tensor matmul(const Tensor& t1, const Tensor& t2)
    {
        return t1.matmul(t2);
    }

    inline Tensor Tensor::T() const
    {
        Tensor t(shape);
        size_t rows = t.shape[t.shape.size()-2], cols = t.shape.back(), end = 1;
        std::swap(t.shape.back(), t.shape[t.shape.size()-2]);

        for (size_t i = 0; i < t.shape.size() - 2; i++)
            end *= t.shape[i];

        const size_t block_size = 16;
        for (size_t i = 0, stride = rows*cols; i < end; i++)
        {
            float64* tp = t.b.get() + i * stride, *thisp = this->b.get() + i * stride;

            /* --- slower, naive version */
            // for (size_t r = 0, r_cols = r*cols; r < rows; r++, r_cols = r*cols)
            //     for (size_t c = 0; c < cols; c++)
            //         tp[c*rows + r] = thisp[r_cols + c];

            size_t crows = rows - rows % block_size, ccols = cols - cols % block_size;
            for (size_t k = 0, kb = block_size; k < crows; k += block_size, kb += block_size)
            {
                for (size_t j = 0, jb = block_size; j < ccols; j += block_size, jb += block_size)
                    for (size_t r = k, r_cols = r*cols; r < kb; r++, r_cols = r*cols)
                        for (size_t c = j; c < jb; c++)
                            tp[c*rows + r] = thisp[r_cols + c];

                for (size_t r = k, r_cols = r*cols; r < kb; r++, r_cols = r*cols)
                    for (size_t c = ccols; c < cols; c++)
                        tp[c*rows + r] = thisp[r_cols + c];
            }
            for (size_t j = 0, jb = block_size; j < ccols; j += block_size, jb += block_size)
                for (size_t r = crows, r_cols = r*cols; r < rows; r++, r_cols = r*cols)
                    for (size_t c = j; c < jb; c++)
                        tp[c*rows + r] = thisp[r_cols + c];
            for (size_t r = crows, r_cols = r*cols; r < rows; r++, r_cols = r*cols)
                for (size_t c = ccols; c < cols; c++)
                    tp[c*rows + r] = thisp[r_cols + c];
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
        return ewOrBroadcast<fn>(*this, t, "Tensor sizes not matching in sum operation");
    }

    inline Tensor Tensor::operator+(const float64 val) const
    {
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
        ewOrBroadcastAssign<fn>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    inline Tensor& Tensor::operator+=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] += val;

        return *this;
    }

    inline Tensor Tensor::operator-(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1-n2;};
        return ewOrBroadcast<fn>(*this, t, "Tensor sizes not matching in subtraction operation");
    }

    inline Tensor Tensor::operator-(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - val;

        return result;
    }

    inline Tensor Tensor::operator-() const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = -this->b[i];

        return result;
    }

    inline Tensor operator-(const float64 val, const Tensor& t)
    {
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val - t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator-=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1-n2;};
        ewOrBroadcastAssign<fn>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    inline Tensor& Tensor::operator-=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] -= val;

        return *this;
    }

    inline Tensor Tensor::operator*(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1*n2;};
        return ewOrBroadcast<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
    }

    inline Tensor Tensor::operator*(const float64 val) const
    {
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
        ewOrBroadcastAssign<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    inline Tensor& Tensor::operator*=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] *= val;

        return *this;
    }

    inline Tensor Tensor::operator/(const Tensor& t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1/n2;};
        return ewOrBroadcast<fn>(*this, t, "Tensor sizes not matching in division operation");
    }

    inline Tensor Tensor::operator/(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / val;

        return result;
    }

    inline Tensor operator/(const float64 val, const Tensor& t)
    {
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val / t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator/=(const Tensor& t)
    {
        constexpr auto fn = [](float64 n1, float64 n2) {return n1+n2;};
        ewOrBroadcastAssign<fn>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }

    inline Tensor& Tensor::operator/=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] /= val;

        return *this;
    }

    inline float64 Tensor::squareSum() const
    {
        constexpr auto fn = [](float64& sq_sum, float64 n) {sq_sum += n*n;};
        return opAlongAllAxes<fn>(*this, 0.);
    }

    inline Tensor Tensor::squareSum(size_t d) const
    {
        constexpr auto fn = [](float64& sq_sum, float64 n) {sq_sum += n*n;};
        return opAlongAxes<fn>(*this, d, 0.);
    }

    inline float64 Tensor::max() const
    {
        constexpr auto fn = [](float64& max, float64 n) {if (max < n) max = n;};
        return opAlongAllAxes<fn>(*this, -std::numeric_limits<float64>::infinity());
    }

    inline Tensor Tensor::max(size_t d) const
    {
        constexpr auto fn = [](float64& max, float64 n) {if (max < n) max = n;};
        return opAlongAxes<fn>(*this, d, -std::numeric_limits<float64>::infinity());
    }

    inline float64 Tensor::min() const
    {
        constexpr auto fn = [](float64& min, float64 n) {if (min > n) min = n;};
        return opAlongAllAxes<fn>(*this, std::numeric_limits<float64>::infinity());
    }

    inline Tensor Tensor::min(size_t d) const
    {
        constexpr auto fn = [](float64& min, float64 n) {if (min > n) min = n;};
        return opAlongAxes<fn>(*this, d, std::numeric_limits<float64>::infinity());
    }

    inline float64 Tensor::sum() const
    {
        constexpr auto fn = [](float64& sum, float64 n) {sum += n;};
        return opAlongAllAxes<fn>(*this, 0.);
    }

    inline Tensor Tensor::sum(size_t d) const
    {
        constexpr auto fn = [](float64& sum, float64 n) {sum += n;};
        return opAlongAxes<fn>(*this, d, 0.);
    }

    inline float64& Tensor::operator()(const size_t* index)
    {
        #ifdef CHECK_BOUNDS 
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (index[i] < 0 || index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (size_t i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    inline float64 Tensor::operator()(const size_t* index)  const 
    {
        #ifdef CHECK_BOUNDS 
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (index[i] < 0 || index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (size_t i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    inline float64 &Tensor::operator()()
    {
        #ifdef CHECK_BOUNDS 
            if (shape.size() != 0)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[0];
    }

    inline float64 Tensor::operator()() const
    {
        #ifdef CHECK_BOUNDS 
            if (shape.size() != 0)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[0];
    }

    inline float64 &Tensor::operator()(size_t x)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= size || shape.size() < 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 Tensor::operator()(size_t x) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= size || shape.size() < 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || shape.size() < 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * shape[1] + y];
    }

    inline float64 Tensor::operator()(size_t x, size_t y) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || shape.size() < 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * shape[1] + y];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y, size_t z)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || z < 0 || z >= shape[2] || shape.size() < 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * shape[1] + y) * shape[2] + z];
    }

    inline float64 Tensor::operator()(size_t x, size_t y, size_t z) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || z < 0 || z >= shape[2] || shape.size() < 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * shape[1] + y) * shape[2] + z];
    }

    inline bool Tensor::operator==(const Tensor &t) const
    {
        if(!sizeMatch(shape, t.shape)) return false;
        
        for (size_t i = 0; i < size; i++)
            if(b[i] != t.b[i]) return false;
        
        return true;
    }

    inline void reprint(std::ostream& os, const Tensor& t, size_t depth, std::vector<size_t>& index)
    {
        if (depth == 0) { os << t(index.data()); return; }

        if (depth > 1)
        {
            os << "[\n";
            for (size_t i = 0; i < t.shape.size() - depth + 1; i++) os << "  ";
        }
        else
            os << "[";
        
        index.push_back(0);
        for (size_t i = 0; i < t.shape[t.shape.size() - depth] - 1; i++)
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
        for (size_t i = 0; i < size; i++)
            b[i] = 0;
    }  

    inline void Tensor::ones()
    {
        for (size_t i = 0; i < size; i++) 
            b[i] = 1;
    }

    inline void Tensor::rand()
    {
        float64 inv = 1. / RAND_MAX;
        for (size_t i = 0; i < size; i++) 
            b[i] = std::rand() * inv;
    }

    inline void Tensor::rand(float64 start, float64 end)
    {
        float64 l = (end - start) / RAND_MAX;
        for (size_t i = 0; i < size; i++) 
            b[i] = std::rand() * l + start;
    }

    inline bool Tensor::sizeMatch(const std::vector<size_t>& s1, const std::vector<size_t>& s2)
    {
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

    template<float64(*fn)(float64)>
    inline Tensor forEach(const Tensor& t)
    {
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = fn(t.b[i]);
        
        return ret;
    }

    inline Tensor forEach(const Tensor& t, std::function<float64(float64)> fn)
    {
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = fn(t.b[i]);
        
        return ret;
    }

    template <void(*fn)(float64&, float64)>
    inline Tensor opAlongAxes(const Tensor& t, size_t d, const float64 init_val)
    {
        d = t.shape.size() - d - 1;
        auto shape = t.shape;
        shape[d] = std::min((size_t)1, shape[d]);
        Tensor ret(shape);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= shape[i];
        for (size_t i = d+1; i < shape.size(); i++) stride *= shape[i];
        
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
    inline float64 opAlongAllAxes(const Tensor& t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
    }

    template<float64(*fn)(float64, float64)>
    inline void for_(Tensor& dst, const Tensor& src1, const Tensor& src2, const size_t* shape, const size_t* shape1, const size_t* shape2, size_t depth, size_t off, size_t off1, size_t off2)
    {
        if (depth > 1)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                for_<fn>(dst, src1, src2, shape+1, shape1+1, shape2+1, depth-1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
        else
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                dst.b[off * *shape + i] = fn(src1.b[off1 * *shape1 + (i & bdc1)], src2.b[off2 * *shape2 + (i & bdc2)]);
    }
    
    template <float64(*fn)(float64, float64)>
    inline Tensor ewOrBroadcast(const Tensor& t1, const Tensor& t2, const char* err_msg)
    {
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
                for_<fn>(result, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size(), (size_t)0, (size_t)0, (size_t)0);
        }
        else
            throw std::length_error(err_msg);

        return result;
    }

    template <float64(*fn)(float64, float64)>
    inline void ewOrBroadcastAssign(Tensor& t1, const Tensor& t2, const char* err_msg)
    {
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
                    for_<fn>(t1, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size(), (size_t)0, (size_t)0, (size_t)0);
            }
            else
            {
                Tensor result(shapeDst);
                if (result.size)
                    for_<fn>(result, t1, t2, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size(), (size_t)0, (size_t)0, (size_t)0);
                t1 = std::move(result);
            }
        }
        else
            throw std::length_error(err_msg);
    }
    
    inline RedFish::Tensor stack(const RedFish::Tensor& t1, const RedFish::Tensor& t2, size_t dim)
    {
        if (t1.shape.size() <= dim)
            throw std::length_error("Tensor has not that many dimensions");
        
        std::vector<size_t> t1_shape = t1.shape;
        std::vector<size_t> t2_shape = t2.shape;

        int t1_1 = 0;
        for (size_t i = 0; i < t1.shape.size() - dim; i++)
            if (t1_shape[i] == 1) t1_shape.erase(t1_shape.begin()), t1_1++;
            else break;
        
        int t2_1 = 0;
        for (size_t i = 0; i < t2.shape.size() - dim; i++)
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
}

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
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::pow(t.b[i], power);
        
        return ret;
    }

    inline RedFish::Tensor pow(const RedFish::Tensor& t, const RedFish::Tensor& power)
    {
        if (!t.sizeMatch(t.shape, power.shape))
            throw std::length_error("Tensor sizes not matching in std::pow operation");

        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::pow(t.b[i], power.b[i]);

        return ret;
    }

}
