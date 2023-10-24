#pragma once 

#define DEBUG

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>

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

    class Tensor
    {
    private:
        
    public:
        Tensor(const std::vector<size_t>& dim = {});
        Tensor(const size_t* dim, size_t len);
        Tensor(const std::vector<size_t>& dim, float64* buff, bool copy = true);
        Tensor(const Tensor& t);                    // Copy Constructor
        ~Tensor();

        Tensor& operator=(const Tensor& t);
        void resize(const std::vector<size_t>& dim);

        Tensor matmul(const Tensor& t) const;
    
        // Operetors
        Tensor  operator+(const Tensor& t)  const;
        Tensor  operator+(const double val) const;
        Tensor& operator+=(const Tensor& t);
        Tensor& operator+=(const double val);
        Tensor  operator-(const Tensor& t)  const;
        Tensor  operator-(const double val) const;
        Tensor  operator-() const;
        Tensor& operator-=(const Tensor& t);
        Tensor& operator-=(const double val);
        Tensor  operator*(const Tensor& t)  const;
        Tensor  operator*(const double val) const;
        Tensor& operator*=(const Tensor& t);
        Tensor& operator*=(const double val);
        Tensor  operator/(const Tensor& t)  const;
        Tensor  operator/(const double val) const;
        Tensor& operator/=(const Tensor& t);
        Tensor& operator/=(const double val);

        float64 squareSum() const;
        Tensor  squareSum(size_t dimension) const;
        float64 max() const;
        Tensor  max(size_t dimension) const;
        float64 sum() const;
        Tensor  sum(size_t dimension) const;

        float64& operator()(const size_t* index);
        float64  operator()(const size_t* index) const;

        float64& operator()(size_t x);
        float64  operator()(size_t x) const;

        float64& operator()(size_t x, size_t y);
        float64  operator()(size_t x, size_t y) const;

        float64& operator()(size_t x, size_t y, size_t z);
        float64  operator()(size_t x, size_t y, size_t z) const;

        friend Tensor operator-(const double, const Tensor&);
        friend Tensor operator/(const double, const Tensor&);
        friend std::ostream& operator<<(std::ostream&, const Tensor&);
        friend void reprint(std::ostream&, const Tensor&, size_t, std::vector<size_t>&);
        friend Tensor empty_like(const Tensor&);
        template<float64(*fn)(float64)>
        friend Tensor forEach(const Tensor&);
        friend Tensor forEach(const Tensor&, std::function<float64(float64)>);
        template <void(*fn)(float64&, float64), float64(*init)(float64)>
        friend Tensor opAlongAxes(const Tensor&);
        friend Tensor std::sqrt(const Tensor&);
        friend Tensor std::exp(const Tensor&);
        friend Tensor std::log(const Tensor&);
        friend Tensor std::pow(const Tensor&, float64);
        friend Tensor std::pow(const Tensor&, const Tensor&);

        void zero();
        bool sizeMatch(const Tensor& t) const;

        size_t colSize() const { return this->dim.back(); }
        size_t rowSize() const { return *(this->dim.end()-2); }
        size_t getSize() const { return size; }

    private:
        std::unique_ptr<float64[]> b;
        size_t size;
        std::vector<size_t> dim;
    };
    
    inline Tensor::Tensor(const std::vector<size_t>& dim)
        :dim(dim)
    {
        size = 1;
        for (size_t i = 0; i < dim.size(); i++)
            size *= dim[i];

        b = std::make_unique<float64[]>(size);
    }

    inline Tensor::Tensor(const size_t* dim, size_t len)
        :dim(dim, dim + len)
    {
        size = 1;
        for (size_t i = 0; i < len; i++)
            size *= dim[i];

        b = std::make_unique<float64[]>(size);
    }

    inline Tensor::Tensor(const std::vector<size_t>& dim, float64 *buff, bool copy)
        :dim(dim)
    {
        size = 1;
        for (size_t i = 0; i < dim.size(); i++)
            size *= dim[i];

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
        this->dim = t.dim;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (int i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }

    inline Tensor& Tensor::operator=(const Tensor& t)
    {
        this->dim = t.dim;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (int i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }
    
    inline Tensor::~Tensor() { }

    inline void Tensor::resize(const std::vector<size_t>& dim)
    {
        this->dim = dim;
        size = 1;
        for (size_t i = 0; i < dim.size(); i++)
            size *= dim[i];

        b = std::make_unique<float64[]>(size);
    }

    inline bool broadcastable(const std::vector<size_t>& i1, const std::vector<size_t>& i2)
    {
        auto p1 = i1.end() - 1;
        auto p2 = i2.end() - 1;

        while (p1 != i1.begin() && p2 != i2.begin())
        {
            if (*p1 != *p2 && (*p1 != 1 || *p2 == 0) && (*p2 != 1 || *p1 == 0))
                return false;
            p1--, p2--;
        }
        return true;
    }

    inline Tensor Tensor::matmul(const Tensor &t) const
    {
        if (this->dim.size() == 0 || t.dim.size() == 0)
            throw std::length_error("Zero dimensional tensor in matmul !");
            
        Tensor mul({0});
        if(this->dim.size() == 1 && t.dim.size() == 1)
        { 
            if (this->dim[0] != t.dim[0])
                throw std::length_error("Size not matching in tensor matmul !");
            
            mul.resize({1});
            mul((size_t)0) = 0.;
            for (size_t i = 0; i < dim[0]; i++)
                mul((size_t)0) += b[i] * t(i);
        }
        else if (this->dim.size() == 2 && t.dim.size() == 2)
        {
            if (this->colSize() != t.rowSize())
                throw std::length_error("Size not matching in tensor matmul !");
            
            mul.resize({ rowSize(), t.colSize() });
            for (size_t j = 0, rs = rowSize(); j < rs; j++)
                for (size_t k = 0, tcs = t.colSize(); k < tcs; k++)
                    mul(j, k) = this->operator()(j, 0) * t(0, k); 
            for (size_t i = 1, cs = this->colSize(); i < cs; i++)
                for (size_t j = 0, rs = rowSize(); j < rs; j++)
                    for (size_t k = 0, tcs = t.colSize(); k < tcs; k++)
                        mul(j, k) += this->operator()(j, i) * t(i, k); 
        }
        else if (this->dim.size() == 1 && t.dim.size() == 2)
        {
            if (this->colSize() != t.rowSize())
                throw std::length_error("Size not matching in tensor matmul !");
            
            mul.resize({ 1, t.colSize() });
            for (size_t k = 0, tcs = t.colSize(); k < tcs; k++)
                mul(0, k) = this->operator()((size_t)0) * t(0, k); 
            for (size_t i = 1, cs = this->colSize(); i < cs; i++)
                for (size_t k = 0, tcs = t.colSize(); k < tcs; k++)
                    mul(0, k) += this->operator()(i) * t(i, k); 
        }
        else if (this->dim.size() == 2 && t.dim.size() == 1)
        {
            if (this->colSize() != t.rowSize())
                throw std::length_error("Size not matching in tensor matmul !");
            
            mul.resize({ this->rowSize() });
            for (size_t j = 0, rs = rowSize(); j < rs; j++)
                mul(j) = this->operator()(j, 0) * t((size_t)0); 
            for (size_t i = 1, cs = this->colSize(); i < cs; i++)
                for (size_t j = 0, rs = rowSize(); j < rs; j++)
                    mul(j) += this->operator()(j, i) * t(i);
        }
        else
        {
            if (this->dim.size() == 1)
            {
                if (this->colSize() != t.dim[t.dim.size()-2])
                    throw std::length_error("Size not matching in tensor matmul !");

                size_t batch_size = 1;
                for (size_t i = 0, end = t.dim.size()-2; i < end; i++)
                    batch_size *= t.dim[i];
                
                auto nsize = t.dim;
                nsize[nsize.size()-2] = 1;
                mul.resize(nsize);
                const auto thisp = this->b.get();
                for (size_t l = 0, t_cols = t.dim.back(); l < batch_size; l++)
                {
                    auto mulp = mul.b.get() + l*batch_size;
                    const auto tp = t.b.get() + l*batch_size;
                    for (size_t k = 0; k < t_cols; k++)
                        mulp[k] = thisp[0] * tp[k]; 
                    for (size_t i = 1; i < this->dim[0]; i++)
                        for (size_t k = 0, i_t_cols = i*t_cols; k < t_cols; k++)
                            mulp[k] += thisp[i] * tp[i_t_cols + k]; 
                }
            }
            else if (t.dim.size() == 1)
            {
                if (this->dim.back() != t.colSize())
                    throw std::length_error("Size not matching in tensor matmul !");

                size_t batch_size = 1;
                for (size_t i = 0, end = t.dim.size()-2; i < end; i++)
                    batch_size *= this->dim[i];
                
                auto nsize = this->dim;
                nsize[nsize.size()-2] = 1;
                mul.resize(nsize);

                const auto tp = t.b.get();
                for (size_t l = 0, this_rows = dim[dim.size()-2], this_cols = this->dim.back(); l < batch_size; l++)
                {
                    auto mulp = mul.b.get() + l*batch_size;
                    const auto thisp = this->b.get() + l*batch_size;
                    for (size_t j = 0; j < this_rows; j++)
                        mulp[j] = thisp[j*this_cols] * tp[0]; 
                    for (size_t j = 0; j < this_rows; j++)
                        for (size_t i = 1; i < this_cols; i++)
                            mulp[j] += thisp[j*this_cols + i] * tp[i];
                }
            }
            else
            {
                if (this->dim.back() != t.dim[t.dim.size()-2])
                    throw std::length_error("Size not matching in tensor matmul !");

                if (this->dim.size() > 2 && 
                    t.dim.size() > 2 && 
                    !broadcastable({this->dim.begin(), this->dim.end()-2}, {t.dim.begin(), t.dim.end()-2}))
                        throw std::length_error("Size not broadcastable in tensor matmul !");

                std::vector<size_t> nsize(std::max(this->dim.size(), t.dim.size()));
                std::vector<size_t> thissize(nsize.size() - this->dim.size(), 1);
                std::vector<size_t> tsize(nsize.size() - t.dim.size(), 1);
                thissize.insert(thissize.end(), this->dim.begin(), this->dim.end());
                tsize.insert(tsize.end(), t.dim.begin(), t.dim.end());

                nsize.back() = t.dim.back();
                nsize[nsize.size()-2] = dim[dim.size()-2];
                
                for (size_t i = 0; i < nsize.size()-2; i++)
                    nsize[i] = std::max(thissize[i], tsize[i]);
                
                mul.resize(nsize);

                size_t mul_cols      = nsize.back(),
                       mul_rows      = nsize[nsize.size()-2],
                       mul_mat_size  = mul_cols*mul_rows,
                       this_cols     = this->dim.back(),
                       this_rows     = this->dim[dim.size()-2],
                       this_mat_size = this_cols*this_rows,
                       t_cols        = t.dim.back(),
                       t_rows        = t.dim[t.dim.size()-2],
                       t_mat_size    = t_cols*t_rows;

                for (std::vector<size_t> index(nsize.size()-2, 0); index[0] < nsize[0];)
                {
                    size_t moff = index[0], thoff = std::min(index[0], thissize[0]-1), toff = std::min(index[0], tsize[0]-1);
                    for (size_t i = 1; i < index.size(); i++)
                    {
                        moff  = moff  *    nsize[i] + index[i];
                        thoff = thoff * thissize[i] + std::min(index[i], thissize[i]-1);
                        toff  = toff  *    tsize[i] + std::min(index[i],    tsize[i]-1);
                    }

                    auto       mulp  = mul.b.get()   + moff  * mul_mat_size;
                    const auto thisp = this->b.get() + thoff * this_mat_size;
                    const auto tp    = t.b.get()     + toff  * t_mat_size;
                    
                    for (size_t j = 0; j < this_rows; j++)
                        for (size_t k = 0; k < t_cols; k++)
                            mulp[j*mul_cols + k] = thisp[j*this_cols] * tp[k]; 
                    for (size_t i = 1; i < this_cols; i++)
                        for (size_t j = 0; j < this_rows; j++)
                            for (size_t k = 0; k < t_cols; k++)
                                mulp[j*mul_cols + k] += thisp[j*this_cols + i] * tp[i*t_cols + k]; 

                    bool add = true;
                    for (size_t i = index.size()-1; add && i > 0; i--)
                    {
                        if (index[i] + 1 < nsize[i])
                            index[i]++,
                            add = false;
                        else
                            index[i] = 0;
                    }
                    if (add) index[0]++;
                }
            }
        }

        return mul;
    }

    inline Tensor matmul(const Tensor& t1, const Tensor& t2)
    {
        return t1.matmul(t2);
    }

    /* Operetors */
    inline Tensor Tensor::operator+(const Tensor& t) const
    {
        Tensor result({0});
        if (sizeMatch(t))
        {
            result.resize(this->dim);
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] + t.b[i];
        }
        else if (broadcastable(this->dim, t.dim))
        {

        }
        else
            throw std::length_error("Tensor sizes not matching in sum operation");

        return result;
    }

    inline Tensor Tensor::operator+(const double val) const
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] + val;

        return result;
    }

    inline Tensor operator+(const double val, const Tensor& t)
    {
        return t + val;
    }

    inline Tensor& Tensor::operator+=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] += t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator+=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] += val;

        return *this;
    }

    inline Tensor Tensor::operator-(const Tensor& t) const
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in subtraction operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - t.b[i];

        return result;
    }

    inline Tensor Tensor::operator-(const double val) const
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - val;

        return result;
    }

    inline Tensor Tensor::operator-() const
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = -this->b[i];

        return result;
    }

    inline Tensor operator-(const double val, const Tensor& t)
    {
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val - t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator-=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] -= t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator-=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] -= val;

        return *this;
    }

    inline Tensor Tensor::operator*(const Tensor& t) const
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in multiplication operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] * t.b[i];

        return result;
    }

    inline Tensor Tensor::operator*(const double val) const
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] * val;

        return result;
    }

    inline Tensor operator*(const double val, const Tensor& t)
    {
        return t * val;
    }

    inline Tensor& Tensor::operator*=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] *= t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator*=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] *= val;

        return *this;
    }

    inline Tensor Tensor::operator/(const Tensor& t) const
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in division operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / t.b[i];

        return result;
    }

    inline Tensor Tensor::operator/(const double val) const
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / val;

        return result;
    }

    inline Tensor operator/(const double val, const Tensor& t)
    {
        Tensor ret = empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val / t.b[i];
        
        return ret;
    }

    inline Tensor& Tensor::operator/=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] /= t.b[i];

        return *this;
    }

    inline Tensor& Tensor::operator/=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] /= val;

        return *this;
    }

    inline float64 Tensor::squareSum() const
    {
        float64 sq_sum = 0.;
        for (size_t i = 0; i < size; i++)
            sq_sum += this->b[i]*this->b[i];
        return sq_sum;
    }

    inline Tensor Tensor::squareSum(size_t d) const
    {
        d = this->dim.size() - d - 1;
        auto dim = this->dim;
        dim[d] = std::min((size_t)1, dim[d]);
        Tensor ret(dim);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= dim[i];
        for (size_t i = d+1; i < dim.size(); i++) stride *= dim[i];
        
        for (size_t k = 0; k < tot; k++)
            for (size_t i = 0; i < stride; i++)
            {
                float64 sm = this->b[i + k * stride * this->dim[d]];
                for (size_t j = 1; j < this->dim[d]; j++)
                    sm += this->b[j * stride + i + k * stride * this->dim[d]]*this->b[j * stride + i + k * stride * this->dim[d]];
                
                ret.b[i + k * stride] = sm;
            }
        
        return ret;
    }

    inline float64 Tensor::max() const
    {
        float64 max = -std::numeric_limits<float64>::infinity();
        for (size_t i = 0; i < size; i++)
            if (max < this->b[i]) max = this->b[i];
        return max;
    }

    inline Tensor Tensor::max(size_t d) const
    {
        d = this->dim.size() - d - 1;
        auto dim = this->dim;
        dim[d] = std::min((size_t)1, dim[d]);
        Tensor ret(dim);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= dim[i];
        for (size_t i = d+1; i < dim.size(); i++) stride *= dim[i];
        size_t full_stride = stride * this->dim[d];

        for (size_t k = 0; k < tot; k++)
            for (size_t i = 0; i < stride; i++)
            {
                float64 tmax = this->b[i + k * full_stride];
                for (size_t j = 1; j < this->dim[d]; j++)
                {
                    float64 n = this->b[j * stride + i + k * full_stride];
                    if (n > tmax) tmax = n;
                }
                ret.b[i + k * stride] = tmax;
            }
        
        return ret;
    }

    inline float64 Tensor::sum() const
    {
        float64 sum = 0.;
        for (size_t i = 0; i < size; i++)
            sum += this->b[i];
        return sum;
    }

    inline Tensor Tensor::sum(size_t d) const
    {
        d = this->dim.size() - d - 1;
        auto dim = this->dim;
        dim[d] = std::min((size_t)1, dim[d]);
        Tensor ret(dim);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= dim[i];
        for (size_t i = d+1; i < dim.size(); i++) stride *= dim[i];
        
        for (size_t k = 0; k < tot; k++)
            for (size_t i = 0; i < stride; i++)
            {
                float64 sm = this->b[i + k * stride * this->dim[d]];
                for (size_t j = 1; j < this->dim[d]; j++)
                    sm += this->b[j * stride + i + k * stride * this->dim[d]];
                
                ret.b[i + k * stride] = sm;
            }
        
        return ret;
    }

    inline float64& Tensor::operator()(const size_t* index)
    {
        #ifdef DEBUG 
            for (int i = 0; i < dim.size(); i++)
            {
                if (index[i] < 0 || index[i] >= dim[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (int i = 0; i < this->dim.size() - 1; i++)
            n = (n + index[i]) * this->dim[i + 1];
        
        return this->b[n + index[this->dim.size() - 1]];
    }

    inline float64 Tensor::operator()(const size_t* index)  const 
    {
        #ifdef DEBUG 
            for (int i = 0; i < dim.size(); i++)
            {
                if (index[i] < 0 || index[i] >= dim[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (int i = 0; i < this->dim.size() - 1; i++)
            n = (n + index[i]) * this->dim[i + 1];
        
        return this->b[n + index[this->dim.size() - 1]];
    }

    inline float64 &Tensor::operator()(size_t x)
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || dim.size() != 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 Tensor::operator()(size_t x) const
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || dim.size() != 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y)
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || y < 0 || y >= dim[1] || dim.size() != 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * dim[1] + y];
    }

    inline float64 Tensor::operator()(size_t x, size_t y) const
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || y < 0 || y >= dim[1] || dim.size() != 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * dim[1] + y];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y, size_t z)
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || y < 0 || y >= dim[1] || z < 0 || z >= dim[2] || dim.size() != 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * dim[1] + y) * dim[2] + z];
    }

    inline float64 Tensor::operator()(size_t x, size_t y, size_t z) const
    {
        #ifdef DEBUG 
            if (x < 0 || x >= dim[0] || y < 0 || y >= dim[1] || z < 0 || z >= dim[2] || dim.size() != 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * dim[1] + y) * dim[2] + z];
    }

    inline void reprint(std::ostream& os, const Tensor& t, size_t depth, std::vector<size_t>& index)
    {
        if (depth == 0) { os << t(index.data()); return; }

        if (depth > 1)
        {
            os << "[\n";
            for (int i = 0; i < t.dim.size() - depth + 1; i++) os << "  ";
        }
        else
            os << "[";
        
        index.push_back(0);
        for (size_t i = 0; i < t.dim[t.dim.size() - depth] - 1; i++)
        {
            index.back() = i;
            if (i == 4 && depth == 1 && t.dim.back() > 8) { os << "...";  i = t.dim.back() - 4 -1;}
            else if (i == 3 && depth == 2 && t.dim[t.dim.size()-depth] > 6) { os << "...";  i = t.dim[t.dim.size()-depth] - 3 -1;}
            else if (i == 2 && depth == 3 && t.dim[t.dim.size()-depth] > 4) { os << "...";  i = t.dim[t.dim.size()-depth] - 2 -1;}
            else if (i == 1 && depth >= 4 && t.dim[t.dim.size()-depth] > 2) { os << "...";  i = t.dim[t.dim.size()-depth] - 1 -1;}
            else
                reprint(os, t, depth-1, index);
            if (depth > 1)
            {
                os << ",\n";
                for (int i = 0; i < t.dim.size() - depth + 1; i++) os << "  ";
            }
            else
                os << ", ";

        }
        index.back() = t.dim[t.dim.size() - depth] - 1;
        reprint(os, t, depth-1, index);
        index.pop_back();
        
        if (depth > 1)
        {
            os << "\n";
            for (int i = 0; i < t.dim.size() - depth; i++) os << "  ";
        }
        os << "]";
    }

    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) 
    {
        os << "Tensor" << std::endl << "Dim: (";
        for (int i = 0; i < t.dim.size() - 1; i++)
            os << t.dim[i] << ", ";
        os << t.dim.back() << ")\n";

        std::vector<size_t> v;
        reprint(os, t, t.dim.size(), v);
        os << '\n';
        return os;
    }

    inline void Tensor::zero()
    {
        for (int i = 0; i < size; i++)
            b[i] = 0;
    }   

    inline bool Tensor::sizeMatch(const Tensor& t) const
    {
        size_t end = std::min(this->dim.size(), t.dim.size());
        for (size_t i = 1; i <= end; i++)
            if (this->dim[this->dim.size() - i] != t.dim[t.dim.size() - i])
                return false;
        for (size_t i = 0; i < this->dim.size() - end; i++) if (this->dim[i] != 1) return false;
        for (size_t i = 0; i < t.dim.size() - end; i++)     if (t.dim[i] != 1)     return false;
        
        return true;
    }


    /* ---------- Functions ---------- */

    inline Tensor empty_like(const Tensor& t)
    {
        return {t.dim};
    }

    template<float64(*fn)(float64)>
    inline Tensor forEach(const Tensor& t)
    {
        Tensor ret(t.dim);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = fn(t.b[i]);
        
        return ret;
    }

    inline Tensor forEach(const Tensor& t, std::function<double(double)> fn)
    {
        Tensor ret(t.dim);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = fn(t.b[i]);
        
        return ret;
    }

    template <void(*fn)(float64&, float64), float64(*init)(float64)>
    Tensor opAlongAxes(const Tensor& t)
    {
        d = t.dim.size() - d - 1;
        auto dim = t.dim;
        dim[d] = std::min((size_t)1, dim[d]);
        Tensor ret(dim);

        size_t tot = 1, stride = 1;
        for (size_t i = 0; i <= d; i++) tot *= dim[i];
        for (size_t i = d+1; i < dim.size(); i++) stride *= dim[i];
        
        for (size_t k = 0; k < tot; k++)
            for (size_t i = 0; i < stride; i++)
            {
                float64 sm = init(t.b[i + k * stride * t.dim[d]]);
                for (size_t j = 1; j < t.dim[d]; j++)
                    fn(sm, t.b[j * stride + i + k * stride * t.dim[d]]);
                
                ret.b[i + k * stride] = sm;
            }
        
        return ret;
    }

}

namespace std
{

    RedFish::Tensor sqrt(const RedFish::Tensor& t)
    {
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::sqrt(t.b[i]);
        
        return ret;
    }

    RedFish::Tensor exp(const RedFish::Tensor& t)
    {
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::exp(t.b[i]);
        
        return ret;
    }

    RedFish::Tensor log(const RedFish::Tensor& t)
    {
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::log(t.b[i]);
        
        return ret;
    }

    RedFish::Tensor pow(const RedFish::Tensor& t, RedFish::float64 power)
    {
        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::pow(t.b[i], power);
        
        return ret;
    }

    RedFish::Tensor pow(const RedFish::Tensor& t, const RedFish::Tensor& power)
    {
        if (!t.sizeMatch(power))
            throw std::length_error("Tensor sizes not matching in std::pow operation");

        RedFish::Tensor ret = RedFish::empty_like(t);
        for (size_t i = 0; i < t.getSize(); i++)
            ret.b[i] = std::pow(t.b[i], power.b[i]);

        return ret;
    }

}
