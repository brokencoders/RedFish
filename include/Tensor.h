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
    bool broadcastable(const std::vector<size_t>&, const std::vector<size_t>&);

    class Tensor
    {
    public:

    class Index
    {
    public:
        Index(Tensor t1, Tensor t2)
            :sizeT1(1), sizeT2(1), sizeT3(1)
        {
            #ifdef CHECK_BOUNDS
            // Check if broadcast is possible
            if(!broadcastable(t1.shape, t2.shape))
                throw std::length_error("Brodcasting not possible!");
            #endif
            
            shapeT1 = t1.shape;
            shapeT2 = t2.shape;

            if (shapeT1.size() > shapeT2.size())
                for (int i = 0; shapeT1.size() != shapeT2.size(); i++)
                    shapeT2.insert(shapeT2.begin(), 1);
            else 
                for (int i = 0; shapeT1.size() != shapeT2.size(); i++)
                    shapeT1.insert(shapeT1.begin(), 1);

            shapeT3.reserve(shapeT1.size());
            for (int i = 0; i < shapeT1.size(); i++)
                shapeT3.push_back(std::max(shapeT1[i], shapeT2[i]));
            
            for (int i = 0; i < shapeT3.size(); i++)
            {
                sizeT1 *= shapeT1[i];
                sizeT2 *= shapeT2[i];
                sizeT3 *= shapeT3[i];
            }

            index1.insert(index1.begin(), shapeT1.size(), 0);
            index2.insert(index2.begin(), shapeT2.size(), 0);
        }

        void increment()
        {
            bool add1 = true, add2 = true;
            for (size_t i = index1.size()-1; (add1 || add2) && i != (size_t)-1; i--)
            {
                size_t index1_i = index1[i];
                if (add1)
                {
                    if (shapeT1[i] == shapeT3[i])
                        if (index1[i] + 1 < shapeT1[i])
                            index1[i]++,
                            add1 = false;
                        else
                            index1[i] = 0;
                    else if (index2[i] + 1 < shapeT2[i])
                        add1 = false;
                    else
                        add1 = true;
                }
                if (add2)
                {
                    if (shapeT2[i] == shapeT3[i])
                        if (index2[i] + 1 < shapeT2[i])
                            index2[i]++,
                            add2 = false;
                        else
                            index2[i] = 0;
                    else if (index1_i + 1 < shapeT1[i])
                        add2 = false;
                    else
                        add2 = true;
                }
            }
        }
        
        std::vector<size_t> shapeT1;
        std::vector<size_t> index1;
        std::vector<size_t> shapeT2;
        std::vector<size_t> index2;
        std::vector<size_t> shapeT3;
        
        size_t sizeT1;
        size_t sizeT2;
        size_t sizeT3;

        struct Iterator 
        {
            Iterator(size_t i0, size_t i1, size_t i2, Index* idx)
                : idx(idx) 
            {
                index[0] = i0;
                index[1] = i1;
                index[2] = i2;
            }

            const size_t* operator*() const { return index; }
            const size_t* operator->() { return index; }
            
            Iterator& operator++()
            { 
                idx->increment();
                index[2] ++;

                size_t offset1 = 0;
                size_t offset2 = 0;
                for (size_t i = 0; i < idx->shapeT3.size() - 1; i++)
                {
                    offset1 = (offset1 + idx->index1[i]) * idx->shapeT1[i + 1];
                    offset2 = (offset2 + idx->index2[i]) * idx->shapeT2[i + 1];
                }

                index[0] = offset1 + idx->index1.back();
                index[1] = offset2 + idx->index2.back();

                return *this; 
            }

            friend bool operator== (const Iterator& a, const Iterator& b) 
            { 
                return a.index[0] == b.index[0] && a.index[1] == b.index[1] && a.index[2] == b.index[2]; 
            };

            friend bool operator!= (const Iterator& a, const Iterator& b) 
            { 
                return a.index[0] != b.index[0] && a.index[1] != b.index[1] && a.index[2] != b.index[2];
            };

            size_t index[3];
            Index* idx;
        };

        Iterator begin() { return Iterator((size_t)0,(size_t)0,(size_t)0, this); }
        Iterator end()   { return Iterator(sizeT1, sizeT2, sizeT3, this); }
    };
        
    public:
        Tensor(const std::vector<size_t>& shape = {});
        Tensor(const size_t* shape, size_t len);
        Tensor(const std::vector<size_t>& shape, float64* buff, bool copy = true);
        Tensor(const Tensor& t);                    // Copy Constructor
        ~Tensor();

        Tensor& operator=(const Tensor& t);
        void resize(const std::vector<size_t>& shape);

        Tensor matmul(const Tensor& t) const;
        Tensor T();
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
        friend Tensor std::sqrt(const Tensor&);
        friend Tensor std::exp(const Tensor&);
        friend Tensor std::log(const Tensor&);
        friend Tensor std::pow(const Tensor&, RedFish::float64);
        friend Tensor std::pow(const Tensor&, const Tensor&);

        void zero();
        void ones();
        void rand();
        void rand(float64 start, float64 end);
        
        bool sizeMatch(const Tensor& t) const;

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

        for (int i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }

    inline Tensor& Tensor::operator=(const Tensor& t)
    {
        this->shape = t.shape;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (int i = 0; i < size; i++)
            this->b[i] = t.b[i];

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
        if (this->shape.size() == 0 || t.shape.size() == 0)
            throw std::length_error("Zero dimensional tensor in matmul !");
            
        Tensor mul({0});
        if(this->shape.size() == 1 && t.shape.size() == 1)
        { 
            if (this->shape[0] != t.shape[0])
                throw std::length_error("Size not matching in tensor matmul !");
            
            mul.resize({1});
            mul((size_t)0) = 0.;
            for (size_t i = 0; i < shape[0]; i++)
                mul((size_t)0) += b[i] * t(i);
        }
        else if (this->shape.size() == 2 && t.shape.size() == 2)
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
        else if (this->shape.size() == 1 && t.shape.size() == 2)
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
        else if (this->shape.size() == 2 && t.shape.size() == 1)
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
            if (this->shape.size() == 1)
            {
                if (this->colSize() != t.shape[t.shape.size()-2])
                    throw std::length_error("Size not matching in tensor matmul !");

                size_t batch_size = 1;
                for (size_t i = 0, end = t.shape.size()-2; i < end; i++)
                    batch_size *= t.shape[i];
                
                auto nsize = t.shape;
                nsize[nsize.size()-2] = 1;
                mul.resize(nsize);
                const auto thisp = this->b.get();
                for (size_t l = 0, t_cols = t.shape.back(); l < batch_size; l++)
                {
                    auto mulp = mul.b.get() + l*batch_size;
                    const auto tp = t.b.get() + l*batch_size;
                    for (size_t k = 0; k < t_cols; k++)
                        mulp[k] = thisp[0] * tp[k]; 
                    for (size_t i = 1; i < this->shape[0]; i++)
                        for (size_t k = 0, i_t_cols = i*t_cols; k < t_cols; k++)
                            mulp[k] += thisp[i] * tp[i_t_cols + k]; 
                }
            }
            else if (t.shape.size() == 1)
            {
                if (this->shape.back() != t.colSize())
                    throw std::length_error("Size not matching in tensor matmul !");

                size_t batch_size = 1;
                for (size_t i = 0, end = t.shape.size()-2; i < end; i++)
                    batch_size *= this->shape[i];
                
                auto nsize = this->shape;
                nsize[nsize.size()-2] = 1;
                mul.resize(nsize);

                const auto tp = t.b.get();
                for (size_t l = 0, this_rows = shape[shape.size()-2], this_cols = this->shape.back(); l < batch_size; l++)
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
                if (this->shape.back() != t.shape[t.shape.size()-2])
                    throw std::length_error("Size not matching in tensor matmul !");

                if (this->shape.size() > 2 && 
                    t.shape.size() > 2 && 
                    !broadcastable({this->shape.begin(), this->shape.end()-2}, {t.shape.begin(), t.shape.end()-2}))
                        throw std::length_error("Size not broadcastable in tensor matmul !");

                std::vector<size_t> nsize(std::max(this->shape.size(), t.shape.size()));
                std::vector<size_t> thissize(nsize.size() - this->shape.size(), 1);
                std::vector<size_t> tsize(nsize.size() - t.shape.size(), 1);
                thissize.insert(thissize.end(), this->shape.begin(), this->shape.end());
                tsize.insert(tsize.end(), t.shape.begin(), t.shape.end());

                nsize.back() = t.shape.back();
                nsize[nsize.size()-2] = shape[shape.size()-2];
                
                for (size_t i = 0; i < nsize.size()-2; i++)
                    nsize[i] = std::max(thissize[i], tsize[i]);
                
                mul.resize(nsize);

                size_t mul_cols      = nsize.back(),
                       mul_rows      = nsize[nsize.size()-2],
                       mul_mat_size  = mul_cols*mul_rows,
                       this_cols     = this->shape.back(),
                       this_rows     = this->shape[shape.size()-2],
                       this_mat_size = this_cols*this_rows,
                       t_cols        = t.shape.back(),
                       t_rows        = t.shape[t.shape.size()-2],
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

    inline Tensor Tensor::T()
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
        Tensor result({0});
        if (sizeMatch(t))
        {
            result.resize(this->shape);
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] + t.b[i];
        }
        else if (broadcastable(this->shape, t.shape))
        {
            // To-Do
        }
        else
            throw std::length_error("Tensor sizes not matching in sum operation");

        return result;
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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] += t.b[i];

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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in subtraction operation");

        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - t.b[i];

        return result;
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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] -= t.b[i];

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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in multiplication operation");

        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] * t.b[i];

        return result;
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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] *= t.b[i];

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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in division operation");

        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / t.b[i];

        return result;
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
        if (!sizeMatch(t))
            throw std::length_error("Tensor sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b[i] /= t.b[i];

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
            for (int i = 0; i < shape.size(); i++)
            {
                if (index[i] < 0 || index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (int i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    inline float64 Tensor::operator()(const size_t* index)  const 
    {
        #ifdef CHECK_BOUNDS 
            for (int i = 0; i < shape.size(); i++)
            {
                if (index[i] < 0 || index[i] >= shape[i])
                    throw new std::range_error("Out of bound in Tesnor () operetor");
            }
        #endif
        
        size_t n = 0;
        for (int i = 0; i < this->shape.size() - 1; i++)
            n = (n + index[i]) * this->shape[i + 1];
        
        return this->b[n + index[this->shape.size() - 1]];
    }

    inline float64 &Tensor::operator()(size_t x)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || shape.size() != 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 Tensor::operator()(size_t x) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || shape.size() != 1)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || shape.size() != 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * shape[1] + y];
    }

    inline float64 Tensor::operator()(size_t x, size_t y) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || shape.size() != 2)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[x * shape[1] + y];
    }

    inline float64 &Tensor::operator()(size_t x, size_t y, size_t z)
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || z < 0 || z >= shape[2] || shape.size() != 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * shape[1] + y) * shape[2] + z];
    }

    inline float64 Tensor::operator()(size_t x, size_t y, size_t z) const
    {
        #ifdef CHECK_BOUNDS 
            if (x < 0 || x >= shape[0] || y < 0 || y >= shape[1] || z < 0 || z >= shape[2] || shape.size() != 3)
                throw new std::range_error("Out of bound in Tesnor () operetor");
        #endif
        return this->b[(x * shape[1] + y) * shape[2] + z];
    }

    inline bool Tensor::operator==(const Tensor &t) const
    {
        if(!sizeMatch(t)) return false;
        
        for (int i = 0; i < size; i++)
            if(b[i] != t.b[i]) return false;
        
        return true;
    }

    inline void reprint(std::ostream& os, const Tensor& t, size_t depth, std::vector<size_t>& index)
    {
        if (depth == 0) { os << t(index.data()); return; }

        if (depth > 1)
        {
            os << "[\n";
            for (int i = 0; i < t.shape.size() - depth + 1; i++) os << "  ";
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
                for (int i = 0; i < t.shape.size() - depth + 1; i++) os << "  ";
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
            for (int i = 0; i < t.shape.size() - depth; i++) os << "  ";
        }
        os << "]";
    }

    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) 
    {
        os << "Tensor" << std::endl << "Shape: (";
        for (int i = 0; i < t.shape.size() - 1; i++)
            os << t.shape[i] << ", ";
        os << t.shape.back() << ")\n";

        std::vector<size_t> v;
        reprint(os, t, t.shape.size(), v);
        os << '\n';
        return os;
    }

    inline void Tensor::zero()
    {
        for (int i = 0; i < size; i++)
            b[i] = 0;
    }  

    inline void Tensor::ones()
    {
        for (int i = 0; i < size; i++) 
            b[i] = 1;
    }

    inline void Tensor::rand()
    {
        float64 inv = 1. / RAND_MAX;
        for (int i = 0; i < size; i++) 
            b[i] = std::rand() * inv;
    }

    inline void Tensor::rand(float64 start, float64 end)
    {
        float64 l = (end - start) / RAND_MAX;
        for (int i = 0; i < size; i++) 
            b[i] = std::rand() * l + start;
    }

    inline bool Tensor::sizeMatch(const Tensor& t) const
    {
        if (size != t.size) return false;
        size_t end = std::min(this->shape.size(), t.shape.size());
        for (size_t i = 1; i <= end; i++)
            if (this->shape[this->shape.size() - i] != t.shape[t.shape.size() - i])
                return false;
        for (size_t i = 0; i < this->shape.size() - end; i++) if (this->shape[i] != 1) return false;
        for (size_t i = 0; i < t.shape.size() - end; i++)     if (t.shape[i] != 1)     return false;
        
        return true;
    }


    /* ---------- Functions ---------- */

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
    Tensor opAlongAxes(const Tensor& t, size_t d, const float64 init_val)
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
    float64 opAlongAllAxes(const Tensor& t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
    }

}

namespace std
{

    RedFish::Tensor sqrt(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::sqrt>(t);
    }

    RedFish::Tensor exp(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::exp>(t);
    }

    RedFish::Tensor log(const RedFish::Tensor& t)
    {
        return RedFish::forEach<std::log>(t);
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
