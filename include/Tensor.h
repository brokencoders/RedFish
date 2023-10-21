#pragma once 

#define DEBUG

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace Algebra {

#ifdef _WIN32
    typedef long double float64;
#else
    typedef double float64;
#endif

    class Tensor
    {
    private:
        
    public:
        Tensor(const std::vector<size_t>& dim);
        Tensor(const Tensor& t);                    // Copy Constructor
        ~Tensor();
    
        // Operetors
        Tensor operator+(const Tensor& t);
        Tensor operator+(const double val);
        Tensor& operator+=(const Tensor& t);
        Tensor& operator+=(const double val);
        Tensor operator-(const Tensor& t);
        Tensor operator-(const double val);
        Tensor& operator-=(const Tensor& t);
        Tensor& operator-=(const double val);
        Tensor operator*(const Tensor& t);
        Tensor operator*(const double val);
        Tensor& operator*=(const Tensor& t);
        Tensor& operator*=(const double val);
        Tensor operator/(const Tensor& t);
        Tensor operator/(const double val);
        Tensor& operator/=(const Tensor& t);
        Tensor& operator/=(const double val);

        float64& operator()(const size_t* index)
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
                n += index[i] * this->dim[i];
            
            return this->b.get()[n + index[this->dim.size() - 1]];
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

        void zero();
        bool sizeMatch(const Tensor& t);
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

    inline Tensor::Tensor(const Tensor& t)
    {
        this->dim = t.dim;
        this->size = t.size;
        this->b = std::make_unique<float64[]>(size);

        for (int i = 0; i < size; i++)
            this->b.get()[i] = t.b.get()[i];
    }
    
    inline Tensor::~Tensor() { }

    /* Operetors */
    inline Tensor Tensor::operator+(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in sum operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] + t.b.get()[i];

        return result;
    }

    inline Tensor Tensor::operator+(const double val)
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] + val;

        return result;
    }

    inline Tensor& Tensor::operator+=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b.get()[i] += t.b.get()[i];

        return *this;
    }

    inline Tensor& Tensor::operator+=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b.get()[i] += val;

        return *this;
    }

    inline Tensor Tensor::operator-(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in subtraction operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] - t.b.get()[i];

        return result;
    }

    inline Tensor Tensor::operator-(const double val)
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] - val;

        return result;
    }

    inline Tensor& Tensor::operator-=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b.get()[i] -= t.b.get()[i];

        return *this;
    }

    inline Tensor& Tensor::operator-=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b.get()[i] -= val;

        return *this;
    }

    inline Tensor Tensor::operator*(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in multiplication operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] * t.b.get()[i];

        return result;
    }

    inline Tensor Tensor::operator*(const double val)
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] * val;

        return result;
    }

    inline Tensor& Tensor::operator*=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b.get()[i] *= t.b.get()[i];

        return *this;
    }

    inline Tensor& Tensor::operator*=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b.get()[i] *= val;

        return *this;
    }

    inline Tensor Tensor::operator/(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in division operation");

        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] / t.b.get()[i];

        return result;
    }

    inline Tensor Tensor::operator/(const double val)
    {
        Tensor result(this->dim);
        for (size_t i = 0; i < size; i++)
            result.b.get()[i] = this->b.get()[i] / val;

        return result;
    }

    inline Tensor& Tensor::operator/=(const Tensor& t)
    {
        if (!sizeMatch(t))
            throw std::length_error("Matrix sizes not matching in sum operation");

        for (size_t i = 0; i < size; i++)
            this->b.get()[i] /= t.b.get()[i];

        return *this;
    }

    inline Tensor& Tensor::operator/=(const double val)
    {
        for (size_t i = 0; i < size; i++)
            this->b.get()[i] /= val;

        return *this;
    }

    // To-Do
    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) 
    {
        os << "Tensor" << std::endl << "Dim: ";

        for (int i = 0; i < t.dim.size(); i++)
        {
            os << " " << t.dim.at(i);
        }
        os << std::endl;

        return os;
    }

    inline void Tensor::zero()
    {
        for (int i = 0; i < size; i++)
            b.get()[i] = 0;
    }   

    inline bool Tensor::sizeMatch(const Tensor& t)
    {
        if (this->dim.size() != t.dim.size())
            return false;
        
        for (int i = 0; i < this->dim.size(); i++)
            if (this->dim[i] != t.dim[i])
                return false;
        return true;
    }
}


/* 

(3,2,4, ... to infinity)

val({ x, y, z})

index = ((((x*3)+y)*2)+z) -> yeeeeee 

[
    [
        [1,2,3,4],
        [1,2,4,1]
    ]
    [
        [1,2,3,4],
        [1,2,4,1]
    ]
    [
        [1,2,3,4],
        [1,2,4,1]
    ]
]

1,2,3,4,
1,2,4,1,
1,2,3,4,
1,2,4,1,
1,2,3,4,
1,2,4,1...
 */