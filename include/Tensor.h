#pragma once 

#include <vector>
#include <cstdlib>

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
        ~Tensor();
    private:
        float64* b;
        size_t size;
        std::vector<size_t> dim;
    };
    
    Tensor::Tensor(const std::vector<size_t>& dim)
        :dim(dim)
    {
        size_t size = 0;
        for (size_t i = 0; i < dim.size(); i++)
            size += dim[i];

        b = new float64[size];
    }
    
    Tensor::~Tensor()
    {
        if (b) std::free(b);
    }
    
}