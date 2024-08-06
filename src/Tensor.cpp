#include "Tensor.h"

#include <chrono>
#include <numeric>
#include "mkl.h"
#include "mkl_vsl.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "OpenCLManager.h"

namespace RedFish
{
    void matmul_gotoblas(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
    void matmul_left_T(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
    void transpose(float64* tp, const float64* thisp, const size_t rows, const size_t cols, const size_t lda, const size_t ldb);
    void conv_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation);
    void conv_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation);
    void cross_correlation_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation);
    void cross_correlation_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation);
    void cross_correlation_3d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple3d t_size, Tuple3d kernel_size, Tuple3d stride, Tuple3d dilation);
    static void copy_2d(float64*, float64*, Tuple2d, size_t, size_t);
    template <auto fn>
    static void for_(const size_t size[], const size_t ld[], std::vector<size_t>& index, size_t height, float64* b, size_t depth=0, size_t off=0);
    template <auto fn, typename... Args>
    static void broadcast_op(float64 *dst, const float64 *src1, const float64 *src2,
                             const size_t *shape, const size_t *shape1, const size_t *shape2,
                             size_t depth,
                             size_t foff, size_t foff1, size_t foff2,
                             Args... args);

    std::random_device Tensor::rd;
    std::default_random_engine Tensor::gen(rd());

    static float64* alloc(size_t size)
    {
        if (!size) return nullptr;
        else return (float64 *)mkl_malloc( size*sizeof( float64 ), 64 );
    }
    
    static void dealloc(float64*& buff)
    {
        if (buff) mkl_free(buff);
        buff = nullptr;
    }

    /* 
     *      CONSTRUCTORS
     */

    /**
     * @brief Construct a new uninitialized Tensor object with given shape
     * 
     * @param shape 
     */
    Tensor::Tensor(const std::vector<size_t>& shape, bool onCPU)
        : shape(shape), size(1), onCPU(onCPU), b(nullptr), buffer(0)
    {
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (onCPU)
            b = alloc(size);
        else
            buffer = OpenCLManager::createBuffer<float64>(size);
    }

    /**
     * @brief  Construct a new uninitialized Tensor object with given shape
     * 
     * @param shape c-like array with tensor shape
     * @param len   shape array length
     */
    Tensor::Tensor(const size_t* shape, size_t len)
        : shape(shape, shape + len), size(1), onCPU(true), buffer(0)
    {
        for (size_t i = 0; i < len; i++)
            size *= shape[i];

        b = alloc(size);
    }

    /**
     * @brief Construct a new Tensor object with given shape from a buffer, optionally copying it
     * 
     * @param shape 
     * @param buff 
     * @param copy whether to copy the buffer to a new one or to take buff as the internal memory 
     */
    Tensor::Tensor(const std::vector<size_t>& shape, float64 *buff)
        : shape(shape), size(1), onCPU(true), b(nullptr), buffer(0)
    {
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        b = alloc(size);
        std::copy(buff, buff + size, b);
    }

    /**
     * @brief Copy construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(const Tensor &t)
        : shape(t.shape), size(t.size), onCPU(t.onCPU), b(nullptr), buffer(0)
    {
        if (onCPU)
        {
            this->b = alloc(size);
            std::copy(t.b, t.b + size, b);
        }
        else
        {
            buffer = OpenCLManager::createBuffer<float64>(size);
            OpenCLManager::copyBuffer<float64>(t.buffer, this->buffer, size);
        }

    }

    /**
     * @brief Move construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(Tensor &&t)
        : shape(t.shape), size(t.size), onCPU(t.onCPU), b(t.b), buffer(t.buffer)
    {
        t.shape  = {0};
        t.size   = 0;
        t.onCPU  = true;
        t.b      = nullptr;
        t.buffer = 0;
    }

    /**
     * @brief Construct a new Tensor object from the shape and data as a list
     * 
     * @param shape of the Tensor t be created 
     * @param data as an initializer list 
     */
    Tensor::Tensor(const std::vector<size_t>& shape, std::initializer_list<float64> data)
        : shape(shape), size(1), onCPU(true), buffer(0)
    {
        if (shape.size() != 0 || data.size() == 1)
        {
            for (size_t i = 0; i < shape.size(); i++)
                size *= shape[i];
        }
        else
        {
            this->shape.push_back(data.size());
            size = data.size();
        }

        if (size != data.size())
            throw std::length_error("Invalid shape for given data in Tensor(const std::vector<size_t>&, std::initializer_list<float64>);");

        b = alloc(size);
        std::copy(data.begin(), data.end(), b);
    }

    /**
     * @brief Construct a new Tensor from an input file stream already opened
     * 
     * @param file std::ifstream& 
     */
    Tensor::Tensor(std::ifstream& file)
        : size(1), onCPU(true), buffer(0)
    {
        const std::string name = "Tensor";
        char rname[sizeof("Tensor")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Tensor(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        uint64_t shape_size = 0;
        file.read((char*)&shape_size, sizeof(shape_size));
        shape.reserve(shape_size);
        for (size_t i = 0; i < shape_size; i++)
        {
            uint64_t shape_size = 0;
            file.read((char*)&shape_size, sizeof(shape_size));
            shape.push_back(shape_size);
            this->size *= shape_size;
        }

        b = alloc(this->size);
        file.read((char*)b, this->size * sizeof(float64));
    }
    
    /* 
     *      CONSTRUCTORS (end)
     */

    Tensor::~Tensor()
    {
        if (onCPU)
            dealloc(b);
        else
            OpenCLManager::destroyBuffer(buffer);
    }

    void Tensor::toDevice()
    {
        if (!onCPU) return;

        buffer = OpenCLManager::createBuffer<float64>(size);
        OpenCLManager::loadWriteBuffer<float64>(buffer, size, b);

        onCPU = false;
        dealloc(b);
    }

    void Tensor::fromDevice()
    {
        if (onCPU) return;

        b = alloc(size);
        OpenCLManager::loadReadBuffer<float64>(buffer, size, b);

        onCPU = true;
        OpenCLManager::destroyBuffer(buffer);
    }

    /* 
     *      ASSIGNMENT OPERATORS
     */
     
    /**
     * @brief Assignment operator
     * 
     * @param t Tensor to be coppied 
     * @return Tensor& this
     */
    Tensor& Tensor::operator=(const Tensor &t)
    {
        if (onCPU) dealloc(b);
        else OpenCLManager::destroyBuffer(buffer);
        this->shape  = t.shape;
        this->size   = t.size;
        this->onCPU  = t.onCPU;

        if (onCPU)
        {
            this->b = alloc(size);
            std::copy(t.b, t.b + size, b);
        }
        else
        {
            buffer = OpenCLManager::createBuffer<float64>(size);
            OpenCLManager::copyBuffer<float64>(t.buffer, this->buffer, size);
        }

        return *this;
    }

    /**
     * @brief Move assignment operator
     * 
     * @param t 
     * @return Tensor& this
     */
    Tensor& Tensor::operator=(Tensor &&t)
    {
        if (onCPU) dealloc(b);
        else OpenCLManager::destroyBuffer(buffer);
        this->shape  = t.shape;
        this->size   = t.size;
        this->onCPU  = t.onCPU;
        this->b      = t.b;
        this->buffer = t.buffer;
        t.shape  = {0};
        t.size   =  0;
        t.onCPU  = true;
        t.b      = nullptr;

        return *this;
    }

    /* 
     *      ASSIGNMENT OPERATORS (end)
     */

    /* 
     *      OPERATORS
     */

    Tensor Tensor::operator+(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_ADD, Kernel::T_TENSOR_ADD_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in sum operation");
    }

    Tensor Tensor::operator+(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] + val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_ADD, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor operator+(const float64 val, const Tensor &t)
    {
        return t + val;
    }

    Tensor &Tensor::operator+=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_broadcast_assign<fn, Kernel::T_TENSOR_ADD>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    Tensor &Tensor::operator+=(const float64 val)
    {
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                this->b[i] += val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_ADD, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(buffer));

        return *this;
    }

    Tensor Tensor::operator-(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_SUB, Kernel::T_TENSOR_SUB_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in subtraction operation");
    }

    Tensor Tensor::operator-(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] - val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_SUB, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor Tensor::operator-() const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = -this->b[i];
        else
            OpenCLManager::execute(Kernel::T_MINUS, {size}, OpenCLManager::getBuffer(buffer), OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor operator-(const float64 val, const Tensor &t)
    {
        Tensor ret = t.empty_like(t);
        if (t.onCPU)
            for (size_t i = 0; i < t.size; i++)
                ret.b[i] = val - t.b[i];
        else
            OpenCLManager::execute(Kernel::T_SCALAR_TENSOR_SUB, {t.size}, OpenCLManager::getBuffer(t.buffer), val, OpenCLManager::getBuffer(ret.buffer));

        return ret;
    }

    Tensor &Tensor::operator-=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        ew_or_broadcast_assign<fn, Kernel::T_TENSOR_SUB>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    Tensor &Tensor::operator-=(const float64 val)
    {
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                this->b[i] -= val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_SUB, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(buffer));

        return *this;
    }

    Tensor Tensor::operator*(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  *  n2; };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_MUL, Kernel::T_TENSOR_MUL_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in multiplication operation");
    }

    Tensor Tensor::operator*(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] * val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_MUL, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor operator*(const float64 val, const Tensor &t)
    {
        return t * val;
    }

    Tensor &Tensor::operator*=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  *  n2; };
        ew_or_broadcast_assign<fn, Kernel::T_TENSOR_MUL>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    Tensor &Tensor::operator*=(const float64 val)
    {
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                this->b[i] *= val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_MUL, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(buffer));

        return *this;
    }

    Tensor Tensor::operator/(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  /  n2; };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_DIV, Kernel::T_TENSOR_DIV_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in division operation");
    }

    Tensor Tensor::operator/(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] / val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_DIV, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor operator/(const float64 val, const Tensor &t)
    {
        Tensor ret = t.empty_like(t);
        if (t.onCPU)
            for (size_t i = 0; i < t.size; i++)
                ret.b[i] = val / t.b[i];
        else
            OpenCLManager::execute(Kernel::T_SCALAR_TENSOR_DIV, {t.size}, OpenCLManager::getBuffer(t.buffer), val, OpenCLManager::getBuffer(ret.buffer));

        return ret;
    }

    Tensor &Tensor::operator/=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_broadcast_assign<fn, Kernel::T_TENSOR_DIV>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }

    Tensor &Tensor::operator/=(const float64 val)
    {
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                this->b[i] /= val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_DIV, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(buffer));

        return *this;
    }
    
    /* 
     *      OPERATORS (end)
     */

    /* 
     *      COMPARISON OPERATORS
     */

    Tensor Tensor::operator==(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 == n2); };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_EQUALS, Kernel::T_TENSOR_EQUALS_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in equality operation");
    }

    Tensor Tensor::operator==(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] == val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_EQUALS, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor Tensor::operator<=(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 <= n2); };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_LT_EQUALS, Kernel::T_TENSOR_LT_EQUALS_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in less then or equal operation");
    }

    Tensor Tensor::operator<=(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] <= val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_LT_EQUALS, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor Tensor::operator>=(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 >= n2); };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_GT_EQUALS, Kernel::T_TENSOR_GT_EQUALS_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in greater then or equal operation");
    }

    Tensor Tensor::operator>=(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] >= val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_GT_EQUALS, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor Tensor::operator<(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 < n2); };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_LT, Kernel::T_TENSOR_LT_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in less then operation");
    }

    Tensor Tensor::operator<(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] < val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_LT, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    Tensor Tensor::operator>(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 > n2); };
        return ew_or_broadcast<fn, Kernel::T_TENSOR_GT, Kernel::T_TENSOR_GT_BRODCAST_N0B1N0>(*this, t, "Tensor sizes not matching in greater then operation");
    }

    Tensor Tensor::operator>(const float64 val) const
    {
        Tensor result(this->shape);
        if (onCPU)
            for (size_t i = 0; i < size; i++)
                result.b[i] = this->b[i] > val;
        else
            OpenCLManager::execute(Kernel::T_SCALAR_GT, {size}, OpenCLManager::getBuffer(buffer), val, OpenCLManager::getBuffer(result.buffer));

        return result;
    }

    /* 
     *      COMPARISON OPERATORS (end)
     */

    float64 &Tensor::operator()(const std::vector<size_t> &index)
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

    float64 Tensor::operator()(const std::vector<size_t> &index) const
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

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,l) -> index: (x,...,x)
     * @return DirectTensorView shape: (l)
     */
    DirectTensorView Tensor::getRow(const std::vector<size_t> &index) 
    { 
        return sliceLastNDims<1>(index);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,l) -> index: (x,...,x)
     * @return DirectTensorView shape: (l)
     */
    const DirectTensorView Tensor::getRow(const std::vector<size_t> &index) const 
    { 
        return sliceLastNDims<1>(index);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,h,w) -> index: (x,...,x)
     * @return DirectTensorView shape: (h, w)
     */
    DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index)
    {
        return sliceLastNDims<2>(index);
    }
    
    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,h,w) -> index: (x,...,x)
     * @return DirectTensorView shape: (h, w)
     */
    const DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index) const
    {
        return sliceLastNDims<2>(index);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,d1,..,dN) -> index: (x,...,x)
     * @param N
     * @return DirectTensorView shape: (d1,..,dN)
     */
    DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index, size_t N)
    {
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end() - index.size() + i - N + 1);
            if (index[i] >= *(shape.end() - index.size() + i - N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i + 1 < N; i++)
            off *= *(shape.end() + i - N + 1);

        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end() - N + i);

        return DirectTensorView({new_shape, new_shape + N}, b + off);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,d1,..,dN) -> index: (x,...,x)
     * @param N
     * @return DirectTensorView shape: (d1,..,dN)
     */
    const DirectTensorView Tensor::sliceLastNDims(const std::vector<size_t> &index, size_t N) const
    {
        if (index.size() + N > shape.size())
            throw new std::range_error("Out of bound in Tensor sliceLastNDims()");

        size_t new_shape[N], off = 0;
        for (size_t i = 0; i < index.size(); i++)
        {
            off = (off + index[i]) * *(shape.end() - index.size() + i - N + 1);
            if (index[i] >= *(shape.end() - index.size() + i - N))
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = 0; i + 1 < N; i++)
            off *= *(shape.end() + i - N + 1);

        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end() - N + i);

        return DirectTensorView({new_shape, new_shape + N}, b + off);
    }

    /**
     * @brief set new size from shape
     * 
     * @param new_shape 
     */
    void Tensor::resize(const std::vector<size_t> &new_shape)
    {
        this->shape = new_shape;
        size = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
            size *= new_shape[i];

        if (onCPU)
        {
            dealloc(b);
            b = alloc(size);
        }
        else
        {
            OpenCLManager::destroyBuffer(buffer);
            buffer = OpenCLManager::createBuffer<float64>(size);
        }
    }

    /**
     * @brief set a new shape for the tensor while keeping the same values (memory)
     * 
     * @param new_shape 
     */
    void Tensor::reshape(const std::vector<size_t> &new_shape)
    {
        size_t new_size = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
            new_size *= new_shape[i];

        if (new_size != size)
            throw std::length_error("Invalid new shape in Tensor reshape");

        this->shape = new_shape;
    }

    DirectTensorView Tensor::asShape(const std::vector<size_t> &new_shape)
    {
        size_t new_size = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
            new_size *= new_shape[i];

        if (new_size != size)
            throw std::length_error("Invalid new shape in Tensor asShape");

        return DirectTensorView(new_shape, b);
    }

    const DirectTensorView Tensor::asShape(const std::vector<size_t> &new_shape) const
    {
        size_t new_size = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
            new_size *= new_shape[i];

        if (new_size != size)
            throw std::length_error("Invalid new shape in Tensor asShape");

        return DirectTensorView(new_shape, b);
    }

    DirectTensorView Tensor::asShapeOneInsert(size_t where, size_t count)
    {
        auto new_shape = this->shape;
        if (where <= new_shape.size()) new_shape.insert(new_shape.end() - where, 1);
        else new_shape.insert(new_shape.begin(), where - new_shape.size() + 1, 1);

        return DirectTensorView(new_shape, b);
    }

    const DirectTensorView Tensor::asShapeOneInsert(size_t where, size_t count) const
    {
        auto new_shape = this->shape;
        if (where <= new_shape.size()) new_shape.insert(new_shape.end() - where, 1);
        else new_shape.insert(new_shape.begin(), where - new_shape.size() + 1, 1);

        return DirectTensorView(new_shape, b);
    }

    /**
     * @brief Get transposed over last two dimensions of this tensor
     * 
     * @return Tensor 
     */
    Tensor Tensor::T() const
    {
        Tensor t;

        if (shape.size() < 2)
        {
            t = *this;
            if (shape.size() == 1)
                t.shape.push_back(1);
        }
        else
        {
            t.resize(shape);
            size_t end = 1;
            const size_t cols = t.shape.back();
            const size_t rows = t.shape.end()[-2];
            std::swap(t.shape.back(), t.shape.end()[-2]);

            for (size_t i = 0; i < (int64_t)t.shape.size() - 2; i++)
                end *= t.shape[i];

            if (onCPU)
                mkl_domatcopy_batch_strided('R', 'N', rows, cols, 1, this->b, cols, rows*cols, t.b, rows, rows*cols, end);
                /* for (size_t i = 0, stride = rows * cols; i < end; i++)
                    transpose(t.b + i * stride, this->b + i * stride, rows, cols, cols, rows); */
            else
                for (size_t i = 0, stride = rows * cols; i < end; i++)
                    OpenCLManager::execute(Kernel::T_TRANSPOSE, std::array<size_t, 2>({rows, cols}),
                                           rows, cols, OpenCLManager::getBuffer(buffer), OpenCLManager::getBuffer(t.buffer));
        }

        return t;
    }


    /**
     * @brief Transpose matrix along dimension d1 and d2 (To-Do)
     * 
     * @param d1 traspose dimension 
     * @param d2 traspose dimension
     * @return Tensor 
     */
    Tensor Tensor::T(size_t d1, size_t d2)
    {
        if (d1 == d2 || d1 >= shape.size() || d2 >= shape.size())
            throw std::range_error("invalid dimensions in Tensor transposition T()");

        if (d1 < d2)
            std::swap(d1, d2);

        Tensor t(shape);
        d1 = t.shape.size() - 1 - d1, d2 = t.shape.size() - 1 - d2;
        size_t rows = t.shape[d1], cols = t.shape[d2], end = 1, stride = 1, step = 1;
        std::swap(t.shape[d1], t.shape[d2]);

        for (size_t i = 0; i < d1; i++)
            end *= t.shape[i];
        for (size_t i = d1 + 1; i < d2; i++)
            step *= t.shape[i];
        for (size_t i = d2 + 1; i < t.shape.size(); i++)
            stride *= t.shape[i];

        // To-Do

        return t;
    }

    /**
     * @brief Set Tensor with all Zero
     * 
     */
    void Tensor::zero()
    {
        if(onCPU)
            std::fill(b, b + size, 0.);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, 0.);
    }

    /**
     * @brief Set tensor with all One
     * 
     */
    void Tensor::ones()
    {
        if(onCPU)
            std::fill(b, b + size, 1.);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, 1.);
    }

    /**
     * @brief Set all tensor elements to val
     * 
     * @param val
     */
    void Tensor::constant(float64 val)
    {
        if(onCPU)
            std::fill(b, b + size, val);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, val);
    }

    void Tensor::linspace(float64 start, float64 stop)
    {
        if(onCPU)
        {
            size_t ssize = shape.size() ? size / shape.back() : 1;
            size_t last  = shape.size() ? shape.back() : 1;
            float64 inc  = (stop - start) / (last-1);
            float64 base = start;
            for (size_t i = 0; i < ssize; i++, base = start)
            for (size_t j = 0; j < last; j++, base += inc)
                b[i*last + j] = base;
        }
        else
            {/* To-Do */}
    }

    /**
     * @brief Set Tensor values to a random uniform value between a and b 
     * 
     * @param a lower value 
     * @param b upper value 
     */
    void Tensor::randUniform(float64 a, float64 b)
    {
        std::uniform_real_distribution<> dis(a, b);
        for (size_t i = 0; i < size; i++)
            this->b[i] = dis(gen);
    }

    /**
     * @brief Set Tensor values randomly according to a gaussian distribution
     * 
     * @param mean 
     * @param std standard deviation
     */
    void Tensor::randNormal(float64 mean, float64 std)
    {
        std::normal_distribution<double> distribution(mean, std);
        for (size_t i = 0; i < size; i++)
            b[i] = distribution(gen);
    }

    /**
     * @brief Square and sum all elements of a tensor
     * 
     * @return float64 
     */
    float64 Tensor::squareSum() const
    {
        constexpr auto fn = [](float64 &sq_sum, float64 n)
        {  sq_sum += n  *  n; };
        return full_reduction<fn>(*this, 0.);
    }

    /**
     * @brief Square and sum all elements along the d dimension 
     * 
     * @param dimension
     * @return Tensor 
     */
    Tensor Tensor::squareSum(size_t d, bool collapse) const
    {
        constexpr auto fn = [](float64 &sq_sum, float64 n)
        {  sq_sum += n  *  n; };
        return axes_reduction<fn>(*this, d, 0., collapse);
    }

    /**
     * @brief Get max Tensor value
     * 
     * @return float64 
     */
    float64 Tensor::max() const
    {
        constexpr auto fn = [](float64 &max, float64 n)
        {if (max < n) max = n; };
        return full_reduction<fn>(*this, -std::numeric_limits<float64>::infinity());
    }

    /**
     * @brief Get max Tensor value along the d dimension 
     * 
     * @param dimension
     * @return Tensor 
     */
    Tensor Tensor::max(size_t d, bool collapse) const
    {
        constexpr auto fn = [](float64 &max, float64 n)
        {if (max < n) max = n; };
        return axes_reduction<fn>(*this, d, -std::numeric_limits<float64>::infinity(), collapse);
    }

    /**
     * @brief Get minimum Tensor value
     * 
     * @return float64 
     */
    float64 Tensor::min() const
    {
        constexpr auto fn = [](float64 &min, float64 n)
        {if (min > n) min = n; };
        return full_reduction<fn>(*this, std::numeric_limits<float64>::infinity());
    }

    /**
     * @brief Get minimum Tensor value along dimension
     * 
     * @param dimension
     * @return float64 
     */
    Tensor Tensor::min(size_t d, bool collapse) const
    {
        constexpr auto fn = [](float64 &min, float64 n)
        {if (min > n) min = n; };
        return axes_reduction<fn>(*this, d, std::numeric_limits<float64>::infinity(), collapse);
    }

    /**
     * @brief Sum all elements of a Tensor
     * 
     * @return float64 
     */
    float64 Tensor::sum() const
    {
        constexpr auto fn = [](float64 &sum, float64 n)
        {  sum += n; };
        return full_reduction<fn>(*this, 0.);
    }

    /**
     * @brief Sum all elements of a Tensor along the d dimension 
     * 
     * @param d 
     * @return Tensor 
     */
    Tensor Tensor::sum(size_t d, bool collapse) const
    {
        constexpr auto fn = [](float64 &sum, float64 n)
        {  sum += n; };
        return axes_reduction<fn>(*this, d, 0., collapse);
    }

    Tensor Tensor::shift(size_t d, int dir, const float64 fill) const
    {
        Tensor result = empty_like(*this);
        
        if (d < shape.size())
        {
            d = shape.size() - d - 1;
            size_t tot = 1, off = 1;
            for (size_t i = 0; i < d; i++) tot *= shape[i];
            for (size_t i = d+1; i < shape.size(); i++) off *= shape[i];

            for (size_t i = 0; i < size; i += off*shape[d])
                for (int64_t j = 0, l = -dir; j < shape[d]; j++, l++)
                {
                    if (l >= 0 && l < shape[d])
                        for (size_t k = 0; k < off; k++)
                            result.b[i + j*off + k] = b[i + l*off + k];
                    else
                        for (size_t k = 0; k < off; k++)
                            result.b[i + j*off + k] = fill;
                }
        }
        else for (size_t i = 0; i < size; i++) result.b[i] = b[i];        

        return result;
    }

    Tensor Tensor::roundShift(size_t d, int dir) const
    {
        Tensor result = empty_like(*this);
        
        if (d < shape.size())
        {
            d = shape.size() - d - 1;
            size_t tot = 1, off = 1;
            for (size_t i = 0; i < d; i++) tot *= shape[i];
            for (size_t i = d+1; i < shape.size(); i++) off *= shape[i];

            dir = -dir;
            while (dir < 0) dir += shape[d];
            for (size_t i = 0; i < size; i += off*shape[d])
                for (size_t j = 0, l = dir % shape[d]; j < shape[d]; j++, l = (l+1) % shape[d])
                    for (size_t k = 0; k < off; k++)
                        result.b[i + j*off + k] = b[i + l*off + k];
        }
        else for (size_t i = 0; i < size; i++) result.b[i] = b[i];        

        return result;
    }

    static struct BCJobs {
        static const int poolsize = 1;
        std::future<void> jobs[poolsize];
        size_t first_job = 0;
        BCJobs() { for (size_t i = 0; i < poolsize; i++) jobs[i] = std::async([](){}); }
        void add_job(const std::function<void(size_t,size_t,size_t,size_t)>& operation, size_t Aoff, size_t Boff, size_t Coff){
            jobs[first_job].wait();
            jobs[first_job] = std::async(operation, Aoff, Boff, Coff, first_job);
            first_job = (first_job + 1) % poolsize;
        }
        void wait() { for (size_t i = 0; i < poolsize; i++) jobs[i].wait(); }
    } broadcast_jobs;

    inline static void broadcast_operation_async(const size_t* Ashape, const size_t* Bshape, const size_t* Cshape,
                                           size_t depth, const std::function<void(size_t,size_t,size_t,size_t)>& operation,
                                           size_t Aoff = 0, size_t Boff = 0, size_t Coff = 0)
    {
        if (depth)
        {
            size_t bdc1 = (*Ashape == *Cshape) * ((size_t)-1);
            size_t bdc2 = (*Bshape == *Cshape) * ((size_t)-1);
            for (size_t i = 0; i < *Cshape; i++)
                broadcast_operation_async(
                    Ashape + 1, Bshape + 1, Cshape + 1,
                    depth - 1, operation,
                    Aoff * *Ashape + (i & bdc1),
                    Boff * *Bshape + (i & bdc2),
                    Coff * *Cshape + i);
        }
        else
            broadcast_jobs.add_job(operation, Aoff, Boff, Coff);
            /* operation(Aoff, Boff, Coff); */
    }

    inline static void broadcast_operation(const size_t* Ashape, const size_t* Bshape, const size_t* Cshape,
                                           size_t depth, const std::function<void(size_t,size_t,size_t)>& operation,
                                           size_t Aoff = 0, size_t Boff = 0, size_t Coff = 0)
    {
        if (depth)
        {
            size_t bdc1 = (*Ashape == *Cshape) * ((size_t)-1);
            size_t bdc2 = (*Bshape == *Cshape) * ((size_t)-1);
            for (size_t i = 0; i < *Cshape; i++)
                broadcast_operation(
                    Ashape + 1, Bshape + 1, Cshape + 1,
                    depth - 1, operation,
                    Aoff * *Ashape + (i & bdc1),
                    Boff * *Bshape + (i & bdc2),
                    Coff * *Cshape + i);
        }
        else
            operation(Aoff, Boff, Coff);
    }

static long long ttime = 0;
void print_ttime() { std::cout << "dgemm time: " << (float)ttime * 1e-9 << "s\n"; }

    /**
     * @brief Matrix Multipication 
     * 
     * @param t 
     * @param trsp wheter one of the two matrices has to be transopsed before multiplication
     * @return Tensor 
     */
    Tensor Tensor::matmul(const Tensor &t, const Transpose trsp) const
    {
        auto begin = std::chrono::high_resolution_clock::now();
        Tensor result;
        auto Ashape = shape, Bshape = t.shape;
        while (Ashape.size() < 2 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 2 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        size_t Acols = Ashape.back(); Ashape.pop_back();
        size_t Arows = Ashape.back(); Ashape.pop_back();
        size_t Bcols = Bshape.back(); Bshape.pop_back();
        size_t Brows = Bshape.back(); Bshape.pop_back();

        size_t Crows = trsp == LEFT  ? Acols : Arows;
        size_t Ccols = trsp == RIGHT ? Brows : Bcols;
        size_t Msize = trsp == LEFT  ? Arows : Acols;
        size_t Asize = 1, Bsize = 1;
        for (auto d : Ashape) Asize *= d;
        for (auto d : Bshape) Bsize *= d;

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.push_back(Crows);
            Ashape.push_back(Ccols);
            result.resize(Ashape);
            cblas_dgemm_batch_strided(CblasRowMajor, trsp == LEFT ? CblasTrans : CblasNoTrans, trsp == RIGHT ? CblasTrans : CblasNoTrans,
                                      Crows, Ccols, Msize, 1, b, Acols, Arows*Acols, t.b, Bcols, Brows*Bcols, 0, result.b, Ccols, Crows*Ccols, Asize);
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size());
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            
            Cshape.push_back(Crows);
            Cshape.push_back(Ccols);

            result.resize(Cshape);

            auto mult = [tr1 = trsp == LEFT ? CblasTrans : CblasNoTrans, tr2 = trsp == RIGHT ? CblasTrans : CblasNoTrans,
                         Acols, Bcols, Crows, Ccols, A = b, B = t.b, C = result.b, Amsize=Arows*Acols, Bmsize=Brows*Bcols, Cmsize=Crows*Ccols, Msize]
                         (size_t Aoff, size_t Boff, size_t Coff) {
                cblas_dgemm(CblasRowMajor, tr1, tr2, Crows, Ccols, Msize, 1, A + Aoff*Amsize, Acols, B + Boff*Bmsize, Bcols, 0, C + Coff*Cmsize, Ccols);
            };

            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), mult);
        }
        else
            throw std::length_error("Shapes not matching in Tensor matmul");

        ttime += (std::chrono::high_resolution_clock::now() - begin).count();
        return result;
    }

    /**
     * @brief Matrix Multiplication 
     * 
     * @param t1 
     * @param t2 
     * @param transpose wheter one of the two matrices has to be transopsed before multiplication
     * @return Tensor 
     */
    Tensor matmul(const Tensor &t1, const Tensor &t2, const Transpose transpose = NONE)
    {
        return t1.matmul(t2, transpose);
    }
    
static long long ctime = 0;
void print_ctime() { std::cout << "conv time: " << (float)ctime * 1e-9 << "s\n"; }

    /**
     * @brief Cross Correlation 1d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm Padding Mode 
     * @return Tensor 
     */
    Tensor Tensor::correlation1d(const Tensor &kernel, size_t padding, size_t stride, size_t dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 1 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 1 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        size_t len  = Ashape.back(); Ashape.pop_back();
        size_t klen = Bshape.back(); Bshape.pop_back();
        size_t olen = len + 1 + 2*padding - klen;
        size_t olen_clamped = len+ 1 + 2*std::min(padding, klen-1) - klen;
        size_t pad_offset = padding > klen - 1 ? padding + 1 - klen : 0;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start = -std::min(padding, klen-1);
        VSLCorrTaskPtr task;
        vsldCorrNewTask1D(&task, VSL_CORR_MODE_AUTO, klen, len, olen_clamped);
        vslCorrSetStart(task, &start);

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.push_back(olen);
            result.resize(Ashape);

            auto begin = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t j = 0; j < pad_offset; j++)
                    result.b[i*olen + j] = result.b[i*olen + j + olen_clamped + pad_offset] = 0;

                vsldCorrExec1D(task, kernel.b + i*klen, 1, b + i*len, 1, result.b + i*olen + pad_offset, 1);
            }
            ctime += (std::chrono::high_resolution_clock::now() - begin).count();
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 1);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.back() = olen;

            result.resize(Cshape);

            auto in = b;
            auto corr = [task, kernel = kernel.b, in, result = result.b, len, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t j = 0; j < pad_offset; j++)
                    result[Coff*olen + j] = result[Coff*olen + j + olen_clamped + pad_offset] = 0;

                vsldCorrExec1D(task, kernel + Boff*klen, 1, in + Aoff*len, 1, result + Coff*olen + pad_offset, 1);
            };

            auto begin = std::chrono::high_resolution_clock::now();
            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), corr);
            /* broadcast_jobs.wait(); */
            ctime += (std::chrono::high_resolution_clock::now() - begin).count();
        }

        vslCorrDeleteTask(&task);

        return result;
    }

    /**
     * @brief Cross Correlation 2d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm Padding Mode
     * @return Tensor 
     */
    Tensor Tensor::correlation2d(const Tensor &kernel, Tuple2d padding, Tuple2d stride, Tuple2d dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 2 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 2 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        long long ilen[3], klen[3], olen[3], olen_clamped[2];
        ilen[1] = Ashape.back(); Ashape.pop_back();
        ilen[0] = Ashape.back(); Ashape.pop_back();
        klen[1] = Bshape.back(); Bshape.pop_back();
        klen[0] = Bshape.back(); Bshape.pop_back();
        olen[1] = ilen[1] + 1 + 2*padding.x - klen[1];
        olen[0] = ilen[0] + 1 + 2*padding.y - klen[0];
        olen_clamped[1] = ilen[1] + 1 + 2*std::min<long long>(padding.x, klen[1]-1) - klen[1];
        olen_clamped[0] = ilen[0] + 1 + 2*std::min<long long>(padding.y, klen[0]-1) - klen[0];
        size_t pad_offset[2] = {padding.y > klen[0] - 1 ? padding.y + 1 - klen[0] : 0,
                                padding.x > klen[1] - 1 ? padding.x + 1 - klen[1] : 0};
        size_t isize = ilen[0]*ilen[1];
        size_t ksize = klen[0]*klen[1];
        size_t osize = olen[0]*olen[1];
        ilen[2] = klen[2] = olen[2] = 1;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start[2] = {-std::min<long long>(padding.y, klen[0]-1),
                              -std::min<long long>(padding.x, klen[1]-1)};
        VSLCorrTaskPtr task;
        vsldCorrNewTask(&task, VSL_CORR_MODE_DIRECT, 2, klen, ilen, olen_clamped);
        vslCorrSetStart(task, start);

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.insert(Ashape.end(), olen, olen + 2);
            result.resize(Ashape);
            
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t k = 0; k < pad_offset[0]; k++)
                for (size_t j = 0; j < olen[1]; j++)
                    result.b[i*osize + k*olen[1] + j] = result.b[i*osize + (k + olen_clamped[0] + pad_offset[0])*olen[1] + j] = 0;

                for (size_t k = pad_offset[0]; k < olen[0] - pad_offset[0]; k++)
                for (size_t j = 0; j < pad_offset[1]; j++)
                    result.b[i*osize + k*olen[1] + j] = result.b[i*osize + k*olen[1] + j + olen_clamped[1] + pad_offset[1]] = 0;

                vsldCorrExec(task, kernel.b + i*ksize, klen+1, b + i*isize, ilen+1, result.b + i*osize, olen+1);
            }
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 2);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.end()[-2] = olen[0];
            Cshape.back()    = olen[1];

            result.resize(Cshape);

            auto in = b;
            auto corr = [task, kernel = kernel.b, in, result = result.b, isize, ksize, osize, ilen, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t k = 0; k < pad_offset[0]; k++)
                for (size_t j = 0; j < olen[1]; j++)
                    result[Coff*osize + k*olen[1] + j] = result[Coff*osize + (k + olen_clamped[0] + pad_offset[0])*olen[1] + j] = 0;

                for (size_t k = pad_offset[0]; k < olen[0] - pad_offset[0]; k++)
                for (size_t j = 0; j < pad_offset[1]; j++)
                    result[Coff*osize + k*olen[1] + j] = result[Coff*osize + k*olen[1] + j + olen_clamped[1] + pad_offset[1]] = 0;

                vsldCorrExec(task, kernel + Boff*ksize, klen+1, in + Aoff*isize, ilen+1, result + Coff*osize, olen+1);
            };

            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), corr);
            
        }

        vslCorrDeleteTask(&task);

        return result;
    }

    /**
     * @brief Cross Correlation 3d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm 
     * @return Tensor 
     */
    Tensor Tensor::correlation3d(const Tensor &kernel, Tuple3d padding, Tuple3d stride, Tuple3d dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 3 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 3 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        long long ilen[4], klen[4], olen[4], olen_clamped[3];
        ilen[2] = Ashape.back(); Ashape.pop_back();
        ilen[1] = Ashape.back(); Ashape.pop_back();
        ilen[0] = Ashape.back(); Ashape.pop_back();
        klen[2] = Bshape.back(); Bshape.pop_back();
        klen[1] = Bshape.back(); Bshape.pop_back();
        klen[0] = Bshape.back(); Bshape.pop_back();
        olen[2] = ilen[2] + 1 + 2*padding.x - klen[2];
        olen[1] = ilen[1] + 1 + 2*padding.y - klen[1];
        olen[0] = ilen[0] + 1 + 2*padding.z - klen[0];
        olen_clamped[2] = ilen[2] + 1 + 2*std::min<long long>(padding.x, klen[2]-1) - klen[2];
        olen_clamped[1] = ilen[1] + 1 + 2*std::min<long long>(padding.y, klen[1]-1) - klen[1];
        olen_clamped[0] = ilen[0] + 1 + 2*std::min<long long>(padding.z, klen[0]-1) - klen[0];
        size_t pad_offset[3] = {padding.z > klen[0] - 1 ? padding.z + 1 - klen[0] : 0,
                                padding.y > klen[1] - 1 ? padding.y + 1 - klen[1] : 0,
                                padding.x > klen[2] - 1 ? padding.x + 1 - klen[2] : 0};
        size_t isize = ilen[0]*ilen[1]*ilen[2];
        size_t ksize = klen[0]*klen[1]*klen[2];
        size_t osize = olen[0]*olen[1]*olen[2];
        ilen[3] = klen[3] = olen[3] = 1;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start[3] = {-std::min<long long>(padding.z, klen[0]-1),
                              -std::min<long long>(padding.y, klen[1]-1),
                              -std::min<long long>(padding.x, klen[2]-1)};
        VSLCorrTaskPtr task;
        vsldCorrNewTask(&task, VSL_CORR_MODE_AUTO, 3, klen, ilen, olen_clamped);
        vslCorrSetStart(task, start);
        
        ilen[1] *= ilen[2];
        klen[1] *= klen[2];

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.insert(Ashape.end(), olen, olen + 3);
            result.resize(Ashape);
            olen[1] *= olen[2];
            
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t l = 0; l < pad_offset[0]; l++)
                    for (size_t k = 0; k < olen[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + ((l + olen_clamped[0] + pad_offset[0])*olen[1] + k)*olen[2] + j] = 0;

                for (size_t l = pad_offset[0]; l < olen[0] - pad_offset[0]; l++)
                {
                    for (size_t k = 0; k < pad_offset[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + (l*olen[1] + k + olen_clamped[1] + pad_offset[1])*olen[2] + j] = 0;

                    for (size_t k = pad_offset[1]; k < olen[1] - pad_offset[1]; k++)
                    for (size_t j = 0; j < pad_offset[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + (l*olen[1] + k)*olen[2] + j + olen_clamped[2] + pad_offset[2]] = 0;
                }

                vsldCorrExec(task, kernel.b + i*ksize, klen+1, b + i*isize, ilen+1, result.b + i*osize, olen+1);
            }
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 3);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.end()[-3] = olen[0];
            Cshape.end()[-2] = olen[1];
            Cshape.back()    = olen[2];

            result.resize(Cshape);
            olen[1] *= olen[2];

            auto in = b;
            auto corr = [task, kernel = kernel.b, in, result = result.b, isize, ksize, osize, ilen, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t l = 0; l < pad_offset[0]; l++)
                    for (size_t k = 0; k < olen[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + ((l + olen_clamped[0] + pad_offset[0])*olen[1] + k)*olen[2] + j] = 0;

                for (size_t l = pad_offset[0]; l < olen[0] - pad_offset[0]; l++)
                {
                    for (size_t k = 0; k < pad_offset[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + (l*olen[1] + k + olen_clamped[1] + pad_offset[1])*olen[2] + j] = 0;

                    for (size_t k = pad_offset[1]; k < olen[1] - pad_offset[1]; k++)
                    for (size_t j = 0; j < pad_offset[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + (l*olen[1] + k)*olen[2] + j + olen_clamped[2] + pad_offset[2]] = 0;
                }

                vsldCorrExec(task, kernel + Boff*ksize, klen+1, in + Aoff*isize, ilen+1, result + Coff*osize, olen+1);
            };

            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), corr);
            
        }

        vslCorrDeleteTask(&task);

        return result;
    }

    /**
     * @brief Convolution 1d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm 
     * @return Tensor 
     */
    Tensor Tensor::convolution1d(const Tensor &kernel, size_t padding, size_t stride, size_t dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 1 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 1 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        size_t len  = Ashape.back(); Ashape.pop_back();
        size_t klen = Bshape.back(); Bshape.pop_back();
        size_t olen = len + 1 + 2*padding - klen;
        size_t olen_clamped = len + 1 + 2*std::min(padding, klen-1) - klen;
        size_t pad_offset = padding > klen - 1 ? padding + 1 - klen : 0;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start = (long long)klen - 1 - std::min(padding, klen-1);
        VSLConvTaskPtr task;
        vsldConvNewTask1D(&task, VSL_CONV_MODE_AUTO, klen, len, olen_clamped);
        vslConvSetStart(task, &start);

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.push_back(olen);
            result.resize(Ashape);

            auto begin = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t j = 0; j < pad_offset; j++)
                    result.b[i*olen + j] = result.b[i*olen + j + olen_clamped + pad_offset] = 0;
                
                vsldConvExec1D(task, kernel.b + i*klen, 1, b + i*len, 1, result.b + i*olen + pad_offset, 1);
            }
            ctime += (std::chrono::high_resolution_clock::now() - begin).count();
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 1);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.back() = olen;

            result.resize(Cshape);

            auto in = b;
            auto conv = [task, kernel = kernel.b, in, result = result.b, len, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t j = 0; j < pad_offset; j++)
                    result[Coff*olen + j] = result[Coff*olen + j + olen_clamped + pad_offset] = 0;

                vsldConvExec1D(task, kernel + Boff*klen, 1, in + Aoff*len, 1, result + Coff*olen + pad_offset, 1);
            };

            auto begin = std::chrono::high_resolution_clock::now();
            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), conv);
            /* broadcast_jobs.wait(); */
            ctime += (std::chrono::high_resolution_clock::now() - begin).count();
        }

        vslConvDeleteTask(&task);

        return result;
    }

    /**
     * @brief Convolution 2d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm Padding mode
     * @return Tensor 
     */
    Tensor Tensor::convolution2d(const Tensor &kernel, Tuple2d padding, Tuple2d stride, Tuple2d dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 2 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 2 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        long long ilen[3], klen[3], olen[3], olen_clamped[2];
        ilen[1] = Ashape.back(); Ashape.pop_back();
        ilen[0] = Ashape.back(); Ashape.pop_back();
        klen[1] = Bshape.back(); Bshape.pop_back();
        klen[0] = Bshape.back(); Bshape.pop_back();
        olen[1] = ilen[1] + 1 + 2*padding.x - klen[1];
        olen[0] = ilen[0] + 1 + 2*padding.y - klen[0];
        olen_clamped[1] = ilen[1] + 1 + 2*std::min<long long>(padding.x, klen[1]-1) - klen[1];
        olen_clamped[0] = ilen[0] + 1 + 2*std::min<long long>(padding.y, klen[0]-1) - klen[0];
        size_t pad_offset[2] = {padding.y > klen[0] - 1 ? padding.y + 1 - klen[0] : 0,
                                padding.x > klen[1] - 1 ? padding.x + 1 - klen[1] : 0};
        size_t isize = ilen[0]*ilen[1];
        size_t ksize = klen[0]*klen[1];
        size_t osize = olen[0]*olen[1];
        ilen[2] = klen[2] = olen[2] = 1;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start[2] = {klen[0] - 1 - std::min<long long>(padding.y, klen[0]-1),
                              klen[1] - 1 - std::min<long long>(padding.x, klen[1]-1)};
        VSLConvTaskPtr task;
        vsldConvNewTask(&task, VSL_CONV_MODE_AUTO, 2, klen, ilen, olen_clamped);
        vslConvSetStart(task, start);

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.insert(Ashape.end(), olen, olen + 2);
            result.resize(Ashape);
            
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t k = 0; k < pad_offset[0]; k++)
                for (size_t j = 0; j < olen[1]; j++)
                    result.b[i*osize + k*olen[1] + j] = result.b[i*osize + (k + olen_clamped[0] + pad_offset[0])*olen[1] + j] = 0;

                for (size_t k = pad_offset[0]; k < olen[0] - pad_offset[0]; k++)
                for (size_t j = 0; j < pad_offset[1]; j++)
                    result.b[i*osize + k*olen[1] + j] = result.b[i*osize + k*olen[1] + j + olen_clamped[1] + pad_offset[1]] = 0;

                vsldConvExec(task, kernel.b + i*ksize, klen+1, b + i*isize, ilen+1, result.b + i*osize, olen+1);
            }
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 2);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.end()[-2] = olen[0];
            Cshape.back()    = olen[1];

            result.resize(Cshape);

            auto in = b;
            auto conv = [task, kernel = kernel.b, in, result = result.b, isize, ksize, osize, ilen, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t k = 0; k < pad_offset[0]; k++)
                for (size_t j = 0; j < olen[1]; j++)
                    result[Coff*osize + k*olen[1] + j] = result[Coff*osize + (k + olen_clamped[0] + pad_offset[0])*olen[1] + j] = 0;

                for (size_t k = pad_offset[0]; k < olen[0] - pad_offset[0]; k++)
                for (size_t j = 0; j < pad_offset[1]; j++)
                    result[Coff*osize + k*olen[1] + j] = result[Coff*osize + k*olen[1] + j + olen_clamped[1] + pad_offset[1]] = 0;

                vsldConvExec(task, kernel + Boff*ksize, klen+1, in + Aoff*isize, ilen+1, result + Coff*osize, olen+1);
            };

            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), conv);
            
        }

        vslConvDeleteTask(&task);

        return result;
    }

    /**
     * @brief Convolution 3d
     * 
     * @param kernel 
     * @param padding 
     * @param stride 
     * @param dilation 
     * @param pm 
     * @return Tensor 
     */
    Tensor Tensor::convolution3d(const Tensor &kernel, Tuple3d padding, Tuple3d stride, Tuple2d dilation, PaddingMode pm) const
    {
        Tensor result({0});
        auto Ashape = shape, Bshape = kernel.shape;
        while (Ashape.size() < 2 || Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), 1);
        while (Bshape.size() < 2 || Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), 1);

        long long ilen[4], klen[4], olen[4], olen_clamped[3];
        ilen[2] = Ashape.back(); Ashape.pop_back();
        ilen[1] = Ashape.back(); Ashape.pop_back();
        ilen[0] = Ashape.back(); Ashape.pop_back();
        klen[2] = Bshape.back(); Bshape.pop_back();
        klen[1] = Bshape.back(); Bshape.pop_back();
        klen[0] = Bshape.back(); Bshape.pop_back();
        olen[2] = ilen[2] + 1 + 2*padding.x - klen[2];
        olen[1] = ilen[1] + 1 + 2*padding.y - klen[1];
        olen[0] = ilen[0] + 1 + 2*padding.z - klen[0];
        olen_clamped[2] = ilen[2] + 1 + 2*std::min<long long>(padding.x, klen[2]-1) - klen[2];
        olen_clamped[1] = ilen[1] + 1 + 2*std::min<long long>(padding.y, klen[1]-1) - klen[1];
        olen_clamped[0] = ilen[0] + 1 + 2*std::min<long long>(padding.z, klen[0]-1) - klen[0];
        size_t pad_offset[3] = {padding.z > klen[0] - 1 ? padding.z + 1 - klen[0] : 0,
                                padding.y > klen[1] - 1 ? padding.y + 1 - klen[1] : 0,
                                padding.x > klen[2] - 1 ? padding.x + 1 - klen[2] : 0};
        size_t isize = ilen[0]*ilen[1]*ilen[2];
        size_t ksize = klen[0]*klen[1]*klen[2];
        size_t osize = olen[0]*olen[1]*olen[2];
        ilen[3] = klen[3] = olen[3] = 1;

        size_t Asize = 1, Bsize = 1;
        for (size_t i = 0; i < Ashape.size(); i++) Asize *= Ashape[i];

        long long start[3] = {klen[0] - 1 - std::min<long long>(padding.z, klen[0]-1),
                              klen[1] - 1 - std::min<long long>(padding.y, klen[1]-1),
                              klen[2] - 1 - std::min<long long>(padding.x, klen[2]-1)};
        VSLConvTaskPtr task;
        vsldConvNewTask(&task, VSL_CONV_MODE_AUTO, 3, klen, ilen, olen_clamped);
        vslConvSetStart(task, start);

        ilen[1] *= ilen[2];
        klen[1] *= klen[2];

        if (sizeMatch(Ashape, Bshape))
        {
            Ashape.insert(Ashape.end(), olen, olen + 3);
            result.resize(Ashape);
            olen[1] *= olen[2];
            
            for (size_t i = 0; i < Asize; i++)
            {
                for (size_t l = 0; l < pad_offset[0]; l++)
                    for (size_t k = 0; k < olen[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + ((l + olen_clamped[0] + pad_offset[0])*olen[1] + k)*olen[2] + j] = 0;

                for (size_t l = pad_offset[0]; l < olen[0] - pad_offset[0]; l++)
                {
                    for (size_t k = 0; k < pad_offset[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + (l*olen[1] + k + olen_clamped[1] + pad_offset[1])*olen[2] + j] = 0;

                    for (size_t k = pad_offset[1]; k < olen[1] - pad_offset[1]; k++)
                    for (size_t j = 0; j < pad_offset[2]; j++)
                        result.b[i*osize + (l*olen[1] + k)*olen[2] + j] = result.b[i*osize + (l*olen[1] + k)*olen[2] + j + olen_clamped[2] + pad_offset[2]] = 0;
                }

                vsldConvExec(task, kernel.b + i*ksize, klen+1, b + i*isize, ilen+1, result.b + i*osize, olen+1);
            }
        }
        else if (broadcastable(Ashape, Bshape))
        {
            std::vector<size_t> Cshape(Ashape.size() + 3);
            size_t Csize = 1;
            for (size_t i = 0; i < Ashape.size(); i++)
                Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i],
                Csize *= Cshape[i];
            
            Cshape.end()[-3] = olen[0];
            Cshape.end()[-2] = olen[1];
            Cshape.back()    = olen[2];

            result.resize(Cshape);
            olen[1] *= olen[2];

            auto in = b;
            auto conv = [task, kernel = kernel.b, in, result = result.b, isize, ksize, osize, ilen, klen, olen, olen_clamped, pad_offset]
                        (size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t l = 0; l < pad_offset[0]; l++)
                    for (size_t k = 0; k < olen[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + ((l + olen_clamped[0] + pad_offset[0])*olen[1] + k)*olen[2] + j] = 0;

                for (size_t l = pad_offset[0]; l < olen[0] - pad_offset[0]; l++)
                {
                    for (size_t k = 0; k < pad_offset[1]; k++)
                    for (size_t j = 0; j < olen[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + (l*olen[1] + k + olen_clamped[1] + pad_offset[1])*olen[2] + j] = 0;

                    for (size_t k = pad_offset[1]; k < olen[1] - pad_offset[1]; k++)
                    for (size_t j = 0; j < pad_offset[2]; j++)
                        result[Coff*osize + (l*olen[1] + k)*olen[2] + j] = result[Coff*osize + (l*olen[1] + k)*olen[2] + j + olen_clamped[2] + pad_offset[2]] = 0;
                }

                vsldConvExec(task, kernel + Boff*ksize, klen+1, in + Aoff*isize, ilen+1, result + Coff*osize, olen+1);
            };

            broadcast_operation(Ashape.data(), Bshape.data(), Cshape.data(), Ashape.size(), conv);
            
        }

        vslConvDeleteTask(&task);

        return result;
    }

    void reprint(std::ostream &os, const Tensor &t, size_t depth, std::vector<size_t> &index)
    {
        if (depth == 0)
        {
            if (t.size)
                if (t.onCPU)
                    os << t(index);
                else
                {
                    float64 num = 0.;
                    size_t n = 0;
                    for (size_t i = 0; i < t.shape.size() - 1; i++)
                        n = (n + index[i]) * t.shape[i + 1];
                    OpenCLManager::loadReadBuffer<float64>(t.buffer, 1, &num, n + index.back());
                    os << num;
                }
            return;
        }

        if (depth > 1)
        {
            os << "[\n";
            for (size_t i = 0; i < t.shape.size() - depth + 1; i++)
                os << "  ";
        }
        else
            os << "[";

        index.push_back(0);
        for (size_t i = 0; i + 1 < t.shape[t.shape.size() - depth]; i++)
        {
            index.back() = i;
            if (i == 4 && depth == 1 && t.shape.back() > 8)
            {
                os << "...";
                i = t.shape.back() - 4 - 1;
            }
            else if (i == 3 && depth == 2 && t.shape[t.shape.size() - depth] > 6)
            {
                os << "...";
                i = t.shape[t.shape.size() - depth] - 3 - 1;
            }
            else if (i == 2 && depth == 3 && t.shape[t.shape.size() - depth] > 4)
            {
                os << "...";
                i = t.shape[t.shape.size() - depth] - 2 - 1;
            }
            else if (i == 1 && depth >= 4 && t.shape[t.shape.size() - depth] > 2)
            {
                os << "...";
                i = t.shape[t.shape.size() - depth] - 1 - 1;
            }
            else
                reprint(os, t, depth - 1, index);
            if (depth > 1)
            {
                os << ",\n";
                for (size_t i = 0; i < t.shape.size() - depth + 1; i++)
                    os << "  ";
            }
            else
                os << ", ";
        }
        index.back() = t.shape[t.shape.size() - depth] - 1;
        reprint(os, t, depth - 1, index);
        index.pop_back();

        if (depth > 1)
        {
            os << "\n";
            for (size_t i = 0; i < t.shape.size() - depth; i++)
                os << "  ";
        }
        os << "]";
    }

    /**
     * @brief Overloading of the << operetor for ostream
     * 
     * @param os 
     * @param t 
     * @return std::ostream& 
     */
    std::ostream &operator<<(std::ostream &os, const Tensor &t)
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
    
    /**
     * @brief write this tensor to file buffer
     * 
     * @param file 
     * @return uint64_t 
     */
    uint64_t Tensor::save(std::ofstream &file) const
    {
        const char name[] = "Tensor";
        file.write(name, sizeof(name));

        uint64_t size = sizeof(uint64_t) + shape.size() * sizeof(uint64_t) + sizeof(float64) * this->size;
        file.write((char*)&size, sizeof(size));

        uint64_t shape_size = shape.size();
        file.write((char*)&shape_size, sizeof(shape_size));
        for (size_t i = 0; i < shape.size(); i++)
        {
            uint64_t shape_size = shape[i];
            file.write((char*)&shape_size, sizeof(shape_size));
        }

        file.write((char*)b, this->size * sizeof(float64));

        return size + sizeof(uint64_t) + sizeof(name);
    }

    /**
     * @brief check if s1 and s2 are compatible sizes for element wise operations
     * 
     * @param s1 
     * @param s2 
     * @return true 
     * @return false 
     */
    bool Tensor::sizeMatch(const std::vector<size_t> &s1, const std::vector<size_t> &s2)
    {
        // if (size != t.size) return false;
        size_t end = std::min(s1.size(), s2.size());
        for (size_t i = 1; i <= end; i++)
            if (s1[s1.size() - i] != s2[s2.size() - i])
                return false;
        for (size_t i = 0; i < s1.size() - end; i++)
            if (s1[i] != 1)
                return false;
        for (size_t i = 0; i < s2.size() - end; i++)
            if (s2[i] != 1)
                return false;

        return true;
    }
    
    /**
     * @brief check if s1 and s2 are compatible sizes for broadcast operations
     * 
     * @param s1 
     * @param s2 
     * @return true 
     * @return false 
     */
    bool Tensor::broadcastable(const std::vector<size_t> &s1, const std::vector<size_t> &s2)
    {
        auto p1 = s1.end() - 1;
        auto p2 = s2.end() - 1;

        while (p1 != s1.begin() && p2 != s2.begin())
        {
            if (*p1 != *p2 && *p1 != 1 && *p2 != 1)
                return false;
            p1--, p2--;
        }
        return true;
    }
    
    /**
     * @brief create new uninitialized tensor with the same shape as t
     * 
     * @param t 
     * @return Tensor 
     */
    Tensor Tensor::empty_like(const Tensor &t)
    {
        return {t.shape};
    }

    /**
     * @brief create new tensor with the same shape as t filled with zeros
     * 
     * @param t 
     * @return Tensor 
     */
    Tensor Tensor::zeros_like(const Tensor &t)
    {
        Tensor zl(t.shape);
        zl.zero();
        return zl;
    }

    /**
     * @brief create new tensor with the same shape as t filled with ones
     * 
     * @param t 
     * @return Tensor 
     */
    Tensor Tensor::ones_like(const Tensor &t)
    {
        Tensor zl(t.shape);
        zl.ones();
        return zl;
    }

    
    RedFish::Tensor Tensor::stack(const RedFish::Tensor &t1, const RedFish::Tensor &t2, size_t dim)
    {
        auto t1_shape = t1.shape, t2_shape = t2.shape;
        while (t1_shape.size() < dim+1 || t1_shape.size() < t2_shape.size()) t1_shape.insert(t1_shape.begin(), 1);
        while (t2_shape.size() < dim+1 || t2_shape.size() < t1_shape.size()) t2_shape.insert(t2_shape.begin(), 1);

        for (size_t i = 0; i < t1_shape.size(); i++)
            if (i != t1_shape.size() - 1 - dim && t1_shape[i] != t2_shape[i])
                throw std::length_error("Tensor shapes not matching in Tensor::stack");

        std::vector<size_t> t3_shape = t1_shape;
        t3_shape[t3_shape.size() - 1 - dim] += t2_shape[t3_shape.size() - 1 - dim];

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

    Tensor Tensor::stack(const std::vector<Tensor> &tensors, size_t dim)
    {
        std::vector<std::vector<size_t>> t_shapes;
        t_shapes.reserve(tensors.size());
        size_t largest_shape_size = 0;
        for (auto& t : tensors)
        {
            t_shapes.emplace_back(t.shape);
            if (largest_shape_size < t.shape.size()) largest_shape_size = t.shape.size();
        }
        if (largest_shape_size < dim+1) largest_shape_size = dim+1;
        for (auto& t_shape : t_shapes)
            if (t_shape.size() < largest_shape_size)
                t_shape.insert(t_shape.begin(), largest_shape_size-t_shape.size(), 1);

        std::vector<size_t> out_shape(largest_shape_size, 0);
        if (t_shapes.size()) out_shape = t_shapes.front();
        out_shape[out_shape.size() - 1 - dim] = 0;
        for (auto& t_shape : t_shapes)
            for (size_t i = 0; i < t_shape.size(); i++)
            {
                if (i == t_shape.size() - 1 - dim)
                    out_shape[i] += t_shape[i];
                else if (t_shape[i] != out_shape[i])
                    throw std::length_error("Tensor shapes not matching in Tensor::stack");
            }

        Tensor result(out_shape);

        std::vector<size_t> n(tensors.size(), 1);
        for (size_t i = out_shape.size() - dim - 1; i < out_shape.size(); i++)
        {
            for (size_t k = 0; k < n.size(); k++)
                n[k] *= t_shapes[k][i];
        }
        size_t nout = std::accumulate(n.begin(), n.end(), (size_t)0);

        size_t p = 1;
        for (size_t i = 0; i < out_shape.size() - dim - 1; i++)
            p *= out_shape[i];

        for (size_t i = 0; i < p; i++)
        {
            size_t in3 = i * nout;
            for (size_t j = 0, l = 0; j < tensors.size(); j++)
            {
                size_t inj = i * n[j];
                for (size_t k = 0; k < n[j]; k++)
                    result.b[in3 + l + k] = tensors[j].b[inj + k];
                l += n[j];
            }
        }

        return result;
    }

    /**
     * @brief For each element applay function 
     * 
     * @param t 
     * @param fn 
     * @return Tensor 
     */
    Tensor forEach(const Tensor &t, std::function<float64(float64)> fn)
    {
        Tensor ret(t.shape);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = fn(t.b[i]);
        return ret;
    }

    /**
     * @brief For each element applay function in place 
     * 
     * @param t 
     * @param fn 
     * @return Tensor 
     */
    Tensor &forEachInPlace(Tensor &t, std::function<float64(float64)> fn)
    {
        for (size_t i = 0; i < t.size; i++)
            t.b[i] = fn(t.b[i]);
        return t;
    }



    /* 
     *      TEMPLATED OPERATIONS
     */


    template <void (*fn)(float64 &, float64)>
    Tensor Tensor::axes_reduction(const Tensor &t, size_t d, const float64 init_val, const bool collapse)
    {
        auto shape = t.shape;
        if (d < t.shape.size())
        {
            d = t.shape.size() - d - 1;        
            shape[d] = 1;
            size_t size = 1;
            for (size_t i = 0; i < d; i++) size *= shape[i];
            if (size == 1) shape.erase(shape.begin(), shape.begin() + d);
        }
        Tensor ret(shape);

        if (d < t.shape.size())
        {
            size_t tot = 1, stride = 1;
            for (size_t i = 0; i < d; i++)
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
            if (collapse) ret.shape.erase(ret.shape.begin() + d);
        }
        else std::copy(t.b, t.b + t.size, ret.b);


        return ret;
    }

    template <void (*fn)(float64 &, float64)>
    Tensor Tensor::axes_reduction(const Tensor &t, const std::vector<size_t> &ds, const float64 init_val) //To-Do
    {
        auto shape = t.shape;
        for (auto d : ds)
            if (d < t.shape.size())
            {
                d = t.shape.size() - d - 1;
                shape[d] = std::min((size_t)1, shape[d]);
            }
        Tensor ret(shape);

        //if (d < t.shape.size())
        {
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
        }
        else std::copy(t.b, t.b + t.size, ret.b);

        return ret;
    }

    template <void (*fn)(float64 &, float64)>
    float64 Tensor::full_reduction(const Tensor &t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
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

    void Tensor::broadcast_ew_device(const size_t fn, Buffer dst, Buffer src1, Buffer src2,
                                    const size_t *shape, const size_t *shape1, const size_t *shape2,
                                    size_t depth,
                                    size_t off, size_t off1, size_t off2)
    {
        if (depth > 3)
            for (size_t i = 0, bdc1 = (*shape1 == *shape) * ((size_t)-1), bdc2 = (*shape2 == *shape) * ((size_t)-1); i < *shape; i++)
                broadcast_ew_device(fn, dst, src1, src2, shape + 1, shape1 + 1, shape2 + 1, depth - 1, off * *shape + i, off1 * *shape1 + (i & bdc1), off2 * *shape2 + (i & bdc2));
        else
            OpenCLManager::execute(fn, std::array<size_t, 3>({shape[0], shape[1], shape[2]}),
                                   OpenCLManager::getSubBuffer<float64>(src1, off1, shape1[0]*shape1[1]*shape1[2]),
                                   OpenCLManager::getSubBuffer<float64>(src2, off2, shape2[0]*shape2[1]*shape2[2]),
                                   OpenCLManager::getSubBuffer<float64>(dst,  off,  shape [0]*shape [1]*shape [2]));
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
    template <float64 (*fn)(float64, float64), size_t fn_device, size_t fn_device_brdc>
    Tensor Tensor::ew_or_broadcast(const Tensor &t1, const Tensor &t2, const char *err_msg)
    {
        if (t1.onCPU != t2.onCPU)
            throw std::runtime_error("Tensor memory not on the same device");

        Tensor result;
        if (!t1.onCPU) result.toDevice();

        if (Tensor::sizeMatch(t1.shape, t2.shape))
        {
            result.resize(t1.shape);

            if (t1.onCPU)
                for (size_t i = 0; i < t1.size; i++)
                    result.b[i] = fn(t1.b[i], t2.b[i]);
            else
                OpenCLManager::execute(fn_device, {t1.size}, OpenCLManager::getBuffer(t1.buffer), OpenCLManager::getBuffer(t2.buffer), OpenCLManager::getBuffer(result.buffer));
        }
        else if (t1.broadcastable(t1.shape, t2.shape))
        {
            auto shapeT1 = t1.shape;
            auto shapeT2 = t2.shape;

            for (size_t i = 0; shapeT1.size() < shapeT2.size(); i++)
                shapeT1.insert(shapeT1.begin(), 1);

            for (size_t i = 0; shapeT2.size() < shapeT1.size(); i++)
                shapeT2.insert(shapeT2.begin(), 1);

            std::vector<size_t> shapeDst(shapeT1.size());
            for (size_t i = 0; i < shapeT1.size(); i++)
                shapeDst[i] = shapeT1[i] == 1 ? shapeT2[i] : shapeT1[i];
                
            result.resize(shapeDst);

            size_t idx = 0;
            enum { SAME, BRD1, BRD2, NONE } lastd = NONE, thisd, last3[3];

            while (idx < shapeT1.size())
            {
                if (shapeT1[idx] == shapeT2[idx])
                    if (shapeT1[idx] == 1) 
                    {
                        shapeT1 .erase(shapeT1 .begin() + idx);
                        shapeT2 .erase(shapeT2 .begin() + idx);
                        shapeDst.erase(shapeDst.begin() + idx);
                        continue;
                    }
                    else thisd = SAME;
                else
                    if (shapeT1[idx] == 1) thisd = BRD1;
                    else thisd = BRD2;
                
                if (thisd == lastd)
                {
                    shapeT1[idx-1]  *= shapeT1[idx];
                    shapeT2[idx-1]  *= shapeT2[idx];
                    shapeDst[idx-1] *= shapeDst[idx];
                    shapeT1 .erase(shapeT1 .begin() + idx);
                    shapeT2 .erase(shapeT2 .begin() + idx);
                    shapeDst.erase(shapeDst.begin() + idx);
                }

                lastd = thisd;
                idx++;
            }

            for (size_t i = 0; shapeT1.size() < 3; i++) shapeT1.insert(shapeT1.begin(), 1);
            for (size_t i = 0; shapeT2.size() < 3; i++) shapeT2.insert(shapeT2.begin(), 1);
            for (size_t i = 0; shapeDst.size() < 3; i++) shapeDst.insert(shapeDst.begin(), 1);

            last3[0] = shapeT1.end()[-3] == shapeT2.end()[-3] ? SAME : (shapeT1.end()[-3] == 1 ? BRD1 : BRD2);
            last3[1] = shapeT1.end()[-2] == shapeT2.end()[-2] ? SAME : (shapeT1.end()[-2] == 1 ? BRD1 : BRD2);
            last3[2] = shapeT1.end()[-1] == shapeT2.end()[-1] ? SAME : (shapeT1.end()[-1] == 1 ? BRD1 : BRD2);

            enum bdrconf { N0B1N0, N0B1B2, N0B2N0, N0B2B1,
                           B1N0B1, B1N0B2, B1B2N0, B1B2B1,
                           B2N0B1, B2N0B2, B2B1N0, B2B1B2  } conf;
            
            if (last3[2] == SAME)
                if (last3[1] == BRD1)
                    if (last3[0] == SAME) conf = N0B1N0; //0
                    else                  conf = B2B1N0; //10
                else
                    if (last3[0] == SAME) conf = N0B2N0; //2
                    else                  conf = B1B2N0; //6

            else if (last3[2] == BRD1)
                if (last3[1] == SAME)
                    if (last3[0] == BRD1) conf = B1N0B1; //4
                    else                  conf = B2N0B1; //8
                else
                    if (last3[2] == SAME) conf = N0B2B1; //3
                    else                  conf = B1B2B1; //7

            else if (last3[2] == BRD2)
                if (last3[1] == SAME)
                    if (last3[0] == BRD1) conf = B1N0B2; //5
                    else                  conf = B2N0B2; //9
                else
                    if (last3[0] == SAME) conf = N0B1B2; //1
                    else                  conf = B2B1B2; //11

            if (result.size)
                if (t1.onCPU)
                    broadcast_ew_assign<fn>(result.b, t1.b, t2.b, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
                else
                    broadcast_ew_device(fn_device_brdc + conf, result.buffer, t1.buffer, t2.buffer, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
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
    template <float64 (*fn)(float64, float64), size_t fn_device>
    void Tensor::ew_or_broadcast_assign(Tensor &t1, const Tensor &t2, const char *err_msg)
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
            {
                Tensor result(shapeDst);
                if (result.size)
                    broadcast_ew_assign<fn>(result.b, t1.b, t2.b, shapeDst.data(), shapeT1.data(), shapeT2.data(), shapeDst.size());
                t1 = std::move(result);
            }
        }
        else
            throw std::length_error(err_msg);
    }

    template <auto fn, typename... Args>
    static void broadcast_op_impl(float64 *dst, const float64 *src1, const float64 *src2,
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
    static void broadcast_op(float64 *dst, const float64 *src1, const float64 *src2,
                             const size_t *shape, const size_t *shape1, const size_t *shape2,
                             size_t depth,
                             size_t foff, size_t foff1, size_t foff2,
                             Args... args)
    {
        broadcast_op_impl<fn, Args...>(dst, src1, src2, shape, shape1, shape2, depth, foff, foff1, foff2, (size_t)0, (size_t)0, (size_t)0, args...);
    }

    template <auto fn>
    static void for_(const size_t size[], const size_t ld[], std::vector<size_t>& index, size_t height, float64* b, size_t depth, size_t off)
    {
        index[depth] = 0;
        if (depth + 1 < height)
        for (size_t i = 0; i < size[depth]; i++, index[depth]++)
            // for_(size, ld, index, height, b, depth + 1, off*(*ld) + i);
        // else
        for (size_t i = 0; i < size[depth]; i++, index[depth]++)
        {
            //fn(b[off*(*ld) + i], index);
        }
        
    }



    /* 
     *      TEMPLATED OPERATIONS (end)
     */


    static void copy_2d(float64 *src, float64 *dst, Tuple2d size, size_t stride_out, size_t stride_in)
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


}