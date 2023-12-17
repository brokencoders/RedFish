#include "Tensor.h"

#include "OpenCLManager.h"

namespace RedFish
{
    void matmul_gotoblas(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
    void matmul_left_T(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
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
        else return new float64[size];
    }
    
    static void dealloc(float64*& buff)
    {
        if (buff) delete buff;
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
    Tensor::Tensor(const std::vector<size_t>& shape)
        : shape(shape), stride(shape), size(1), onCPU(true)
    {
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        b = alloc(size);
//                OpenCLManager::createBuffer<float64>(size);
    }

    /**
     * @brief  Construct a new uninitialized Tensor object with given shape
     * 
     * @param shape c-like array with tensor shape
     * @param len   shape array length
     */
    Tensor::Tensor(const size_t* shape, size_t len)
        : shape(shape, shape + len), stride(shape, shape + len), size(1), onCPU(true)
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
    Tensor::Tensor(const std::vector<size_t>& shape, float64 *buff, bool copy)
        : shape(shape), stride(shape), size(1), onCPU(true), b(nullptr)
    {
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (copy)
        {
            b = alloc(size);
            std::copy(buff, buff + size, b);
        }
        else
            b = buff;
    }

    /**
     * @brief Copy construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(const Tensor &t)
        : shape(t.shape), stride(t.stride), size(t.size), onCPU(t.onCPU)
    {
        if (onCPU)
        {
            this->b = alloc(size);
            std::copy(t.b, t.b + size, b);
        }
        else
            /* GPU copy */;

    }

    /**
     * @brief Move construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(Tensor &&t)
        : shape(t.shape), stride(t.stride), size(t.size), onCPU(t.onCPU), b(t.b), buffer(t.buffer)
    {
        t.shape  = {0};
        t.stride = {0};
        t.size   = 0;
        t.onCPU  = true;
        t.b      = nullptr;
    }

    /**
     * @brief Construct a new Tensor object from the shape and data as a list
     * 
     * @param shape of the Tensor t be created 
     * @param data as an initializer list 
     */
    Tensor::Tensor(const std::vector<size_t>& shape, std::initializer_list<float64> data)
        : shape(shape), stride(shape), size(1), onCPU(true)
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
        : size(1)
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
        stride = shape;

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
            /* GPU dealloc */;
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
        this->shape  = t.shape;
        this->stride = t.stride;
        this->size   = t.size;
        this->onCPU  = t.onCPU;

        if (onCPU)
        {
            this->b = alloc(size);
            std::copy(t.b, t.b + size, b);
        }
        else
            /* GPU copy */;

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
        this->shape  = t.shape;
        this->stride = t.stride;
        this->size   = t.size;
        this->onCPU  = t.onCPU;
        this->b      = t.b;
        this->buffer = t.buffer;
        t.shape  = {0};
        t.stride = {0};
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
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in sum operation");
    }

    Tensor Tensor::operator+(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] + val;

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
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in sum operation");
        return *this;
    }

    Tensor &Tensor::operator+=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] += val;

        return *this;
    }

    Tensor Tensor::operator-(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in subtraction operation");
    }

    Tensor Tensor::operator-(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] - val;

        return result;
    }

    Tensor Tensor::operator-() const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = -this->b[i];

        return result;
    }

    Tensor operator-(const float64 val, const Tensor &t)
    {
        Tensor ret = t.empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val - t.b[i];

        return ret;
    }

    Tensor &Tensor::operator-=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  -  n2; };
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in subtruction operation");
        return *this;
    }

    Tensor &Tensor::operator-=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] -= val;

        return *this;
    }

    Tensor Tensor::operator*(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  *  n2; };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
    }

    Tensor Tensor::operator*(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] * val;

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
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in multiplication operation");
        return *this;
    }

    Tensor &Tensor::operator*=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] *= val;

        return *this;
    }

    Tensor Tensor::operator/(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  /  n2; };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in division operation");
    }

    Tensor Tensor::operator/(const float64 val) const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++)
            result.b[i] = this->b[i] / val;

        return result;
    }

    Tensor operator/(const float64 val, const Tensor &t)
    {
        Tensor ret = t.empty_like(t);
        for (size_t i = 0; i < t.size; i++)
            ret.b[i] = val / t.b[i];

        return ret;
    }

    Tensor &Tensor::operator/=(const Tensor &t)
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return n1  +  n2; };
        ew_or_broadcast_assign<fn>(*this, t, "Tensor sizes not matching in division operation");
        return *this;
    }

    Tensor &Tensor::operator/=(const float64 val)
    {
        for (size_t i = 0; i < size; i++)
            this->b[i] /= val;

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
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in equality operation");
    }

    Tensor Tensor::operator<=(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 <= n2); };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in less then or equal operation");
    }

    Tensor Tensor::operator>=(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 >= n2); };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in greater then or equal operation");
    }


    Tensor Tensor::operator<(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 < n2); };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in less then operation");
    }

    Tensor Tensor::operator>(const Tensor &t) const
    {
        constexpr auto fn = [](float64 n1, float64 n2)
        {  return (float64)(n1 > n2); };
        return ew_or_broadcast<fn>(*this, t, "Tensor sizes not matching in greater then operation");
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
        for (size_t i = 0; i < N - 1; i++)
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
        for (size_t i = 0; i < N - 1; i++)
            off *= *(shape.end() + i - N + 1);

        for (size_t i = 0; i < N; i++)
            new_shape[i] = *(shape.end() - N + i);

        return DirectTensorView({new_shape, new_shape + N}, b + off);
    }

    /**
     * @brief set new size from a shape wile copping existing values 
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

    /**
     * @brief Get transposed over last two dimensions of this tensor
     * 
     * @return Tensor 
     */
    Tensor Tensor::T() const
    {
        Tensor t(shape);

        if (shape.size() < 1)
            t.b[0] = this->b[0];
        else
        {
            size_t end = 1;
            size_t cols = t.shape.back();
            size_t rows = shape.size() < 2 ? 1 : t.shape[t.shape.size() - 2];
            std::swap(t.shape.back(), t.shape[t.shape.size() - 2]);

            for (size_t i = 0; i < (int64_t)t.shape.size() - 2; i++)
                end *= t.shape[i];

            const size_t block_size = 8;
            const size_t k_end = rows - rows % block_size;
            const size_t j_end = cols - cols % block_size;

            for (size_t i = 0, stride = rows * cols; i < end; i++)
            {
                float64 *tp = t.b + i * stride, *thisp = this->b + i * stride;

                for (size_t k = 0; k < k_end; k += block_size)
                {
                    for (size_t j = 0; j < j_end; j += block_size)
                        for (size_t r = k; r < k + block_size; r++)
                            for (size_t c = j; c < j + block_size; c++)
                                tp[c * rows + r] = thisp[r * cols + c];

                    for (size_t r = k; r < k + block_size; r++)
                        for (size_t c = j_end; c < cols; c++)
                            tp[c * rows + r] = thisp[r * cols + c];
                }
                for (size_t j = 0; j < j_end; j += block_size)
                    for (size_t r = k_end; r < rows; r++)
                        for (size_t c = j; c < j + block_size; c++)
                            tp[c * rows + r] = thisp[r * cols + c];

                for (size_t r = k_end; r < rows; r++)
                    for (size_t c = j_end; c < cols; c++)
                        tp[c * rows + r] = thisp[r * cols + c];
            }
        }

        return t;
    }


    /**
     * @brief Transpose matrix dimension d1 and d2
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
        for (size_t i = 0; i < size; i++)
            b[i] = 0;
    }

    /**
     * @brief Set tensor with all One
     * 
     */
    void Tensor::ones()
    {
        if(onCPU)
        {
            for (size_t i = 0; i < size; i++)
                b[i] = 1;
        } else {
            OpenCLManager::execute(Kernel::SET, size, OpenCLManager::getBuffer(buffer), 1.0);
        }
    }

    /**
     * @brief Set all tensor elements to val
     * 
     * @param val
     */
    void Tensor::costant(float64 val)
    {
        for (size_t i = 0; i < size; i++)
            b[i] = val;
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
    Tensor Tensor::squareSum(size_t d) const
    {
        constexpr auto fn = [](float64 &sq_sum, float64 n)
        {  sq_sum += n  *  n; };
        return axes_reduction<fn>(*this, d, 0.);
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
    Tensor Tensor::max(size_t d) const
    {
        constexpr auto fn = [](float64 &max, float64 n)
        {if (max < n) max = n; };
        return axes_reduction<fn>(*this, d, -std::numeric_limits<float64>::infinity());
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
    Tensor Tensor::min(size_t d) const
    {
        constexpr auto fn = [](float64 &min, float64 n)
        {if (min > n) min = n; };
        return axes_reduction<fn>(*this, d, std::numeric_limits<float64>::infinity());
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
    Tensor Tensor::sum(size_t d) const
    {
        constexpr auto fn = [](float64 &sum, float64 n)
        {  sum += n; };
        return axes_reduction<fn>(*this, d, 0.);
    }

    /**
     * @brief Matrix Multipication 
     * 
     * @param t 
     * @param trsp wheter one of the two matrices has to be transopsed before multiplication
     * @return Tensor 
     */
    Tensor Tensor::matmul(const Tensor &t, const Transpose trsp) const
    {
        Tensor result;
        std::vector<size_t> shapeT1, matShapeT1(shape.begin() + std::max<int64_t>(0, (int64_t)shape.size() - 2), shape.end());
        std::vector<size_t> shapeT2, matShapeT2(t.shape.begin() + std::max<int64_t>(0, (int64_t)t.shape.size() - 2), t.shape.end());
        size_t size1 = 1, size2 = 1;

        for (size_t i = 0; i < (int64_t)shape.size() - 2; i++)
            shapeT1.push_back(shape[i]), size1 *= shape[i];
        for (size_t i = 0; i < (int64_t)t.shape.size() - 2; i++)
            shapeT2.push_back(t.shape[i]), size2 *= t.shape[i];
        matShapeT1.insert(matShapeT1.begin(), std::max<int64_t>(0, (int64_t)2 - shape.size()), 1);
        matShapeT2.insert(matShapeT2.begin(), std::max<int64_t>(0, (int64_t)2 - t.shape.size()), 1);
        if (t.shape.size() == 1)
            std::swap(matShapeT2[0], matShapeT2[1]);
        if (trsp == LEFT)
            std::swap(matShapeT1[0], matShapeT1[1]);
        else if (trsp == RIGHT)
            std::swap(matShapeT2[0], matShapeT2[1]);

        if (shapeT1.size() > shapeT2.size())
            shapeT2.insert(shapeT2.begin(), shapeT1.size() - shapeT2.size(), 1);
        else if (shapeT1.size() < shapeT2.size())
            shapeT1.insert(shapeT1.begin(), shapeT2.size() - shapeT1.size(), 1);

        if (matShapeT1[1] != matShapeT2[0])
            throw std::length_error("Matrix size not matching in Tensor matmul");

        size_t rows = matShapeT1[0];
        size_t mid = matShapeT1[1];
        size_t cols = matShapeT2[1];
        size_t matsize0 = rows * cols;
        size_t matsize1 = rows * mid;
        size_t matsize2 = mid * cols;

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
                    matmul_left_T(result.b + i * matsize0, b + i * matsize1, t.b + i * matsize2, rows, mid, cols, cols, rows, cols);
                break;
            case RIGHT:
            {
                auto tt = t.T();
                for (size_t i = 0; i < size1; i++)
                    matmul_gotoblas(result.b + i * matsize0, b + i * matsize1, tt.b + i * matsize2, rows, mid, cols, cols, mid, mid);
                break;
            }
            case NONE:
            default:
                for (size_t i = 0; i < size1; i++)
                    matmul_gotoblas(result.b + i * matsize0, b + i * matsize1, t.b + i * matsize2, rows, mid, cols, cols, mid, cols);
                break;
            }
        }
        else if (broadcastable(shapeT1, shapeT2))
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
                                                shapeDst.size() - 2,
                                                rows * cols, rows * mid, mid * cols,
                                                rows, mid, cols, cols, rows, cols);
                    break;
                case RIGHT:
                    broadcast_op<matmul_gotoblas>(result.b, this->b, t.T().b,
                                                  shapeDst.data(), shapeT1.data(), shapeT2.data(),
                                                  shapeDst.size() - 2,
                                                  rows * cols, rows * mid, mid * cols,
                                                  rows, mid, cols, cols, mid, cols);
                    break;
                case NONE:
                default:
                    broadcast_op<matmul_gotoblas>(result.b, this->b, t.b,
                                                  shapeDst.data(), shapeT1.data(), shapeT2.data(),
                                                  shapeDst.size() - 2,
                                                  rows * cols, rows * mid, mid * cols,
                                                  rows, mid, cols, cols, mid, cols);
                    break;
                }
        }
        else
            throw std::length_error("Shapes not matching in Tensor matmul");

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
    Tensor Tensor::crossCorrelation1d(const Tensor &kernel, size_t padding, size_t stride, size_t dilation, PaddingMode pm) const
    {
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 1)
            shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 1)
            shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 1 < shape_k.size(); i++)
            if (shape_k[i] != 1)
                throw std::length_error("Invalid kernel shape in crossCorrelation1d");
        if (shape_k.back() == 0)
            throw std::length_error("Invalid kernel shape in crossCorrelation1d");
        if (stride == 0)
            throw std::length_error("Invalid stride in crossCorrelation1d");

        for (size_t i = 0; i + 1 < shape_t.size(); i++)
            size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2 * padding + shape_t.back() + stride >= (shape_k.back() - 1) * dilation + 1)
            shape_r.back() = (2 * padding + shape_t.back() - (shape_k.back() - 1) * dilation - 1) / stride + 1;
        else
            shape_r.back() = 0, size_batch = 0;
        size_t off_t = shape_t.back();
        size_t off_r = shape_r.back();
        size_t off_p = shape_t.back() + 2 * padding;

        Tensor result = zeros_like(shape_r);

        std::unique_ptr<float64[]> padded = std::make_unique<float64[]>(off_p);
        for (size_t i = 0; i < size_batch; i++)
        {
            for (size_t c = 0; c < shape_t.back(); c++)
                padded[c + padding] = b[c + i * off_t];

            cross_correlation_1d_impl(
                result.b + i * off_r,
                padded.get(),
                kernel.b,
                shape_t.back() + 2 * padding,
                shape_k.back(),
                stride, dilation);
        }

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
    Tensor Tensor::crossCorrelation2d(const Tensor &kernel, Tuple2d padding, Tuple2d stride, Tuple2d dilation, PaddingMode pm) const
    {
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 2)
            shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 2)
            shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 2 < shape_k.size(); i++)
            if (shape_k[i] != 1)
                throw std::length_error("Invalid kernel shape in crossCorrelation2d");
        if (shape_k.back() == 0 || shape_k.end()[-2] == 0)
            throw std::length_error("Invalid kernel shape in crossCorrelation2d");
        if (stride.x == 0 || stride.y == 0)
            throw std::length_error("Invalid stride in crossCorrelation2d");

        for (size_t i = 0; i + 2 < shape_t.size(); i++)
            size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2 * padding.x + shape_t.back() + stride.x >= (shape_k.back() - 1) * dilation.x + 1)
            shape_r.back() = (2 * padding.x + shape_t.back() - (shape_k.back() - 1) * dilation.x + stride.x - 1) / stride.x;
        else
            shape_r.back() = 0, size_batch = 0;
        if (2 * padding.y + shape_t.end()[-2] + stride.y >= (shape_k.end()[-2] - 1) * dilation.y + 1)
            shape_r.end()[-2] = (2 * padding.y + shape_t.end()[-2] - (shape_k.end()[-2] - 1) * dilation.y + stride.y - 1) / stride.y;
        else
            shape_r.end()[-2] = 0, size_batch = 0;
        size_t off_t = shape_t.end()[-2] * shape_t.back();
        size_t off_r = shape_r.end()[-2] * shape_r.back();
        size_t ph = shape_t.end()[-2] + 2 * padding.h, pw = shape_t.back() + 2 * padding.w;
        size_t off_p = ph * pw;

        Tensor result = zeros_like(shape_r);

        std::unique_ptr<float64[]> padded = std::make_unique<float64[]>(off_p);
        for (size_t i = 0; i < size_batch; i++)
        {
            copy_2d(b, padded.get(), {shape_t.end()[-2], shape_t.back()}, pw, shape_t.back());

            cross_correlation_2d_impl(
                result.b + i * off_r,
                padded.get(),
                kernel.b,
                {ph, pw},
                {shape_k.end()[-2], shape_k.back()},
                stride, dilation);
        }

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
    Tensor Tensor::crossCorrelation3d(const Tensor &kernel, Tuple3d padding, Tuple3d stride, Tuple3d dilation, PaddingMode pm) const
    {
        return Tensor();
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
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 1)
            shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 1)
            shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 1 < shape_k.size(); i++)
            if (shape_k[i] != 1)
                throw std::length_error("Invalid kernel shape in convolution1d");
        if (shape_k.back() == 0)
            throw std::length_error("Invalid kernel shape in convolution1d");
        if (stride == 0)
            throw std::length_error("Invalid stride in convolution1d");

        for (size_t i = 0; i + 1 < shape_t.size(); i++)
            size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2 * padding + shape_t.back() + stride >= (shape_k.back() - 1) * dilation + 1)
            shape_r.back() = (2 * padding + shape_t.back() - (shape_k.back() - 1) * dilation - 1) / stride + 1;
        else
            shape_r.back() = 0, size_batch = 0;
        size_t off_t = shape_t.back();
        size_t off_r = shape_r.back();
        size_t off_p = shape_t.back() + 2 * padding;

        Tensor result = zeros_like(shape_r);

        std::unique_ptr<float64[]> padded = std::make_unique<float64[]>(off_p);
        std::unique_ptr<float64[]> f_kern = std::make_unique<float64[]>(kernel.size);
        for (size_t r = 0; r < kernel.shape.back(); r++)
            f_kern[r] = kernel.b[kernel.shape.back() - r - 1];
        
        for (size_t i = 0; i < size_batch; i++)
        {
            for (size_t c = 0; c < shape_t.back(); c++)
                padded[c + padding] = b[c];

            cross_correlation_1d_impl(
                result.b + i * off_r,
                padded.get(),
                f_kern.get(),
                shape_t.back() + 2 * padding,
                shape_k.back(),
                stride, dilation);
        }

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
        std::vector<size_t> shape_t = shape;
        std::vector<size_t> shape_k = kernel.shape;
        size_t size_batch = 1;

        while (shape_k.size() < 2)
            shape_k.insert(shape_k.begin(), 1);
        while (shape_t.size() < 2)
            shape_t.insert(shape_t.begin(), 1);

        for (size_t i = 0; i + 2 < shape_k.size(); i++)
            if (shape_k[i] != 1)
                throw std::length_error("Invalid kernel shape in convolution2d");
        if (shape_k.back() == 0 || shape_k.end()[-2] == 0)
            throw std::length_error("Invalid kernel shape in convolution2d");
        if (stride.x == 0 || stride.y == 0)
            throw std::length_error("Invalid stride in convolution2d");

        for (size_t i = 0; i + 2 < shape_t.size(); i++)
            size_batch *= shape_t[i];

        auto shape_r = shape_t;
        if (2 * padding.x + shape_t.back() + stride.x >= (shape_k.back() - 1) * dilation.x + 1)
            shape_r.back() = (2 * padding.x + shape_t.back() - (shape_k.back() - 1) * dilation.x + stride.x - 1) / stride.x;
        else
            shape_r.back() = 0, size_batch = 0;
        if (2 * padding.y + shape_t.end()[-2] + stride.y >= (shape_k.end()[-2] - 1) * dilation.y + 1)
            shape_r.end()[-2] = (2 * padding.y + shape_t.end()[-2] - (shape_k.end()[-2] - 1) * dilation.y + stride.y - 1) / stride.y;
        else
            shape_r.end()[-2] = 0, size_batch = 0;
        size_t off_t = shape_t.end()[-2] * shape_t.back();
        size_t off_r = shape_r.end()[-2] * shape_r.back();
        size_t ph = shape_t.end()[-2] + 2 * padding.h, pw = shape_t.back() + 2 * padding.w;
        size_t off_p = ph * pw;

        Tensor result = zeros_like(shape_r);

        std::unique_ptr<float64[]> padded = std::make_unique<float64[]>(off_p);
        std::unique_ptr<float64[]> f_kern = std::make_unique<float64[]>(shape_k.end()[-2] * shape_k.back());
        for (size_t r = 0; r < shape_k.end()[-2]; r++)
        for (size_t c = 0; c < shape_k.back(); c++)
            f_kern[r*shape_k.back() + c] = kernel.b[(shape_k.end()[-2] - r)*shape_k.back() - c - 1];
        
        for (size_t i = 0; i < size_batch; i++)
        {
            copy_2d(b, padded.get(), {shape_t.end()[-2], shape_t.back()}, pw, shape_t.back());

            cross_correlation_2d_impl(
                result.b + i * off_r,
                padded.get(),
                f_kern.get(),
                {ph, pw},
                {shape_k.end()[-2], shape_k.back()},
                stride, dilation);
        }

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
        return Tensor();
    }

    void reprint(std::ostream &os, const Tensor &t, size_t depth, std::vector<size_t> &index)
    {
        if (depth == 0)
        {
            if (t.size)
                os << t(index);
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
        if(t.onCPU)
        {
            os << "Tensor" << std::endl
               << "Shape: (";
            for (size_t i = 0; i < t.shape.size() - 1; i++)
                os << t.shape[i] << ", ";
            os << t.shape.back() << ")\n";

            std::vector<size_t> v;
            reprint(os, t, t.shape.size(), v);
            os << '\n';
        } else {
            // Print tensor in GPU
        }
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
    Tensor Tensor::axes_reduction(const Tensor &t, size_t d, const float64 init_val)
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
    float64 Tensor::full_reduction(const Tensor &t, const float64 init_val)
    {
        float64 value = init_val;
        for (size_t i = 0; i < t.size; i++)
            fn(value, t.b[i]);
        return value;
    }

    template <float64 (*fn)(float64, float64)>
    void Tensor::broadcast_ew_assign(Tensor &dst, const Tensor &src1, const Tensor &src2,
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
    Tensor Tensor::ew_or_broadcast(const Tensor &t1, const Tensor &t2, const char *err_msg)
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
            for_(size, ld, index, height, b, depth + 1, off*(*ld) + i);
        else
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