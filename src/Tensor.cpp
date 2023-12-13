#include "Tensor.h"

namespace RedFish
{
    void matmul_gotoblas(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
    void matmul_left_T(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2);
    void conv_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation);
    void conv_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation);
    void cross_correlation_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation);
    void cross_correlation_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation);
    void cross_correlation_3d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple3d t_size, Tuple3d kernel_size, Tuple3d stride, Tuple3d dilation);

    /* 
     *      CONSTRUCTORS
     */

    /**
     * @brief Construct a new uninitialized Tensor object with given shape
     * 
     * @param shape 
     */
    Tensor::Tensor(const std::vector<size_t> &shape)
        : shape(shape)
    {
        size = 1;
        for (size_t i = 0; i < shape.size(); i++)
            size *= shape[i];

        if (size)
            b = (b_mem = std::make_unique<float64[]>(size)).get();
    }

    /**
     * @brief  Construct a new uninitialized Tensor object with given shape
     * 
     * @param shape c-like array with tensor shape
     * @param len   shape array length
     */
    Tensor::Tensor(const size_t *shape, size_t len)
        : shape(shape, shape + len)
    {
        size = 1;
        for (size_t i = 0; i < len; i++)
            size *= shape[i];

        if (size)
            b = (b_mem = std::make_unique<float64[]>(size)).get();
    }

    /**
     * @brief Construct a new Tensor object with given shape from a buffer, optionally copying it
     * 
     * @param shape 
     * @param buff 
     * @param copy whether to copy the buffer to a new one or to take buff as the internal memory 
     */
    Tensor::Tensor(const std::vector<size_t> &shape, float64 *buff, bool copy)
        : shape(shape)
    {
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

    /**
     * @brief Copy construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(const Tensor &t)
    {
        this->shape = t.shape;
        this->size = t.size;
        if (size)
            this->b_mem = std::make_unique<float64[]>(size);
        b = b_mem.get();

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];
    }

    /**
     * @brief Move construct a new Tensor object from t
     * 
     * @param t 
     */
    Tensor::Tensor(Tensor &&t)
    {
        this->shape = t.shape;
        this->size = t.size;
        this->b_mem = std::move(t.b_mem);
        b = b_mem.get();
        t.shape = {0};
        t.size = 0;
    }

    /**
     * @brief Construct a new Tensor object from the shape and data as a list
     * 
     * @param shape of the Tensor t be created 
     * @param data as an initializer list 
     */
    Tensor::Tensor(const std::vector<size_t> &shape, std::initializer_list<float64> data)
        : shape(shape)
    {
        if (shape.size() != 0)
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

    /**
     * @brief Construct a new Tensor from an input file stream already opened
     * 
     * @param file std::ifstream& 
     */
    Tensor::Tensor(std::ifstream& file)
    {
        const std::string name = "Tensor";
        char rname[sizeof("Tensor")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in Tensor(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        this->size = 1;
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

        b = (b_mem = std::make_unique<float64[]>(this->size)).get();
        file.read((char*)b, this->size * sizeof(float64));
    }
    
    /* 
     *      CONSTRUCTORS (end)
     */

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
        this->shape = t.shape;
        this->size = t.size;
        if (size)
            this->b_mem = std::make_unique<float64[]>(size);
        else
            this->b_mem = nullptr;
        b = b_mem.get();

        for (size_t i = 0; i < size; i++)
            this->b[i] = t.b[i];

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
        this->shape = t.shape;
        this->size = t.size;
        this->b_mem = std::move(t.b_mem);
        b = b_mem.get();
        t.shape = {0};
        t.size = 0;
        t.b = nullptr;

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

        b_mem = std::make_unique<float64[]>(size);
        b = b_mem.get();
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
        for (size_t i = 0; i < size; i++)
            b[i] = 1;
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
        return op_along_all_axes<fn>(*this, 0.);
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
        return op_along_axes<fn>(*this, d, 0.);
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
        return op_along_all_axes<fn>(*this, -std::numeric_limits<float64>::infinity());
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
        return op_along_axes<fn>(*this, d, -std::numeric_limits<float64>::infinity());
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
        return op_along_all_axes<fn>(*this, std::numeric_limits<float64>::infinity());
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
        return op_along_axes<fn>(*this, d, std::numeric_limits<float64>::infinity());
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
        return op_along_all_axes<fn>(*this, 0.);
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
        return op_along_axes<fn>(*this, d, 0.);
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

        Tensor result(shape_r);
        result.zero();

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

        Tensor result(shape_r);
        result.zero();

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

        Tensor result(shape_r);
        result.zero();

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

        Tensor result(shape_r);
        result.zero();

        std::unique_ptr<float64[]> padded;
        if (padding.x || padding.y)
            padded = std::make_unique<float64[]>(off_p);
        std::unique_ptr<float64[]> f_kern = std::make_unique<float64[]>(kernel.size);
        for (size_t r = 0; r < kernel.shape.end()[-2]; r++)
        for (size_t c = 0; c < kernel.shape.back(); c++)
            f_kern[r*kernel.shape.back() + c] = kernel.b[(kernel.shape.end()[-2] - r)*kernel.shape.back() - c - 1];

        for (size_t i = 0; i < size_batch; i++)
        {
            float64 *ptr = b + i * off_t;
            if (padding.x || padding.y)
            {
                copy_2d(b, padded.get(), {shape_t.end()[-2], shape_t.back()}, pw, shape_t.back());
                ptr = padded.get();
            }

            cross_correlation_2d_impl(
                result.b + i * off_r,
                ptr,
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
        os << "Tensor" << std::endl
           << "Shape: (";
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

}