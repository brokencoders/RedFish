#include "Tensor.h"

#include <chrono>
#include <numeric>
#include <thread>
#include <mutex>
#include <queue>
#include "mkl.h"
#include "mkl_dfti.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "OpenCLManager.h"

namespace RedFish
{
    
    static class ThreadPool { 
    public:
        ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
            : active_threads(0), size(num_threads)
        {
            for (size_t i = 0; i < num_threads; ++i) {
                threads.emplace_back([this, i] {
                    while (true) {
                        std::function<void(size_t)> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            cv.wait(lock, [this] { return !tasks.empty() || stop; });
                            if (stop && tasks.empty()) { return; }
                            task = std::move(tasks.front());
                            tasks.pop();
                            active_threads++;
                        }
                        task(i);
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            active_threads--;
                            thread_done.notify_one();
                        }
                    } 
                }); 
            } 
        }
        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex); 
                stop = true; 
            }

            cv.notify_all(); 
            for (auto& thread : threads)
                thread.join();
        }
        void enqueue(std::function<void(size_t)> task) 
        { 
            { 
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (tasks.size()) thread_done.wait(lock, [this] {return tasks.empty();});
                tasks.emplace(std::move(task)); 
            }
            cv.notify_one();
        }
        void waitToFinish()
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            thread_done.wait(lock, [this] {return active_threads == 0;});
        }
    
    private:
        std::vector<std::thread> threads; 
        std::queue<std::function<void(size_t)>> tasks; 
        std::mutex queue_mutex;
        std::condition_variable cv, thread_done;
        bool stop = false;
        size_t active_threads;
    public:
        const size_t size;
    } thread_pool;


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
    
    static float64* realloc(float64* ptr, size_t size)
    {
        if (!size) { if (ptr) mkl_free(ptr); return nullptr;}
        else return (float64 *)mkl_realloc(ptr, size*sizeof( float64 ));
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

        /* std::function<void(size_t,size_t,size_t)> operation;
        std::vector<size_t> Ashape = shape, Bshape = t.shape;
        if      (Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), Bshape.size() - Ashape.size(), 1);
        else if (Bshape.size() < Ashape.size()) Bshape.insert(Bshape.begin(), Ashape.size() - Bshape.size(), 1);
        if (!broadcastable(Ashape, Bshape))
            throw new std::length_error("Tensor sizes not matching in sum operation");

        std::vector<size_t> Cshape(Ashape.size());

        for (size_t i = 0; i < Ashape.size(); i++)
            Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i];

        Tensor sum(Cshape);
        
        size_t Ainc = Ashape.back() == Cshape.back();
        size_t Binc = Bshape.back() == Cshape.back();
        size_t Coff = 0;

        struct {
            size_t bdc1, bdc2, i;
            size_t Aoff, Boff;
        } stack[10];
        stack[0] = {0,0,(size_t)-1,0,0};
        size_t stack_size = Cshape.size() - 1, depth = 0;

        for (size_t i = 0; i < shape.size(); i++)
        {
            stack[i].bdc1 = (Ashape[i] == Cshape[i]) * ((size_t)-1);
            stack[i].bdc2 = (Bshape[i] == Cshape[i]) * ((size_t)-1);
        }
        
        while (depth != (size_t)-1)
        {
            if (depth == stack_size)
            {
                size_t end = Coff + Cshape[depth];
                size_t Aoff = stack[depth].Aoff * Ashape[depth];
                size_t Boff = stack[depth].Boff * Bshape[depth];
                for (; Coff < end; Coff++, Aoff += Ainc, Boff += Binc)
                    sum.b[Coff] = b[Aoff] + t.b[Boff];
                depth--;
            }
            else
            {
                stack[depth].i++;
                if (stack[depth].i < Cshape[depth])
                {
                    stack[depth + 1].Aoff = stack[depth].Aoff * Ashape[depth] + (stack[depth].i & stack[depth].bdc1);
                    stack[depth + 1].Boff = stack[depth].Boff * Bshape[depth] + (stack[depth].i & stack[depth].bdc2);
                    stack[depth + 1].i = (size_t)-1;
                    depth++;
                }
                else depth--;
            }
        }

        return sum; */
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
        return sliceLastNDims(index, 1);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,l) -> index: (x,...,x)
     * @return DirectTensorView shape: (l)
     */
    const DirectTensorView Tensor::getRow(const std::vector<size_t> &index) const 
    { 
        return sliceLastNDims(index, 1);
    }

    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,h,w) -> index: (x,...,x)
     * @return DirectTensorView shape: (h, w)
     */
    DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index)
    {
        return sliceLastNDims(index, 2);
    }
    
    /**
     * @brief Returns a View of this tensor on index
     * 
     * @param index shape: (x,...,x,h,w) -> index: (x,...,x)
     * @return DirectTensorView shape: (h, w)
     */
    const DirectTensorView Tensor::getMatrix(const std::vector<size_t> &index) const
    {
        return sliceLastNDims(index, 2);
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
        std::vector<size_t> new_shape = shape;
        size_t off = 0;

        if (new_shape.size() < N + index.size())
            new_shape.insert(new_shape.begin(), N+index.size()-new_shape.size(), 1);
        else if (new_shape.size() > N + index.size())
            new_shape.erase(new_shape.begin(), new_shape.begin() + new_shape.size()-N-index.size());

        for (size_t i = 0; i < index.size(); i++)
        {
            off = off*new_shape[i] + index[i];
            if (index[i] >= new_shape[i])
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = index.size(); i < new_shape.size(); i++)
            off = off*new_shape[i];

        new_shape.erase(new_shape.begin(), new_shape.begin() + index.size());

        return DirectTensorView({new_shape.begin(), new_shape.begin() + N}, b + off);
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
        std::vector<size_t> new_shape = shape;
        size_t off = 0;

        if (new_shape.size() < N + index.size())
            new_shape.insert(new_shape.begin(), N+index.size()-new_shape.size(), 1);
        else if (new_shape.size() > N + index.size())
            new_shape.erase(new_shape.begin(), new_shape.begin() + new_shape.size()-N-index.size());

        for (size_t i = 0; i < index.size(); i++)
        {
            off = off*new_shape[i] + index[i];
            if (index[i] >= new_shape[i])
                throw new std::range_error("Out of bound in Tensor sliceLastNDims()");
        }
        for (size_t i = index.size(); i < new_shape.size(); i++)
            off = off*new_shape[i];

        new_shape.erase(new_shape.begin(), new_shape.begin() + index.size());

        return DirectTensorView({new_shape.begin(), new_shape.begin() + N}, b + off);
    }

    /**
     * @brief set new size from shape
     * 
     * @param new_shape 
     */
    void Tensor::resize(const std::vector<size_t> &new_shape)
    {
        this->shape = new_shape;
        size_t new_size = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
            new_size *= new_shape[i];

        if (new_size != size)
            if (onCPU)
            {
                if (new_size < size) b = realloc(b, new_size);
                else
                {
                    dealloc(b);
                    b = alloc(new_size);
                }
            }
            else
            {
                OpenCLManager::destroyBuffer(buffer);
                buffer = OpenCLManager::createBuffer<float64>(new_size);
            }
        
        size = new_size;
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
    Tensor& Tensor::zero()
    {
        if(onCPU)
            std::fill(b, b + size, 0.);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, 0.);
        return *this;
    }

    /**
     * @brief Set tensor with all One
     * 
     */
    Tensor& Tensor::ones()
    {
        if(onCPU)
            std::fill(b, b + size, 1.);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, 1.);
        return *this;
    }

    /**
     * @brief Set all tensor elements to val
     * 
     * @param val
     */
    Tensor& Tensor::constant(float64 val)
    {
        if(onCPU)
            std::fill(b, b + size, val);
        else
            OpenCLManager::setBuffer<float64>(buffer, size, val);
        return *this;
    }

    Tensor& Tensor::linspace(float64 start, float64 stop)
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
        return *this;
    }

    /**
     * @brief Set Tensor values to 1 with probability p and 0 with probability 1-p
     * 
     * @param a lower value 
     * @param b upper value 
     */
    Tensor& Tensor::randBernulli(float64 p)
    {
        std::bernoulli_distribution distribution(p);
        for (size_t i = 0; i < size; i++)
            this->b[i] = distribution(gen);
        return *this;
    }

    /**
     * @brief Set Tensor values to a random uniform value between a and b 
     * 
     * @param a lower value 
     * @param b upper value 
     */
    Tensor& Tensor::randUniform(float64 a, float64 b)
    {
        std::uniform_real_distribution<> distribution(a, b);
        for (size_t i = 0; i < size; i++)
            this->b[i] = distribution(gen);
        return *this;
    }

    /**
     * @brief Set Tensor values randomly according to a gaussian distribution
     * 
     * @param mean 
     * @param std standard deviation
     */
    Tensor& Tensor::randNormal(float64 mean, float64 std)
    {
        std::normal_distribution<double> distribution(mean, std);
        for (size_t i = 0; i < size; i++)
            b[i] = distribution(gen);
        return *this;
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
            thread_pool.enqueue(std::bind(operation, Aoff, Boff, Coff, std::placeholders::_1));
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

    std::vector<size_t> Tensor::broadcast_shape(std::vector<size_t>& Ashape, std::vector<size_t>& Bshape)
    {
        if (Ashape.size() < Bshape.size()) Ashape.insert(Ashape.begin(), Bshape.size() - Ashape.size(), 1);
        if (Ashape.size() > Bshape.size()) Bshape.insert(Bshape.begin(), Ashape.size() - Bshape.size(), 1);
        for (size_t i = 0; i < Ashape.size(); i++)
            if (Ashape[i] != 1 && Bshape[i] != 1 && Ashape[i] != Bshape[i])
                throw std::length_error("Tensor shapes are not broadcastable");
        std::vector<size_t> Cshape(Ashape.size());
        for (size_t i = 0; i < Cshape.size(); i++)
            Cshape[i] = Ashape[i] == 1 ? Bshape[i] : Ashape[i];
        return Cshape;
    }

    void Tensor::broadcast_operation(const std::vector<size_t>& Ashape, const std::vector<size_t>& Bshape, const std::vector<size_t>& Cshape,
                                    const std::function<void(size_t,size_t,size_t)>& operation,
                                    size_t depth, size_t Aoff, size_t Boff, size_t Coff)
    {
        if (Cshape.size() - depth > 8)
        {
            size_t bdc1 = (Ashape[0] != 1) * ((size_t)-1);
            size_t bdc2 = (Bshape[0] != 1) * ((size_t)-1);
            size_t bdc3 = (Cshape[0] != 1) * ((size_t)-1);
            size_t end = bdc3 ? Cshape[0] : (bdc2 ? Bshape[0] : Ashape[0]);
            for (size_t i = 0; i < end; i++)
                broadcast_operation(
                    {Ashape.begin() + 1, Ashape.end()},
                    {Bshape.begin() + 1, Bshape.end()},
                    {Cshape.begin() + 1, Cshape.end()},
                    operation, depth + 1,
                    Aoff * Ashape.front() + (i & bdc1),
                    Boff * Bshape.front() + (i & bdc2),
                    Coff * Cshape.front() + (i & bdc3));
        }
        else
        {
            size_t Ash[8], Bsh[8], Csh[8], bdc1[8], bdc2[8], bdc3[8], end[8];
            std::fill(Ash, Ash + 8-Ashape.size(), 1);
            std::fill(Bsh, Bsh + 8-Bshape.size(), 1);
            std::fill(Csh, Csh + 8-Cshape.size(), 1);
            std::copy(Ashape.begin(), Ashape.end(), Ash + 8-Ashape.size());
            std::copy(Bshape.begin(), Bshape.end(), Bsh + 8-Bshape.size());
            std::copy(Cshape.begin(), Cshape.end(), Csh + 8-Cshape.size());

            for (size_t i = 0; i < 8; i++)
            {
                bdc1[i] = (Ash[i] != 1) * ((size_t)-1);
                bdc2[i] = (Bsh[i] != 1) * ((size_t)-1);
                bdc3[i] = (Csh[i] != 1) * ((size_t)-1);
                end[i] = bdc3[i] ? Csh[i] : (bdc2[i] ? Bsh[i] : Ash[i]);
            }

            for (size_t i = 0; i < end[0]; i++) { size_t Aoff1 = Aoff  * Ash[0] + (i & bdc1[0]), Boff1 = Boff  * Bsh[0] + (i & bdc2[0]), Coff1 = Coff  * Csh[0] + (i & bdc3[0]);
            for (size_t i = 0; i < end[1]; i++) { size_t Aoff2 = Aoff1 * Ash[1] + (i & bdc1[1]), Boff2 = Boff1 * Bsh[1] + (i & bdc2[1]), Coff2 = Coff1 * Csh[1] + (i & bdc3[1]);
            for (size_t i = 0; i < end[2]; i++) { size_t Aoff3 = Aoff2 * Ash[2] + (i & bdc1[2]), Boff3 = Boff2 * Bsh[2] + (i & bdc2[2]), Coff3 = Coff2 * Csh[2] + (i & bdc3[2]);
            for (size_t i = 0; i < end[3]; i++) { size_t Aoff4 = Aoff3 * Ash[3] + (i & bdc1[3]), Boff4 = Boff3 * Bsh[3] + (i & bdc2[3]), Coff4 = Coff3 * Csh[3] + (i & bdc3[3]);
            for (size_t i = 0; i < end[4]; i++) { size_t Aoff5 = Aoff4 * Ash[4] + (i & bdc1[4]), Boff5 = Boff4 * Bsh[4] + (i & bdc2[4]), Coff5 = Coff4 * Csh[4] + (i & bdc3[4]);
            for (size_t i = 0; i < end[5]; i++) { size_t Aoff6 = Aoff5 * Ash[5] + (i & bdc1[5]), Boff6 = Boff5 * Bsh[5] + (i & bdc2[5]), Coff6 = Coff5 * Csh[5] + (i & bdc3[5]);
            for (size_t i = 0; i < end[6]; i++) { size_t Aoff7 = Aoff6 * Ash[6] + (i & bdc1[6]), Boff7 = Boff6 * Bsh[6] + (i & bdc2[6]), Coff7 = Coff6 * Csh[6] + (i & bdc3[6]);
            for (size_t i = 0; i < end[7]; i++)
                operation(Aoff7 * Ash[7] + (i & bdc1[7]), Boff7 * Bsh[7] + (i & bdc2[7]), Coff7 * Csh[7] + (i & bdc3[7]));
            }}}}}}}
        }
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
        if (Ashape.size() < 2) Ashape.insert(Ashape.begin(), 2 - Ashape.size(), 1);
        if (Bshape.size() < 2) Bshape.insert(Bshape.begin(), 2 - Ashape.size(), 1);

        size_t Acols = Ashape.back(); Ashape.pop_back();
        size_t Arows = Ashape.back(); Ashape.pop_back();
        size_t Bcols = Bshape.back(); Bshape.pop_back();
        size_t Brows = Bshape.back(); Bshape.pop_back();

        std::vector<size_t> Cshape = broadcast_shape(Ashape, Bshape);

        size_t Crows = trsp == LEFT  ? Acols : Arows;
        size_t Ccols = trsp == RIGHT ? Brows : Bcols;
        size_t Msize = trsp == LEFT  ? Arows : Acols;
        
        Cshape.push_back(Crows);
        Cshape.push_back(Ccols);
        result.resize(Cshape);
        Cshape.erase(Cshape.end() - 2, Cshape.end());

        auto mult = [tr1 = trsp == LEFT ? CblasTrans : CblasNoTrans, tr2 = trsp == RIGHT ? CblasTrans : CblasNoTrans,
                        Acols, Bcols, Crows, Ccols, A = b, B = t.b, C = result.b, Amsize=Arows*Acols, Bmsize=Brows*Bcols, Cmsize=Crows*Ccols, Msize]
                        (size_t Aoff, size_t Boff, size_t Coff)
        {
            cblas_dgemm(CblasRowMajor, tr1, tr2, Crows, Ccols, Msize, 1, A + Aoff*Amsize, Acols, B + Boff*Bmsize, Bcols, 0, C + Coff*Cmsize, Ccols);
        };

        broadcast_operation(Ashape, Bshape, Cshape, mult);

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

    template <size_t N, bool conv>
    Tensor Tensor::convcorr(const Tensor &kernel,
                            TupleNd<N> padding,
                            TupleNd<N> stride,
                            TupleNd<N> dilation,
                            PaddingMode pm,
                            size_t sum_dimension,
                            bool collapse) const
    {
        auto Ashape = shape, Bshape = kernel.shape;
        if (Ashape.size() < N) Ashape.insert(Ashape.begin(), N-Ashape.size(), 1);
        if (Bshape.size() < N) Bshape.insert(Bshape.begin(), N-Ashape.size(), 1);

        long long ilen[N], iilen[N], klen[N], kklen[N], olen[N], fftlen[N], rs[N+1], cs[N+1];
        size_t isize = 1, iisize = 1, ksize = 1, kksize = 1, osize = 1;
        std::copy(Ashape.end() - N, Ashape.end(), iilen);
        std::copy(Bshape.end() - N, Bshape.end(), kklen);
        Ashape.erase(Ashape.end() - N, Ashape.end());
        Bshape.erase(Bshape.end() - N, Bshape.end());

        for (size_t i = 0; i < N; i++)
        {
            ilen[i]    = iilen[i] + 2*padding[i];
            klen[i]    = kklen[i] * dilation[i] + 1 - dilation[i];
            olen[i]    = klen[i] > ilen[i] ? 0 : (ilen[i] + 1 - klen[i]) / stride[i];
            fftlen[i]  = std::max(ilen[i], klen[i]);
            isize     *= ilen[i];
            ksize     *= klen[i];
            osize     *= olen[i];
            iisize    *= iilen[i];
            kksize    *= kklen[i];
        }

        rs[N] = cs[N] = 1;
        rs[N-1] = (fftlen[N-1]/2+1)*2;
        cs[N-1] =  fftlen[N-1]/2+1;
        for (size_t i = N-1; i > 0; i--)
            rs[i-1] = rs[i]*fftlen[i-1],
            cs[i-1] = cs[i]*fftlen[i-1];
        rs[0] = cs[0] = 0;

        std::vector<size_t> Cshape = broadcast_shape(Ashape, Bshape);
        size_t Asize = 1, Bsize = 1, Csize = 1, fft_stride = 1;
        float64 fft_backward_scale = 1;
        
        if (sum_dimension >= N && sum_dimension < N + Cshape.size()) Cshape.end()[N - sum_dimension - 1] = 1;

        for (size_t i = 0; i < Ashape.size(); i++)
        {
            Asize *= Ashape[i];
            Bsize *= Bshape[i];
            Csize *= Cshape[i];
        }
        
        auto Ishape = Ashape, Kshape = Bshape, Oshape = Cshape;
        for (size_t i = 0; i < N - 1; i++)
        {
            Ishape.push_back(fftlen[i]);
            Kshape.push_back(fftlen[i]);
            Oshape.push_back(fftlen[i]);
            fft_stride *= fftlen[i];
            fft_backward_scale /= fftlen[i];
        }
        Ishape.push_back((fftlen[N-1]/2 + 1)*2);
        Kshape.push_back((fftlen[N-1]/2 + 1)*2);
        Oshape.push_back((fftlen[N-1]/2 + 1)*2);
        fft_stride *= (fftlen[N-1]/2 + 1)*2;
        fft_backward_scale /= fftlen[N-1];
        
        Tensor I(Ishape), K(Kshape), result(Oshape);

        /* 0: prepare inputs */

        for (size_t i = 0; i < Asize; i++)
        {
            /* copy */
            if constexpr (N == 1)
            {
                for (size_t x = 0; x < iilen[N-1]; x++)
                    I.b[i*fft_stride + x+padding[N-1]] = b[i*iisize + x];
                    
                for (size_t x = 0; x < padding[N-1]; x++)
                    I.b[i*fft_stride + x] = 0;
                for (size_t x = padding[N-1] + iilen[N-1]; x < rs[N-1]; x++)
                    I.b[i*fft_stride + x] = 0;

            }
            if constexpr (N == 2)
            {
                for (size_t y = 0; y < iilen[N-2]; y++)
                {
                    for (size_t x = 0; x < iilen[N-1]; x++)
                        I.b[i*fft_stride + (y+padding[N-2])*rs[N-1] + x+padding[N-1]] = b[i*iisize + y*iilen[N-1] + x];

                    for (size_t x = 0; x < padding[N-1]; x++)
                        I.b[i*fft_stride + (y+padding[N-2])*rs[N-1] + x] = 0;
                    for (size_t x = padding[N-1] + iilen[N-1]; x < rs[N-1]; x++)
                        I.b[i*fft_stride + (y+padding[N-2])*rs[N-1] + x] = 0;
                }
                for (size_t x = 0; x < padding[N-2]*rs[N-1]; x++)
                    I.b[i*fft_stride + x] = I.b[i*fft_stride + (padding[N-2] + iilen[N-2])*rs[N-1] + x] = 0;
            }
            if constexpr (N == 3)
            {
                for (size_t z = 0; z < iilen[N-3]; z++)
                {
                    for (size_t y = 0; y < iilen[N-2]; y++)
                    {
                        for (size_t x = 0; x < iilen[N-1]; x++)
                            I.b[i*fft_stride + (z+padding[N-3])*rs[N-2] + (y+padding[N-2])*rs[N-1] + x+padding[N-1]] = b[i*iisize + (z*iilen[N-2] + y)*iilen[N-1] + x];

                        for (size_t x = 0; x < padding[N-1]; x++)
                            I.b[i*fft_stride + (z+padding[N-3])*rs[N-2] + (y+padding[N-2])*rs[N-1] + x] = 0;
                        for (size_t x = padding[N-1] + iilen[N-1]; x < rs[N-1]; x++)
                            I.b[i*fft_stride + (z+padding[N-3])*rs[N-2] + (y+padding[N-2])*rs[N-1] + x] = 0;
                    }
                    for (size_t x = 0; x < padding[N-2]*rs[N-1]; x++)
                        I.b[i*fft_stride + (z+padding[N-3])*rs[N-2] + x] =
                        I.b[i*fft_stride + (z+padding[N-3])*rs[N-2] + (padding[N-2] + iilen[N-2])*rs[N-1] + x] = 0;
                }
                for (size_t x = 0; x < padding[N-3]*rs[N-2]; x++)
                    I.b[i*fft_stride + x] = I.b[i*fft_stride + (padding[N-3] + iilen[N-3])*rs[N-2] + x] = 0;
            }

            /* padding */

        }

        K.zero();
        for (size_t i = 0; i < Bsize; i++)
        {
            if constexpr (N == 1)
                for (size_t x = 0; x < kklen[N-1]; x++)
                    K.b[i*fft_stride + x*dilation[N-1]] = kernel.b[i*kksize + x];
            if constexpr (N == 2)
                for (size_t y = 0; y < kklen[N-2]; y++)
                for (size_t x = 0; x < kklen[N-1]; x++)
                    K.b[i*fft_stride + y*dilation[N-2]*rs[N-1] + x*dilation[N-1]] = kernel.b[i*kksize + y*kklen[N-1] + x];
            if constexpr (N == 3)
                for (size_t z = 0; z < kklen[N-3]; z++)
                for (size_t y = 0; y < kklen[N-2]; y++)
                for (size_t x = 0; x < kklen[N-1]; x++)
                    K.b[i*fft_stride + z*dilation[N-3]*rs[N-2] + y*dilation[N-2]*rs[N-1] + x*dilation[N-1]] = kernel.b[i*kksize + (z*kklen[N-2] + y)*kklen[N-1] + x];
        }

        /* 1: fft on input and kernel */

        DFTI_DESCRIPTOR_HANDLE fft_handle = nullptr;
        if constexpr (N == 1) DftiCreateDescriptor(&fft_handle, DFTI_DOUBLE, DFTI_REAL, N, *fftlen);
        else                  DftiCreateDescriptor(&fft_handle, DFTI_DOUBLE, DFTI_REAL, N,  fftlen);
        DftiSetValue(fft_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(fft_handle, DFTI_INPUT_STRIDES,  rs);
        DftiSetValue(fft_handle, DFTI_OUTPUT_STRIDES, cs);
        DftiSetValue(fft_handle, DFTI_BACKWARD_SCALE, fft_backward_scale);
        DftiSetValue(fft_handle, DFTI_INPUT_DISTANCE, fft_stride);
        DftiSetValue(fft_handle, DFTI_OUTPUT_DISTANCE, fft_stride/2);

        DftiSetValue(fft_handle, DFTI_NUMBER_OF_TRANSFORMS, Asize);
        DftiCommitDescriptor(fft_handle);
        DftiComputeForward(fft_handle, I.b);

        DftiSetValue(fft_handle, DFTI_NUMBER_OF_TRANSFORMS, Bsize);
        DftiCommitDescriptor(fft_handle);
        DftiComputeForward(fft_handle, K.b);

        /* 2: broadcast multiply */

        result.zero();

        if constexpr (conv)
        {
            auto multconv = [fftlen, fft_stride, dest = result.b, i = I.b, k = K.b](size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t x = 0; x < fft_stride; x += 2)
                {
                    dest[Coff*fft_stride + x]   += i[Aoff*fft_stride + x]   * k[Boff*fft_stride + x]
                                                -  i[Aoff*fft_stride + x+1] * k[Boff*fft_stride + x+1];
                    dest[Coff*fft_stride + x+1] += i[Aoff*fft_stride + x+1] * k[Boff*fft_stride + x]
                                                +  i[Aoff*fft_stride + x]   * k[Boff*fft_stride + x+1];
                }
            };
            broadcast_operation(Ashape, Bshape, Cshape, multconv);
        }
        else
        {
            auto multcorr = [fftlen, fft_stride, dest = result.b, i = I.b, k = K.b](size_t Aoff, size_t Boff, size_t Coff)
            {
                for (size_t x = 0; x < fft_stride; x += 2)
                {
                    dest[Coff*fft_stride + x]   += i[Aoff*fft_stride + x]   * k[Boff*fft_stride + x]
                                                +  i[Aoff*fft_stride + x+1] * k[Boff*fft_stride + x+1];
                    dest[Coff*fft_stride + x+1] += i[Aoff*fft_stride + x+1] * k[Boff*fft_stride + x]
                                                -  i[Aoff*fft_stride + x]   * k[Boff*fft_stride + x+1];
                }
            };
            broadcast_operation(Ashape, Bshape, Cshape, multcorr);
        }

        /* 3: ifft result */
        
        DftiSetValue(fft_handle, DFTI_INPUT_STRIDES,  cs);
        DftiSetValue(fft_handle, DFTI_OUTPUT_STRIDES, rs);
        DftiSetValue(fft_handle, DFTI_INPUT_DISTANCE, fft_stride/2);
        DftiSetValue(fft_handle, DFTI_OUTPUT_DISTANCE, fft_stride);
        DftiSetValue(fft_handle, DFTI_NUMBER_OF_TRANSFORMS, Csize);
        DftiCommitDescriptor(fft_handle);
        DftiComputeBackward(fft_handle, result.b);
        
        DftiFreeDescriptor(&fft_handle);
        
        /* 4: sqeeze fft results */

        for (size_t i = 0; i < Csize; i++)
            if constexpr (conv)
            {
                if constexpr (N == 1)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + x] = result.b[i*fft_stride + x*stride[N-1] + klen[N-1] - 1];
                else if constexpr (N == 2)
                    for (size_t y = 0; y < olen[N-2]; y++)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + y*olen[N-1] + x] = result.b[i*fft_stride + (y*stride[N-2] + klen[N-2] - 1)*rs[N-1] + x*stride[N-1] + klen[N-1] - 1];
                else if constexpr (N == 3)
                    for (size_t z = 0; z < olen[N-3]; z++)
                    for (size_t y = 0; y < olen[N-2]; y++)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + (z*olen[N-2] + y)*olen[N-1] + x] = result.b[i*fft_stride + (z*stride[N-3] + klen[N-2] - 1)*rs[N-2] + (y*stride[N-2] + klen[N-2] - 1)*rs[N-1] + x*stride[N-1] + klen[N-1] - 1];
            }
            else
            {
                if constexpr (N == 1)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + x] = result.b[i*fft_stride + x*stride[N-1]];
                else if constexpr (N == 2)
                    for (size_t y = 0; y < olen[N-2]; y++)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + y*olen[N-1] + x] = result.b[i*fft_stride + y*stride[N-2]*rs[N-1] + x*stride[N-1]];
                else if constexpr (N == 3)
                    for (size_t z = 0; z < olen[N-3]; z++)
                    for (size_t y = 0; y < olen[N-2]; y++)
                    for (size_t x = 0; x < olen[N-1]; x++)
                        result.b[i*osize + (z*olen[N-2] + y)*olen[N-1] + x] = result.b[i*fft_stride + z*stride[N-3]*rs[N-2] + y*stride[N-2]*rs[N-1] + x*stride[N-1]];
            }

        Cshape.insert(Cshape.end(), olen, olen + N);
        result.resize(Cshape);
        /* Cshape.erase(Cshape.end() - N, Cshape.end()); */

        if (sum_dimension < N)
            result = result.sum(sum_dimension, collapse);
        else if (collapse && sum_dimension < result.shape.size())
            result.shape.erase(result.shape.end() - 1 - sum_dimension);

        return result;
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
    Tensor Tensor::correlation1d(const Tensor &kernel,
                                 TupleNd<1> padding,
                                 TupleNd<1> stride,
                                 TupleNd<1> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<1, false>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
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
    Tensor Tensor::correlation2d(const Tensor &kernel,
                                 TupleNd<2> padding,
                                 TupleNd<2> stride,
                                 TupleNd<2> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<2, false>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
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
    Tensor Tensor::correlation3d(const Tensor &kernel,
                                 TupleNd<3> padding,
                                 TupleNd<3> stride,
                                 TupleNd<3> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<3, false>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
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
    Tensor Tensor::convolution1d(const Tensor &kernel,
                                 TupleNd<1> padding,
                                 TupleNd<1> stride,
                                 TupleNd<1> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<1, true>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
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
    Tensor Tensor::convolution2d(const Tensor &kernel,
                                 TupleNd<2> padding,
                                 TupleNd<2> stride,
                                 TupleNd<2> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<2, true>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
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
    Tensor Tensor::convolution3d(const Tensor &kernel,
                                 TupleNd<3> padding,
                                 TupleNd<3> stride,
                                 TupleNd<3> dilation,
                                 PaddingMode pm,
                                 size_t sum_dimension,
                                 bool collapse) const
    {
        return convcorr<3, true>(kernel, padding, stride, dilation, pm, sum_dimension, collapse);
    }

    void reprint(std::ostream &os, const Tensor &t, size_t depth, std::vector<size_t> &index)
    {
        if (depth == 0)
        {
            if (t.size)
            {
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
        auto p1 = s1.end();
        auto p2 = s2.end();

        while (p1 != s1.begin() && p2 != s2.begin())
        {
            p1--, p2--;
            if (*p1 != *p2 && *p1 != 1 && *p2 != 1)
                return false;
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

}