#pragma once 
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "CL/opencl.hpp"

#include "Data.h"

namespace RedFish
{
    using Buffer = size_t;

    enum Platform 
    {
        CPU = 0,
        AMD,
        INTEL,
        NVIDIA,
        ARM
    };

    enum Kernel : size_t
    {
        MATMUL = 0,
        STRASSEN_MAT_MUL,
        T_SCALAR_ADD,
        T_TENSOR_ADD,
        T_SCALAR_SUB,
        T_TENSOR_SUB,
        T_SCALAR_MUL,
        T_TENSOR_MUL,
        T_SCALAR_DIV,
        T_TENSOR_DIV,
        T_MINUS,
        T_SCALAR_TENSOR_SUB,
        T_SCALAR_TENSOR_DIV,
        T_TENSOR_EQUALS,
        T_SCALAR_EQUALS,
        T_TENSOR_GT_EQUALS,
        T_SCALAR_GT_EQUALS,
        T_TENSOR_LT_EQUALS,
        T_SCALAR_LT_EQUALS,
        T_TENSOR_GT,
        T_SCALAR_GT,
        T_TENSOR_LT,
        T_SCALAR_LT,
        /* Broadcast ops */
        T_TENSOR_ADD_BRODCAST_N0B1N0,
        T_TENSOR_ADD_BRODCAST_N0B1B2,
        T_TENSOR_ADD_BRODCAST_N0B2N0,
        T_TENSOR_ADD_BRODCAST_N0B2B1,
        T_TENSOR_ADD_BRODCAST_B1N0B1,
        T_TENSOR_ADD_BRODCAST_B1N0B2,
        T_TENSOR_ADD_BRODCAST_B1B2N0,
        T_TENSOR_ADD_BRODCAST_B1B2B1,
        T_TENSOR_ADD_BRODCAST_B2N0B1,
        T_TENSOR_ADD_BRODCAST_B2N0B2,
        T_TENSOR_ADD_BRODCAST_B2B1N0,
        T_TENSOR_ADD_BRODCAST_B2B1B2,
        T_TENSOR_SUB_BRODCAST_N0B1N0,
        T_TENSOR_SUB_BRODCAST_N0B1B2,
        T_TENSOR_SUB_BRODCAST_N0B2N0,
        T_TENSOR_SUB_BRODCAST_N0B2B1,
        T_TENSOR_SUB_BRODCAST_B1N0B1,
        T_TENSOR_SUB_BRODCAST_B1N0B2,
        T_TENSOR_SUB_BRODCAST_B1B2N0,
        T_TENSOR_SUB_BRODCAST_B1B2B1,
        T_TENSOR_SUB_BRODCAST_B2N0B1,
        T_TENSOR_SUB_BRODCAST_B2N0B2,
        T_TENSOR_SUB_BRODCAST_B2B1N0,
        T_TENSOR_SUB_BRODCAST_B2B1B2,
        T_TENSOR_MUL_BRODCAST_N0B1N0,
        T_TENSOR_MUL_BRODCAST_N0B1B2,
        T_TENSOR_MUL_BRODCAST_N0B2N0,
        T_TENSOR_MUL_BRODCAST_N0B2B1,
        T_TENSOR_MUL_BRODCAST_B1N0B1,
        T_TENSOR_MUL_BRODCAST_B1N0B2,
        T_TENSOR_MUL_BRODCAST_B1B2N0,
        T_TENSOR_MUL_BRODCAST_B1B2B1,
        T_TENSOR_MUL_BRODCAST_B2N0B1,
        T_TENSOR_MUL_BRODCAST_B2N0B2,
        T_TENSOR_MUL_BRODCAST_B2B1N0,
        T_TENSOR_MUL_BRODCAST_B2B1B2,
        T_TENSOR_DIV_BRODCAST_N0B1N0,
        T_TENSOR_DIV_BRODCAST_N0B1B2,
        T_TENSOR_DIV_BRODCAST_N0B2N0,
        T_TENSOR_DIV_BRODCAST_N0B2B1,
        T_TENSOR_DIV_BRODCAST_B1N0B1,
        T_TENSOR_DIV_BRODCAST_B1N0B2,
        T_TENSOR_DIV_BRODCAST_B1B2N0,
        T_TENSOR_DIV_BRODCAST_B1B2B1,
        T_TENSOR_DIV_BRODCAST_B2N0B1,
        T_TENSOR_DIV_BRODCAST_B2N0B2,
        T_TENSOR_DIV_BRODCAST_B2B1N0,
        T_TENSOR_DIV_BRODCAST_B2B1B2,
        T_TENSOR_EQUALS_BRODCAST_N0B1N0,
        T_TENSOR_EQUALS_BRODCAST_N0B1B2,
        T_TENSOR_EQUALS_BRODCAST_N0B2N0,
        T_TENSOR_EQUALS_BRODCAST_N0B2B1,
        T_TENSOR_EQUALS_BRODCAST_B1N0B1,
        T_TENSOR_EQUALS_BRODCAST_B1N0B2,
        T_TENSOR_EQUALS_BRODCAST_B1B2N0,
        T_TENSOR_EQUALS_BRODCAST_B1B2B1,
        T_TENSOR_EQUALS_BRODCAST_B2N0B1,
        T_TENSOR_EQUALS_BRODCAST_B2N0B2,
        T_TENSOR_EQUALS_BRODCAST_B2B1N0,
        T_TENSOR_EQUALS_BRODCAST_B2B1B2,
        T_TENSOR_GT_EQUALS_BRODCAST_N0B1N0,
        T_TENSOR_GT_EQUALS_BRODCAST_N0B1B2,
        T_TENSOR_GT_EQUALS_BRODCAST_N0B2N0,
        T_TENSOR_GT_EQUALS_BRODCAST_N0B2B1,
        T_TENSOR_GT_EQUALS_BRODCAST_B1N0B1,
        T_TENSOR_GT_EQUALS_BRODCAST_B1N0B2,
        T_TENSOR_GT_EQUALS_BRODCAST_B1B2N0,
        T_TENSOR_GT_EQUALS_BRODCAST_B1B2B1,
        T_TENSOR_GT_EQUALS_BRODCAST_B2N0B1,
        T_TENSOR_GT_EQUALS_BRODCAST_B2N0B2,
        T_TENSOR_GT_EQUALS_BRODCAST_B2B1N0,
        T_TENSOR_GT_EQUALS_BRODCAST_B2B1B2,
        T_TENSOR_LT_EQUALS_BRODCAST_N0B1N0,
        T_TENSOR_LT_EQUALS_BRODCAST_N0B1B2,
        T_TENSOR_LT_EQUALS_BRODCAST_N0B2N0,
        T_TENSOR_LT_EQUALS_BRODCAST_N0B2B1,
        T_TENSOR_LT_EQUALS_BRODCAST_B1N0B1,
        T_TENSOR_LT_EQUALS_BRODCAST_B1N0B2,
        T_TENSOR_LT_EQUALS_BRODCAST_B1B2N0,
        T_TENSOR_LT_EQUALS_BRODCAST_B1B2B1,
        T_TENSOR_LT_EQUALS_BRODCAST_B2N0B1,
        T_TENSOR_LT_EQUALS_BRODCAST_B2N0B2,
        T_TENSOR_LT_EQUALS_BRODCAST_B2B1N0,
        T_TENSOR_LT_EQUALS_BRODCAST_B2B1B2,
        T_TENSOR_GT_BRODCAST_N0B1N0,
        T_TENSOR_GT_BRODCAST_N0B1B2,
        T_TENSOR_GT_BRODCAST_N0B2N0,
        T_TENSOR_GT_BRODCAST_N0B2B1,
        T_TENSOR_GT_BRODCAST_B1N0B1,
        T_TENSOR_GT_BRODCAST_B1N0B2,
        T_TENSOR_GT_BRODCAST_B1B2N0,
        T_TENSOR_GT_BRODCAST_B1B2B1,
        T_TENSOR_GT_BRODCAST_B2N0B1,
        T_TENSOR_GT_BRODCAST_B2N0B2,
        T_TENSOR_GT_BRODCAST_B2B1N0,
        T_TENSOR_GT_BRODCAST_B2B1B2,
        T_TENSOR_LT_BRODCAST_N0B1N0,
        T_TENSOR_LT_BRODCAST_N0B1B2,
        T_TENSOR_LT_BRODCAST_N0B2N0,
        T_TENSOR_LT_BRODCAST_N0B2B1,
        T_TENSOR_LT_BRODCAST_B1N0B1,
        T_TENSOR_LT_BRODCAST_B1N0B2,
        T_TENSOR_LT_BRODCAST_B1B2N0,
        T_TENSOR_LT_BRODCAST_B1B2B1,
        T_TENSOR_LT_BRODCAST_B2N0B1,
        T_TENSOR_LT_BRODCAST_B2N0B2,
        T_TENSOR_LT_BRODCAST_B2B1N0,
        T_TENSOR_LT_BRODCAST_B2B1B2,
    };

    class OpenCLManager
    {
    public:

        static void showDevices();
        static void init(Platform plat = CPU, size_t device = 0);
        static void free();

        static void createSource(const std::string& src);
        static void createSourceFromFile(const std::string& src_path);
        static void createProgram();

        static size_t createKernel(const std::string& name);

        template <typename T>
        static Buffer createBuffer(size_t size);

        static void destroyBuffer(Buffer& buffer);

        static cl::Buffer& getBuffer(Buffer buffer);

        template <typename T>
        static cl::Buffer getSubBuffer(Buffer buffer, size_t offset, size_t size);

        template <typename T>
        static void loadWriteBuffer(Buffer buffer, size_t size, void* data, size_t offset = 0);
        
        template <typename T>
        static void loadReadBuffer(Buffer buffer, size_t size, void* data, size_t offset = 0);

        template <typename T>
        static void copyBuffer(Buffer from, Buffer to, size_t size);

        template <typename T>
        static void setBuffer(Buffer buffer, size_t size, T value);

        template<typename... Args>
        static void execute(size_t kernel, std::vector<size_t> size, std::vector<size_t> ts, Args... args);

        template<typename... Args>
        static void execute(size_t kernel, size_t times, Args... args);

        static void execute(size_t kernel, size_t times);

    private:
        OpenCLManager() = delete;

        static inline std::vector<cl::Platform> all_platforms;
        static inline std::vector<cl::Device> all_devices;
        static inline cl::Platform default_platform;
        static inline cl::Device default_device;
        static inline cl::Context context;
        static inline cl::Program program;
        static inline cl::CommandQueue queue;

        static inline cl::Program::Sources sources;
 
        static inline std::vector<cl::Kernel> kernels;
        static inline std::map<size_t, cl::Buffer> buffers;
    };

    template <typename T>
    inline Buffer OpenCLManager::createBuffer(size_t size)
    {
        if (size == 0) return 0;
        size_t last = buffers.size() ? buffers.rbegin()->first : 0;
        buffers.emplace(last + 1, cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * size));
        return last + 1;
    }

    template <typename T>
    cl::Buffer OpenCLManager::getSubBuffer(Buffer buffer, size_t offset, size_t size)
    {
        cl_buffer_region region = { offset, size * sizeof(T) };
        return buffers.at(buffer).createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
    }

    template <typename T>
    inline void OpenCLManager::loadWriteBuffer(Buffer buffer, size_t size, void* data, size_t offset)
    {
        if (buffer)
            queue.enqueueWriteBuffer(buffers.at(buffer), CL_TRUE, sizeof(T) * offset, sizeof(T) * size, data);
    }

    template <typename T>
    inline void OpenCLManager::loadReadBuffer(Buffer buffer, size_t size, void* data, size_t offset)
    {
        if (buffer)
            queue.enqueueReadBuffer(buffers.at(buffer), CL_TRUE, sizeof(T) * offset, sizeof(T) * size, data);
    }

    template <typename T>
    inline void OpenCLManager::copyBuffer(Buffer from, Buffer to, size_t size)
    {
        if (from && to)
            queue.enqueueCopyBuffer(buffers.at(from), buffers.at(to), 0, 0, sizeof(T) * size);
    }

    template <typename T>
    inline void OpenCLManager::setBuffer(Buffer buffer, size_t size, T value)
    {
        if (buffer)
            queue.enqueueFillBuffer(buffers.at(buffer), value, 0, sizeof(T) * size);
    }

    template<typename... Args>
    inline void OpenCLManager::execute(size_t kernel, std::vector<size_t> size,  std::vector<size_t> ts, Args... args)
    {
        if(kernel >= kernels.size())
            throw std::runtime_error("Is not a valid kernel!\n");
        
        size_t expectedArgs = kernels[kernel].getInfo<CL_KERNEL_NUM_ARGS>();
        if (sizeof...(Args) != expectedArgs) {
            throw std::runtime_error("Incorrect number of kernel arguments");
        }

        size_t i = 0;
        ([&] { kernels[kernel].setArg(i++, args); } (), ...);

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(size[0], size[1], size[2]), cl::NDRange(ts[0], ts[1], ts[2]));
        cl_int status = queue.finish();
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Is not a valid kernel!\n");
        }
    }

    template<typename... Args>
    inline void OpenCLManager::execute(size_t kernel, size_t times, Args... args)
    {
        if(kernel >= kernels.size())
            throw std::runtime_error("Is not a valid kernel!\n");
        
        size_t expectedArgs = kernels[kernel].getInfo<CL_KERNEL_NUM_ARGS>();
        if (sizeof...(Args) != expectedArgs) {
            throw std::runtime_error("Incorrect number of kernel arguments");
        }

        size_t i = 0;
        ([&] { kernels[kernel].setArg(i++, args); } (), ...);

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(times), cl::NullRange);
        cl_int status = queue.finish();
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Is not a valid kernel!\n");
        }
    }


    void showDevice();
    void device(Platform plat = CPU, size_t device = 0);
} 
