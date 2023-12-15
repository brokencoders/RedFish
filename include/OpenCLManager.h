#pragma once 
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "CL/cl2.hpp"

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
        MATMULL = 0,
        STRASSEN_MAT_MULL,
        PRINT,
        SET
    };

    class OpenCLManager
    {
    public:
        static inline bool USEOPENGL = false;

        static void showDevices();
        static void init(Platform plat = CPU, size_t device = 0);
        static void free();

        static void createSource(const std::string& src);
        static void createSourceFromFile(const std::string& src_path);
        static void createProgram();

        static size_t createKernel(const std::string& name);

        template <typename T>
        static Buffer createBuffer(size_t size);

        static cl::Buffer& getBuffer(Buffer buffer);

        template <typename T>
        static void loadWriteBuffer(Buffer buffer, size_t size, void* data);
        
        template <typename T>
        static void loadReadBuffer(Buffer buffer, size_t size, void* data);

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
        static inline std::vector<cl::Buffer> buffers;
    };

    template <typename T>
    inline Buffer OpenCLManager::createBuffer(size_t size)
    {
        buffers.emplace_back(context, CL_MEM_READ_WRITE, sizeof(T) * size);
        return buffers.size()-1;
    }

    template <typename T>
    inline void OpenCLManager::loadWriteBuffer(Buffer buffer, size_t size, void* data)
    {
        queue.enqueueWriteBuffer(buffers[buffer], CL_TRUE, 0, sizeof(T) * size, data);
    }

    template <typename T>
    inline void OpenCLManager::loadReadBuffer(Buffer buffer, size_t size, void* data)
    {
        queue.enqueueReadBuffer(buffers[buffer], CL_TRUE, 0, sizeof(T) * size, data);
    }

    template<int I = 0, typename Arg, typename... Args>
    inline void set_args(cl::Kernel& kernel, Arg arg, Args... args)
    {
        kernel.setArg(I, arg);
        if constexpr (sizeof...(args) > 0)
            set_args<I + 1>(kernel, args...);
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

        set_args(kernels[kernel], args...);

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(size[0], size[1]), cl::NDRange(ts[0], ts[1]));
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

        set_args(kernels[kernel], args...);

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(times), cl::NullRange);
        cl_int status = queue.finish();
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Is not a valid kernel!\n");
        }
    }


    void showDevice();
    void device(Platform plat = CPU, size_t device = 0);
} 
