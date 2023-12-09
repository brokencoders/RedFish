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
    using Kernel = size_t;
    using Buffer = size_t;

    class OpenCLManager
    {
    public:
        static void init();
        static void free();

        static void createSource(const std::string& src);
        static void createSourceFromFile(const std::string& src_path);
        static void createProgram();

        static size_t createKernel(const std::string& name);

        template <typename T>
        static size_t createBuffer(size_t size);

        template <typename T>
        static void loadWriteBuffer(size_t buffer, size_t size, void* data);
        
        template <typename T>
        static void loadReadBuffer(size_t buffer, size_t size, void* data);

        static void execute(size_t kernel, std::vector<int> int_arguments, std::vector<size_t> buffers, std::vector<size_t> size, std::vector<size_t> ts);

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
    
    inline void OpenCLManager::init()
    {
        if (cl::Platform::get(&all_platforms) !=  CL_SUCCESS)
            throw std::runtime_error("No OpenCL platforms found. Check OpenCL installation!\n");

        default_platform = all_platforms[0];
        std::cout << "Using platform : " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "Version        : " << default_platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
        
        if (default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices) != CL_SUCCESS)
            throw std::runtime_error("No devices found. Check OpenCL installation!\n");

        default_device = all_devices[0];
        std::cout << "Using device   : " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "Vendor         : " << default_device.getInfo<CL_DEVICE_VENDOR>() << "\n";
        std::cout << "Device version : " << default_device.getInfo<CL_DEVICE_VERSION>() << "\n";
        std::cout << "Driver version : " << default_device.getInfo<CL_DRIVER_VERSION>() << "\n";

        cl_int contextError;
        context = cl::Context({default_device}, nullptr, nullptr, nullptr, &contextError);

        if (contextError != CL_SUCCESS)
            throw std::runtime_error("OpenCL context creation failed");

        cl_int queueError;
        queue = cl::CommandQueue(context,default_device, 0, &queueError);
        if (queueError != CL_SUCCESS)
            throw std::runtime_error("OpenCL command queue creation failed");
    }

    inline void OpenCLManager::free() { }

    inline void OpenCLManager::createSource(const std::string& src)
    {
        sources.push_back({src.c_str(), src.length()});
    }

    inline void OpenCLManager::createSourceFromFile(const std::string& src_path)
    {
        std::string s = loadFile(src_path);
        sources.push_back({s.c_str(), s.length()});
    }
    
    inline void OpenCLManager::createProgram()
    {
        program = cl::Program(context, sources);
        if(program.build({default_device}) != CL_SUCCESS)
            throw std::runtime_error(" Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) + "\n");
    }

    inline size_t OpenCLManager::createKernel(const std::string& name)
    {
        kernels.emplace_back(program, name.c_str());
        return kernels.size() - 1;
    }

    template <typename T>
    inline size_t OpenCLManager::createBuffer(size_t size)
    {
        buffers.emplace_back(context, CL_MEM_READ_WRITE, sizeof(T) * size);
        return buffers.size() - 1;
    }

    template <typename T>
    inline void OpenCLManager::loadWriteBuffer(size_t buffer, size_t size, void* data)
    {
        queue.enqueueWriteBuffer(buffers[buffer], CL_TRUE, 0, sizeof(T) * size, data);
    }

    template <typename T>
    inline void OpenCLManager::loadReadBuffer(size_t buffer, size_t size, void* data)
    {
        queue.enqueueReadBuffer(buffers[buffer], CL_TRUE, 0, sizeof(T) * size, data);
    }

    inline void OpenCLManager::execute(size_t kernel, std::vector<int> int_arguments, std::vector<size_t> arguments, std::vector<size_t> size,  std::vector<size_t> ts)
    {
        if(kernel >= kernels.size())
            throw std::runtime_error("Is not a valid kernel!\n");
        
        size_t expectedArgs = kernels[kernel].getInfo<CL_KERNEL_NUM_ARGS>();
        if (arguments.size() + int_arguments.size() != expectedArgs) {
            throw std::runtime_error("Incorrect number of kernel arguments");
        }

        for (size_t i = 0; i < int_arguments.size(); i++)
            kernels[kernel].setArg(i, int_arguments[i]);

        for (size_t i = 0; i < arguments.size(); i++)
        {
            size_t bufferIndex = arguments[i];
            if (bufferIndex >= buffers.size()) {
                throw std::runtime_error("Invalid buffer index for argument " + std::to_string(i) + "\n");
            }
            kernels[kernel].setArg(i + int_arguments.size(), buffers[arguments[i]]);
        }

        // Find Solution for NDRange
        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(size[0], size[1]), cl::NDRange(ts[0], ts[1]));
        cl_int status = queue.finish();
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Is not a valid kernel!\n");
        }
    }
} 