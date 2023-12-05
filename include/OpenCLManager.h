#pragma once 
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
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

        static void execute(size_t kernel, std::vector<size_t> buffers);

    private:
        OpenCLManager() = delete;
        static inline std::vector<cl::Platform> all_platforms;
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
        // get all platforms (drivers)
        cl::Platform::get(&all_platforms);

        if (all_platforms.empty())
            throw std::runtime_error("No OpenCL platforms found. Check OpenCL installation!\n");

        default_platform = all_platforms[0];
        std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        // get default device of the default platform
        std::vector<cl::Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (all_devices.size() == 0)
            throw std::runtime_error(" No devices found. Check OpenCL installation!\n");

        default_device = all_devices[0];
        std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

        context = cl::Context({default_device});

        //create queue to which we will push commands for the device.
        queue = cl::CommandQueue(context,default_device);
    }

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

    inline void OpenCLManager::execute(size_t kernel, std::vector<size_t> arguments)
    {
        if(kernel >= kernels.size())
            throw std::runtime_error("Is not a valid kernel!\n");
        
        size_t expectedArgs = kernels[kernel].getInfo<CL_KERNEL_NUM_ARGS>();
        if (arguments.size() != expectedArgs) {
            throw std::runtime_error("Incorrect number of kernel arguments");
        }

        for (size_t i = 0; i < arguments.size(); i++)
            kernels[kernel].setArg(i, buffers[arguments[i]]);

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(10), cl::NullRange);
        queue.finish();
    }
} 
