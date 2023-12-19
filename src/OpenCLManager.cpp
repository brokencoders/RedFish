#include "OpenCLManager.h"

namespace RedFish 
{
    void OpenCLManager::showDevices()
    {
        #ifdef __linux__
        std::cout << "--------------------CPU--------------------\n";
        std::ifstream cpuinfo("/proc/cpuinfo");

        if (!cpuinfo.is_open())
            std::cout << "Can't read cpu info" << std::endl;

        std::string line;
        std::string modelName, cpuMHz, cacheSize;

        while (std::getline(cpuinfo, line)) {
            // Find lines containing CPU information
            if (line.find("model name") != std::string::npos)
                std::cout << "Model Name    : " <<  line.substr(line.find(":") + 2) << "\n";
            else if (line.find("cpu MHz") != std::string::npos)
                std::cout << "Frequency     : " << line.substr(line.find(":") + 2)  << " MHz" << std::endl;
            else if (line.find("cache size") != std::string::npos)
            {
                std::cout << "Cache Size    : " << line.substr(line.find(":") + 2) << "\n";
                break;
            }
        }

        cpuinfo.close();
        #endif 

        if (cl::Platform::get(&all_platforms) !=  CL_SUCCESS)
            throw std::runtime_error("No OpenCL platforms found. Check OpenCL installation!\n");

        std::cout << "-----------------Platforms-----------------\n";
        for(auto platform : all_platforms)
        {
            std::cout << "Platform       : " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
            std::cout << "Version        : " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
        
            if (platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices) != CL_SUCCESS)
                throw std::runtime_error("No devices found. Check OpenCL installation!\n");

            for (size_t i = 0; i < all_devices.size(); i++)
            {                
                auto device = all_devices[i];
                std::cout << "-----------------Device " << i << "------------------\n";
                std::cout << "Device         : " << device.getInfo<CL_DEVICE_NAME>() << "\n";
                std::cout << "Vendor         : " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
                std::cout << "Device version : " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
                std::cout << "Driver version : " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
                std::cout << "Address bits   : " << device.getInfo<CL_DEVICE_ADDRESS_BITS>() << "\n";
                std::cout << "Group Size     : " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
            }
            std::cout << "-------------------------------------------\n";
        }
    }

    void OpenCLManager::init(Platform plat, size_t device_index)
    {
        if(plat == Platform::CPU)
            return;

        if (cl::Platform::get(&all_platforms) !=  CL_SUCCESS)
            throw std::runtime_error("No OpenCL platforms found. Check OpenCL installation!\n");

        for(size_t i = 0; i < all_platforms.size(); i++)
        {
            auto platform = all_platforms[i];
            if (platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices) != CL_SUCCESS)
                throw std::runtime_error("No devices found. Check OpenCL installation!\n");
    
            auto device = all_devices[device_index];

            std::string vendor_name = device.getInfo<CL_DEVICE_VENDOR>();
            std::transform(vendor_name.begin(), vendor_name.end(), vendor_name.begin(), ::toupper);
            switch(plat)
            {
                case Platform::AMD:
                    if(vendor_name.find("AMD") != std::string::npos)
                    {
                        default_platform = platform;
                        default_device = device;
                        std::cout << "-------------------------------------------\n";
                        std::cout << "       Using AMD DEVICE number " << device_index << "\n"; 
                        std::cout << "-------------------------------------------\n";
                    }
                    break;
                case Platform::INTEL:
                    if(vendor_name.find("INTEL") != std::string::npos)
                    {
                        default_platform = platform;
                        default_device = device;
                        std::cout << "-------------------------------------------\n";
                        std::cout << "       Using INTEL DEVICE number " << device_index << "\n"; 
                        std::cout << "-------------------------------------------\n";
                    }
                    break;
                case Platform::NVIDIA:
                    if(vendor_name.find("NVIDIA") != std::string::npos)
                    {
                        default_platform = platform;
                        default_device = device;
                        std::cout << "-------------------------------------------\n";
                        std::cout << "       Using NVIDIA DEVICE number " << device_index << "\n"; 
                        std::cout << "-------------------------------------------\n";
                    }
                    break;
                default:
                    throw std::runtime_error("Device not found!");
            }
        }

        cl_int contextError;
        context = cl::Context({default_device}, nullptr, nullptr, nullptr, &contextError);
        if (contextError != CL_SUCCESS)
            throw std::runtime_error("OpenCL context creation failed");

        cl_int queueError;
        queue = cl::CommandQueue(context,default_device, 0, &queueError);
        if (queueError != CL_SUCCESS)
            throw std::runtime_error("OpenCL command queue creation failed");
    }

    void OpenCLManager::free() { }

    void OpenCLManager::destroyBuffer(Buffer& buffer)
    {
        if (buffer)
        {
            buffers.erase(buffer);
            buffer = 0;
        }
    }

    void OpenCLManager::createSource(const std::string& src)
    {
        sources.push_back({src.c_str(), src.length()});
        sources.push_back("\n");
    }

    void OpenCLManager::createSourceFromFile(const std::string& src_path)
    {
        std::string s = loadFile(src_path);
        sources.push_back({s.c_str(), s.length()});
        sources.push_back("\n");
    }
    
    void OpenCLManager::createProgram()
    {
        program = cl::Program(context, sources);
        if(program.build({default_device}) != CL_SUCCESS)
            throw std::runtime_error(" Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) + "\n");
        std::cout << "Build done!\n";
    }

    size_t OpenCLManager::createKernel(const std::string& name)
    {
        kernels.emplace_back(program, name.c_str());
        return kernels.size() - 1;
    }

    cl::Buffer& OpenCLManager::getBuffer(Buffer buffer)
    {
        return buffers.at(buffer);
    }

    void OpenCLManager::execute(size_t kernel, size_t times)
    {
        if(kernel >= kernels.size())
            throw std::runtime_error("Is not a valid kernel!\n");

        std::cout << "WE" << std::endl;

        queue.enqueueNDRangeKernel(kernels[kernel], cl::NullRange, cl::NDRange(times), cl::NullRange);
        cl_int status = queue.finish();
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Is not a valid kernel!\n");
        }
    }

    void showDevice()
    {
        OpenCLManager::showDevices();
    }

    void device(Platform plat, size_t device)
    {
        OpenCLManager::init(plat, device);
        
        // All Source Files
        OpenCLManager::createSourceFromFile("../src/kernels/TensorBasic.cl");
        OpenCLManager::createSourceFromFile("../src/kernels/TensorBasicBroadcast.cl");
        OpenCLManager::createSourceFromFile("../src/kernels/TensorMul.cl");
        OpenCLManager::createSourceFromFile("../src/kernels/TensorStrassenMul.cl");

        // Build 
        OpenCLManager::createProgram();

        // Create all Kernels
        if (OpenCLManager::createKernel("tensor_tensor_math_mul")           != Kernel::MATMUL               ||
            OpenCLManager::createKernel("tensor_tensor_strassen_math_mul")  != Kernel::STRASSEN_MAT_MUL     ||
            OpenCLManager::createKernel("tensor_print")                     != Kernel::PRINT                ||
            OpenCLManager::createKernel("tensor_scalar_add")                != Kernel::T_SCALAR_ADD         ||
            OpenCLManager::createKernel("tensor_tensor_add")                != Kernel::T_TENSOR_ADD         ||
            OpenCLManager::createKernel("tensor_scalar_sub")                != Kernel::T_SCALAR_SUB         ||
            OpenCLManager::createKernel("tensor_tensor_sub")                != Kernel::T_TENSOR_SUB         ||
            OpenCLManager::createKernel("tensor_scalar_mul")                != Kernel::T_SCALAR_MUL         ||
            OpenCLManager::createKernel("tensor_tensor_mul")                != Kernel::T_TENSOR_MUL         ||
            OpenCLManager::createKernel("tensor_scalar_div")                != Kernel::T_SCALAR_DIV         ||
            OpenCLManager::createKernel("tensor_tensor_div")                != Kernel::T_TENSOR_DIV         ||
            OpenCLManager::createKernel("tensor_minus")                     != Kernel::T_MINUS              ||
            OpenCLManager::createKernel("scalar_tensor_sub")                != Kernel::T_SCALAR_TENSOR_SUB  ||
            OpenCLManager::createKernel("scalar_tensor_div")                != Kernel::T_SCALAR_TENSOR_DIV  ||
            OpenCLManager::createKernel("tensor_tensor_equals")             != Kernel::T_TENSOR_EQUALS      ||
            OpenCLManager::createKernel("tensor_scalar_equals")             != Kernel::T_SCALAR_EQUALS      ||
            OpenCLManager::createKernel("tensor_tensor_gt_equals")          != Kernel::T_TENSOR_GT_EQUALS   ||
            OpenCLManager::createKernel("tensor_scalar_gt_equals")          != Kernel::T_SCALAR_GT_EQUALS   ||
            OpenCLManager::createKernel("tensor_tensor_lt_equals")          != Kernel::T_TENSOR_LT_EQUALS   ||
            OpenCLManager::createKernel("tensor_scalar_lt_equals")          != Kernel::T_SCALAR_LT_EQUALS   ||
            OpenCLManager::createKernel("tensor_tensor_gt")                 != Kernel::T_TENSOR_GT          ||
            OpenCLManager::createKernel("tensor_scalar_gt")                 != Kernel::T_SCALAR_GT          ||
            OpenCLManager::createKernel("tensor_tensor_lt")                 != Kernel::T_TENSOR_LT          ||
            OpenCLManager::createKernel("tensor_scalar_lt")                 != Kernel::T_SCALAR_LT          ||
            /* Brodcast ops */
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_n0_b1_n0") != Kernel::T_TENSOR_ADD_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_n0_b1_b2") != Kernel::T_TENSOR_ADD_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_n0_b2_n0") != Kernel::T_TENSOR_ADD_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_n0_b2_b1") != Kernel::T_TENSOR_ADD_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b1_n0_b1") != Kernel::T_TENSOR_ADD_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b1_n0_b2") != Kernel::T_TENSOR_ADD_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b1_b2_n0") != Kernel::T_TENSOR_ADD_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b1_b2_b1") != Kernel::T_TENSOR_ADD_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b2_n0_b1") != Kernel::T_TENSOR_ADD_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b2_n0_b2") != Kernel::T_TENSOR_ADD_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b2_b1_n0") != Kernel::T_TENSOR_ADD_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_add_b2_b1_b2") != Kernel::T_TENSOR_ADD_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_n0_b1_n0") != Kernel::T_TENSOR_SUB_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_n0_b1_b2") != Kernel::T_TENSOR_SUB_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_n0_b2_n0") != Kernel::T_TENSOR_SUB_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_n0_b2_b1") != Kernel::T_TENSOR_SUB_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b1_n0_b1") != Kernel::T_TENSOR_SUB_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b1_n0_b2") != Kernel::T_TENSOR_SUB_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b1_b2_n0") != Kernel::T_TENSOR_SUB_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b1_b2_b1") != Kernel::T_TENSOR_SUB_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b2_n0_b1") != Kernel::T_TENSOR_SUB_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b2_n0_b2") != Kernel::T_TENSOR_SUB_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b2_b1_n0") != Kernel::T_TENSOR_SUB_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_sub_b2_b1_b2") != Kernel::T_TENSOR_SUB_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_n0_b1_n0") != Kernel::T_TENSOR_MUL_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_n0_b1_b2") != Kernel::T_TENSOR_MUL_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_n0_b2_n0") != Kernel::T_TENSOR_MUL_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_n0_b2_b1") != Kernel::T_TENSOR_MUL_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b1_n0_b1") != Kernel::T_TENSOR_MUL_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b1_n0_b2") != Kernel::T_TENSOR_MUL_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b1_b2_n0") != Kernel::T_TENSOR_MUL_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b1_b2_b1") != Kernel::T_TENSOR_MUL_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b2_n0_b1") != Kernel::T_TENSOR_MUL_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b2_n0_b2") != Kernel::T_TENSOR_MUL_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b2_b1_n0") != Kernel::T_TENSOR_MUL_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_mul_b2_b1_b2") != Kernel::T_TENSOR_MUL_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_n0_b1_n0") != Kernel::T_TENSOR_DIV_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_n0_b1_b2") != Kernel::T_TENSOR_DIV_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_n0_b2_n0") != Kernel::T_TENSOR_DIV_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_n0_b2_b1") != Kernel::T_TENSOR_DIV_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b1_n0_b1") != Kernel::T_TENSOR_DIV_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b1_n0_b2") != Kernel::T_TENSOR_DIV_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b1_b2_n0") != Kernel::T_TENSOR_DIV_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b1_b2_b1") != Kernel::T_TENSOR_DIV_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b2_n0_b1") != Kernel::T_TENSOR_DIV_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b2_n0_b2") != Kernel::T_TENSOR_DIV_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b2_b1_n0") != Kernel::T_TENSOR_DIV_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_div_b2_b1_b2") != Kernel::T_TENSOR_DIV_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_n0_b1_n0") != Kernel::T_TENSOR_EQUALS_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_n0_b1_b2") != Kernel::T_TENSOR_EQUALS_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_n0_b2_n0") != Kernel::T_TENSOR_EQUALS_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_n0_b2_b1") != Kernel::T_TENSOR_EQUALS_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b1_n0_b1") != Kernel::T_TENSOR_EQUALS_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b1_n0_b2") != Kernel::T_TENSOR_EQUALS_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b1_b2_n0") != Kernel::T_TENSOR_EQUALS_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b1_b2_b1") != Kernel::T_TENSOR_EQUALS_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b2_n0_b1") != Kernel::T_TENSOR_EQUALS_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b2_n0_b2") != Kernel::T_TENSOR_EQUALS_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b2_b1_n0") != Kernel::T_TENSOR_EQUALS_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_equals_b2_b1_b2") != Kernel::T_TENSOR_EQUALS_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_n0_b1_n0") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_n0_b1_b2") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_n0_b2_n0") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_n0_b2_b1") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b1_n0_b1") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b1_n0_b2") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b1_b2_n0") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b1_b2_b1") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b2_n0_b1") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b2_n0_b2") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b2_b1_n0") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_equals_b2_b1_b2") != Kernel::T_TENSOR_GT_EQUALS_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_n0_b1_n0") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_n0_b1_b2") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_n0_b2_n0") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_n0_b2_b1") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b1_n0_b1") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b1_n0_b2") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b1_b2_n0") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b1_b2_b1") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b2_n0_b1") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b2_n0_b2") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b2_b1_n0") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_equals_b2_b1_b2") != Kernel::T_TENSOR_LT_EQUALS_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_n0_b1_n0") != Kernel::T_TENSOR_GT_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_n0_b1_b2") != Kernel::T_TENSOR_GT_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_n0_b2_n0") != Kernel::T_TENSOR_GT_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_n0_b2_b1") != Kernel::T_TENSOR_GT_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b1_n0_b1") != Kernel::T_TENSOR_GT_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b1_n0_b2") != Kernel::T_TENSOR_GT_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b1_b2_n0") != Kernel::T_TENSOR_GT_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b1_b2_b1") != Kernel::T_TENSOR_GT_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b2_n0_b1") != Kernel::T_TENSOR_GT_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b2_n0_b2") != Kernel::T_TENSOR_GT_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b2_b1_n0") != Kernel::T_TENSOR_GT_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_gt_b2_b1_b2") != Kernel::T_TENSOR_GT_BRODCAST_B2B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_n0_b1_n0") != Kernel::T_TENSOR_LT_BRODCAST_N0B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_n0_b1_b2") != Kernel::T_TENSOR_LT_BRODCAST_N0B1B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_n0_b2_n0") != Kernel::T_TENSOR_LT_BRODCAST_N0B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_n0_b2_b1") != Kernel::T_TENSOR_LT_BRODCAST_N0B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b1_n0_b1") != Kernel::T_TENSOR_LT_BRODCAST_B1N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b1_n0_b2") != Kernel::T_TENSOR_LT_BRODCAST_B1N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b1_b2_n0") != Kernel::T_TENSOR_LT_BRODCAST_B1B2N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b1_b2_b1") != Kernel::T_TENSOR_LT_BRODCAST_B1B2B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b2_n0_b1") != Kernel::T_TENSOR_LT_BRODCAST_B2N0B1  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b2_n0_b2") != Kernel::T_TENSOR_LT_BRODCAST_B2N0B2  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b2_b1_n0") != Kernel::T_TENSOR_LT_BRODCAST_B2B1N0  ||
            OpenCLManager::createKernel("tensor_tensor_broadcast_lt_b2_b1_b2") != Kernel::T_TENSOR_LT_BRODCAST_B2B1B2
        )
            throw std::runtime_error("Wrong Kernel index");
    }
}