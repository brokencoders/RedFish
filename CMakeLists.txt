cmake_minimum_required(VERSION 3.9.0)
project(RedFish VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FOLDER "src")
set(HEADER_FOLDER "include")

find_package(OpenCL REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(RedFish ${SOURCE_FOLDER}/ActivationLayer.cpp
                    ${SOURCE_FOLDER}/LinearLayer.cpp
                    ${SOURCE_FOLDER}/RecurrentLayer.cpp
                    ${SOURCE_FOLDER}/Loss.cpp
                    ${SOURCE_FOLDER}/Optimizer.cpp
                    ${SOURCE_FOLDER}/Layer.cpp
                    ${SOURCE_FOLDER}/ConvLayer.cpp
                    ${SOURCE_FOLDER}/Model.cpp
                    ${SOURCE_FOLDER}/MaxPoolLayer.cpp
                    ${SOURCE_FOLDER}/FlattenLayer.cpp
                    ${SOURCE_FOLDER}/DropoutLayer.cpp
                    ${SOURCE_FOLDER}/OpenCLManager.cpp
                    ${SOURCE_FOLDER}/Tensor.cpp
                    ${SOURCE_FOLDER}/TensorStd.cpp
                    ${SOURCE_FOLDER}/TensorView.cpp
                    ${SOURCE_FOLDER}/Tensor/algebra.cpp
                    ${SOURCE_FOLDER}/Tensor/conv.cpp
                    ${SOURCE_FOLDER}/Tensor/fft.cpp
)
add_executable(RedFishMain "${SOURCE_FOLDER}/main.cpp")
add_executable(MatrixMain "${SOURCE_FOLDER}/mainMatrixMul.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    target_link_libraries(RedFish PUBLIC OpenMP::OpenMP_CXX)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

target_compile_options(MatrixMain PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_compile_options(RedFish PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(RedFish PUBLIC include lib/ lib/OpenCL-CLHPP/include/ ${OpenCL_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(RedFishMain PUBLIC include lib/ lib/OpenCL-CLHPP/include/ ${OpenCL_INCLUDE_DIRS})
target_include_directories(MatrixMain  PUBLIC include lib/ lib/OpenCL-CLHPP/include/ ${OpenCL_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(RedFish PUBLIC OpenCL::OpenCL CUDA::cudart CUDA::cublas $<LINK_ONLY:MKL::MKL>) #Matplot++::matplot)
target_link_libraries(RedFishMain PUBLIC RedFish)
target_link_libraries(MatrixMain  PUBLIC RedFish CUDA::cudart CUDA::cublas $<LINK_ONLY:MKL::MKL>)

add_subdirectory(test)

