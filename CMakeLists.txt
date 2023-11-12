cmake_minimum_required(VERSION 3.9.0)
project(RedFish VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FOLDER "src")
set(HEADER_FOLDER "include")

find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_library(RedFish ${SOURCE_FOLDER}/ActivationLayer.cpp
                    ${SOURCE_FOLDER}/LinearLayer.cpp
                    ${SOURCE_FOLDER}/Loss.cpp
                    ${SOURCE_FOLDER}/Optimizer.cpp
                    ${SOURCE_FOLDER}/Layer.cpp
                    ${SOURCE_FOLDER}/ConvLayer.cpp
                    ${SOURCE_FOLDER}/Model.cpp)
add_executable(RedFishMain "${SOURCE_FOLDER}/main.cpp")

target_link_libraries(RedFish PRIVATE Boost::iostreams Boost::system Boost::filesystem)

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

target_include_directories(RedFish PUBLIC include)
target_include_directories(RedFishMain PUBLIC include lib/gnuplot/ lib/)
target_link_libraries(RedFishMain PUBLIC RedFish)

add_subdirectory(test)

