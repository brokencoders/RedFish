cmake_minimum_required(VERSION 3.9.0)
project(RedFish VERSION 0.1.0 LANGUAGES CXX)

set(SOURCE_FOLDER "src")
set(HEADER_FOLDER "include")

find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(RedFish "${SOURCE_FOLDER}/main.cpp")
add_executable(AlgTest "${SOURCE_FOLDER}/test.cpp")

target_link_libraries(RedFish PRIVATE Boost::iostreams Boost::system Boost::filesystem)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(RedFish PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(AlgTest PUBLIC OpenMP::OpenMP_CXX)
#endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

target_include_directories(RedFish PUBLIC include lib/gnuplot/ lib/)

