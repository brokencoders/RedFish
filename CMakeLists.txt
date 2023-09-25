cmake_minimum_required(VERSION 3.5.0)
project(RedFish VERSION 0.1.0 LANGUAGES CXX)

set(SOURCE_FOLDER "src")
set(HEADER_FOLDER "include")
file(GLOB SOURCE_FILES "${SOURCE_FOLDER}/*.cpp")

find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(RedFish ${SOURCE_FILES})

target_link_libraries(RedFish PRIVATE Boost::iostreams Boost::system Boost::filesystem)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include_directories(include lib/gnuplot/)

