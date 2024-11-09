cmake_minimum_required(VERSION 3.7)

project(opencl_cmake VERSION 0.0.1 LANGUAGES CXX)

add_executable(vadd main.cpp)
target_compile_features(vadd PRIVATE cxx_auto_type)

find_package(OpenCL REQUIRED)
target_link_libraries(vadd OpenCL::OpenCL)