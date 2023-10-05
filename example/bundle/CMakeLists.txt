# define cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# search path to *.cmake
# TORCH_DIR is {libtorch_directory}/share/cmake
set(CMAKE_PREFIX_PATH $ENV{TORCH_DIR})

# specify project
project(example)
add_executable(example "../app.cpp" "../../src/libtorch-summary.cpp")

# dependencies
# libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(example ${TORCH_LIBRARIES})
# libtorch-summary
include_directories(example "../../include")

get_target_property(DEBUG_VAR torch INTERFACE_INCLUDE_DIRECTORIES)

set_property(TARGET example PROPERTY CXX_STANDARD 17)
