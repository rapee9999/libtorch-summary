# define cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) 
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# search path to *.cmake
set(CMAKE_PREFIX_PATH $ENV{TORCH_DIR}) 
# set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

# specify project
project(libtorch-summary VERSION 1.0.0 DESCRIPTION "torch model summary")
add_library(libtorch-summary SHARED "src/libtorch-summary.cpp")

# specify version
set_target_properties(libtorch-summary PROPERTIES VERSION ${PROJECT_VERSION})
# *.so file version
set_target_properties(libtorch-summary PROPERTIES SOVERSION 1)
# declare public API
set_target_properties(libtorch-summary PROPERTIES PUBLIC_HEADER "include/libtorch-summary.h")
# in linux, default library filename is lib{name}.so - this removes lib prefix.
set_target_properties(libtorch-summary PROPERTIES PREFIX "")

# dependencies
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(libtorch-summary "include")
target_link_libraries(libtorch-summary ${TORCH_LIBRARIES})

get_target_property(DEBUG_VAR torch INTERFACE_INCLUDE_DIRECTORIES)

set_property(TARGET libtorch-summary PROPERTY CXX_STANDARD 17)
