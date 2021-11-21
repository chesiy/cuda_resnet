cmake_minimum_required(VERSION 3.20)
project(cudnn_resnet)

set(CMAKE_CXX_STANDARD 14)

add_executable(cudnn_resnet try.cpp blocks.cpp resnet.cpp myblocks.cpp)
