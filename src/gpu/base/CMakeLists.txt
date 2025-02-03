cmake_minimum_required(VERSION 3.29)
project(IQM-LibBase)

add_library(IQM-LibBase STATIC vulkan_runtime.cpp)

add_library(IQM::LibBase ALIAS IQM-LibBase)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-LibBase
        PUBLIC "../../../include"
)

target_link_libraries(IQM-LibBase PUBLIC Vulkan::Vulkan)