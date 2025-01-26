/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_NOISE_POWER_H
#define FSIM_NOISE_POWER_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    class FSIMNoisePower {
    public:
        explicit FSIMNoisePower(const VulkanRuntime &runtime);
        void computeNoisePower(const VulkanRuntime &runtime, const vk::raii::Buffer &filterSums, const vk::raii::Buffer &fftBuffer, int width, int height);

        vk::raii::DeviceMemory noisePowersMemory = VK_NULL_HANDLE;
        vk::raii::Buffer noisePowers = VK_NULL_HANDLE;

        vk::raii::DeviceMemory noisePowersSortMemory = VK_NULL_HANDLE;
        vk::raii::Buffer noisePowersSortBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory noisePowersTempMemory = VK_NULL_HANDLE;
        vk::raii::Buffer noisePowersTempBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory noisePowersSortHistogramMemory = VK_NULL_HANDLE;
        vk::raii::Buffer noisePowersSortHistogramBuf = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelSort = VK_NULL_HANDLE;
        vk::raii::ShaderModule kernelSortHistogram = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSort = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSortHistogram = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSort = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSortHistogram = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSort = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortOdd = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramOdd = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelNoisePower = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutNoisePower = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineNoisePower = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutNoisePower = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetNoisePower = VK_NULL_HANDLE;
    private:
        void prepareStorage(const VulkanRuntime &runtime, const vk::raii::Buffer &fftBuffer, const vk::raii::Buffer &filterSums, unsigned size, unsigned histBufSize);
    };
}

#endif //FSIM_NOISE_POWER_H
