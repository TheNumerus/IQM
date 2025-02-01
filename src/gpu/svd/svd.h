/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef SVD_H
#define SVD_H

#include <vector>
#include <opencv2/core/mat.hpp>

#include "../../input_image.h"
#include "../base/vulkan_runtime.h"
#include "../../timestamps.h"

namespace IQM::GPU {
    struct SVDResult {
        std::vector<float> imageData;
        unsigned int width;
        unsigned int height;
        float msvd;
        Timestamps timestamps;
    };

    class SVD {
    public:
        explicit SVD(const vk::raii::Device &device);
        SVDResult computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);

    private:
        void prepareBuffers(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput, size_t histBufInput);
        void reduceSingularValues(const VulkanRuntime &runtime, uint32_t valueCount);
        void sortBlocks(const VulkanRuntime &runtime, uint32_t nValues, uint32_t nSortWorkgroups, uint32_t nBlocksPerWorkgroup, uint32_t sortGlobalInvocationSize);
        void computeMsvd(const VulkanRuntime &runtime, uint32_t nValues);
        void copyToGpu(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput, size_t histBufInput);
        void copyFromGpu(const VulkanRuntime &runtime, size_t sizeOutput);

        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutReduce = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineReduce = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutReduce = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetReduce = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSort = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSortHistogram = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSort = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSortHistogram = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSort = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortOdd = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramOdd = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSum = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSum = VK_NULL_HANDLE;

        vk::raii::Buffer inputBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory inputMemory = VK_NULL_HANDLE;

        vk::raii::Buffer outBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory outMemory = VK_NULL_HANDLE;

        vk::raii::Buffer outSortBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory outSortMemory = VK_NULL_HANDLE;

        vk::raii::Buffer outSortTempBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory outSortTempMemory = VK_NULL_HANDLE;

        vk::raii::Buffer outSortHistBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory outSortHistMemory = VK_NULL_HANDLE;

        vk::raii::Buffer stgBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgMemory = VK_NULL_HANDLE;
    };
}

#endif //SVD_H
