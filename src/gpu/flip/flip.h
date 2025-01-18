/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FLIP_H
#define FLIP_H

#include "../../input_image.h"
#include "../img_params.h"
#include "../../timestamps.h"
#include "../base/vulkan_runtime.h"

namespace IQM::GPU {
    struct FLIPResult {
        float mean_flip;
        Timestamps timestamps;
    };

    struct FLIPArguments {
        float monitor_resolution_x = 2560;
        float monitor_distance = 0.7;
        float monitor_width = 0.6;
    };

    class FLIP {
    public:
        explicit FLIP(const VulkanRuntime &runtime);
        FLIPResult computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref, const FLIPArguments &args);

    private:
        void prepareImageStorage(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref, int kernel_size);
        void convertToYCxCz(const VulkanRuntime &runtime);
        void createFeatureFilters(const VulkanRuntime &runtime, float pixels_per_degree, int kernel_size);
        void computeFeatureErrorMap(const VulkanRuntime &runtime);

        ImageParameters imageParameters;

        vk::raii::ShaderModule inputConvertKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout inputConvertLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline inputConvertPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout inputConvertDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet inputConvertDescSet = VK_NULL_HANDLE;

        vk::raii::ShaderModule featureFilterCreateKernel = VK_NULL_HANDLE;
        vk::raii::ShaderModule featureFilterNormalizeKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout featureFilterCreateLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline featureFilterCreatePipeline = VK_NULL_HANDLE;
        vk::raii::Pipeline featureFilterNormalizePipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout featureFilterCreateDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet featureFilterCreateDescSet = VK_NULL_HANDLE;

        vk::raii::ShaderModule featureDetectKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout featureDetectLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline featureDetectPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout featureDetectDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet featureDetectDescSet = VK_NULL_HANDLE;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;

        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;
        std::shared_ptr<VulkanImage> imageYccInput;
        std::shared_ptr<VulkanImage> imageYccRef;
        std::shared_ptr<VulkanImage> imageFeatureError;

        std::shared_ptr<VulkanImage> imageFeaturePointFilter;
        std::shared_ptr<VulkanImage> imageFeatureEdgeFilter;
    };
}

#endif //FLIP_H