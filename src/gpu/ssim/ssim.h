/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef SSIM_H
#define SSIM_H

#include <vector>

#include "../../input_image.h"
#include "../img_params.h"
#include "../base/vulkan_runtime.h"
#include "../../timestamps.h"

namespace IQM::GPU {
    struct SSIMResult {
        std::vector<float> imageData;
        unsigned int width;
        unsigned int height;
        float mssim;
        Timestamps timestamps;
    };

    class SSIM {
    public:
        explicit SSIM(const VulkanRuntime &runtime);
        SSIMResult computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);

        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
        float sigma = 1.5;
    private:
        ImageParameters imageParameters;

        vk::raii::ShaderModule kernelSsim = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSsim = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSsim = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSsim = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSsim = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelLumapack = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutLumapack = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineLumapack = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutLumapack = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetLumapack = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelGauss = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutGauss = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGauss = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutGauss = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGauss = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelMssim = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutMssim = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineMssim = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutMssim = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetMssim = VK_NULL_HANDLE;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;

        vk::raii::Fence transferFence = VK_NULL_HANDLE;
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer mssimBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory mssimMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesBlurred;
        std::vector<std::shared_ptr<VulkanImage>> imagesBlurredTemp;

        // R f32 output image
        std::shared_ptr<VulkanImage> imageOut;

        void prepareImages(const VulkanRuntime &runtime, Timestamps& timestamps, const InputImage &image, const InputImage &ref);
    };
}

#endif //SSIM_H
