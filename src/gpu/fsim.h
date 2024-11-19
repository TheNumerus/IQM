#ifndef FSIM_H
#define FSIM_H
#include <opencv2/core/mat.hpp>
#include "base/vulkan_runtime.h"

namespace IQM::GPU {
    class FSIM {
    public:
        explicit FSIM(const VulkanRuntime &runtime);
        cv::Mat computeMetric(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);

        bool doColorComparison = true;

    private:
        static int computeDownscaleFactor(int cols, int rows);
        void sendImagesToGpu(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);
        void createDownscaledImages(const VulkanRuntime & runtime, int width_downscale, int height_downscale);
        void computeDownscaledImages(const VulkanRuntime & runtime, int, int, int);
        void createLowpassFilter(const VulkanRuntime & runtime, int, int);

        vk::raii::ShaderModule downscaleKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutDownscale = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineDownscale = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleRef = VK_NULL_HANDLE;

        VulkanImage imageInput;
        VulkanImage imageRef;

        VulkanImage imageInputDownscaled;
        VulkanImage imageRefDownscaled;
        VulkanImage imageLowpassFilter;

        vk::raii::ShaderModule lowpassFilterKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutLowpassFilter = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineLowpassFilter = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetLowpassFilter = VK_NULL_HANDLE;
    };
}

#endif //FSIM_H