/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_FLIP_H
#define IQM_FLIP_H

#include <IQM/flip/color_pipeline.h>
#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FLIPArguments {
        float monitor_resolution_x = 2560;
        float monitor_distance = 0.7;
        float monitor_width = 0.6;
    };

    /**
     * Input parameters for FLIP computation.
     *
     * Source image views `ivTest` and `ivRef` are expected to be views into RGBA u8 images of WxH.
     * Rest of image views are expected to be in R f32 format with dimensions WxH.
     * All images should be in layout GENERAL.
     *
     *    W x H x  4 std::shared_ptr<VulkanImage> imageInput;
          W x H x  4 std::shared_ptr<VulkanImage> imageRef;
          W x H x 16 std::shared_ptr<VulkanImage> imageYccInput;
          W x H x 16 std::shared_ptr<VulkanImage> imageYccRef;
          W x H x 16 std::shared_ptr<VulkanImage> imageFilterTempInput;
          W x H x 16 std::shared_ptr<VulkanImage> imageFilterTempRef;
          W x H x  4 std::shared_ptr<VulkanImage> imageFeatureError;
        256 x 1 x 16 std::shared_ptr<VulkanImage> imageColorMap;
          K x 1 x 16 std::shared_ptr<VulkanImage> imageFeatureFilters;
          W x H x 16 std::shared_ptr<VulkanImage> imageOut;
          S x S x 16 std::shared_ptr<VulkanImage> csfFilter;
          W x H x 16 std::shared_ptr<VulkanImage> inputPrefilterTemp;
          W x H x 16 std::shared_ptr<VulkanImage> refPrefilterTemp;
          W x H x 16 std::shared_ptr<VulkanImage> inputPrefilter;
          W x H x 16 std::shared_ptr<VulkanImage> refPrefilter;
          W x H x  4 std::shared_ptr<VulkanImage> imageColorError;
     */
    struct FLIPInput {
        const FLIPArguments args;
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef, *ivOut, *ivColorMap, *ivFeatErr, *ivColorErr, *ivFeatFilter, *ivColorFilter;
        const vk::raii::ImageView *ivTemp[8];
        const vk::raii::Image *imgOut;
        const vk::raii::Buffer *bufMean;
        unsigned width, height;
    };

    class FLIP {
    public:
        explicit FLIP(const vk::raii::Device &device);
        void computeMetric(const FLIPInput& input);

        float static pixelsPerDegree(const FLIPArguments &args);
        unsigned static spatialKernelSize(const FLIPArguments &args);
        unsigned static featureKernelSize(const FLIPArguments &args);

    private:
        void convertToYCxCz(const FLIPInput& input);
        void createFeatureFilters(const FLIPInput& input);
        void computeFeatureErrorMap(const FLIPInput& input);
        void computeFinalErrorMap(const FLIPInput& input);
        void setUpDescriptors(const FLIPInput& input);

        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        FLIPColorPipeline colorPipeline;

        vk::raii::PipelineLayout inputConvertLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline inputConvertPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout inputConvertDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet inputConvertDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout featureFilterCreateLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline featureFilterCreatePipeline = VK_NULL_HANDLE;
        vk::raii::Pipeline featureFilterNormalizePipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout featureFilterCreateDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet featureFilterCreateDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout featureFilterHorizontalLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline featureFilterHorizontalPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout featureFilterHorizontalDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet featureFilterHorizontalDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout featureDetectLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline featureDetectPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet featureDetectDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout errorCombineLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline errorCombinePipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout errorCombineDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet errorCombineDescSet = VK_NULL_HANDLE;
    };
}

#endif //IQM_FLIP_H
