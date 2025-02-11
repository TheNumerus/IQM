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
     * Intermediate image views `ivFeatErr`, `ivColorErr` should be in format R f32 with dimensions WxH.
     * All `ivTemp` image should be in format RGBA f32 with dimensions WxH.
     * `ivFeatFilter` should be in RGBA f32 format with dimensions Kx1 where K is returned from `featureKernelSize`.
     * `ivColorFilter` should be in RGBA f32 format with dimensions SxS where S is returned from `spatialKernelSize`.
     * `ivColorMap` is for coloring the final output, it should be in format RGBA f32, with custom dimensions.
     * All images should be in layout GENERAL.
     */
    struct FLIPInput {
        const FLIPArguments args;
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef, *ivOut, *ivColorMap, *ivFeatErr, *ivColorErr, *ivFeatFilter, *ivColorFilter;
        const vk::raii::ImageView *ivTemp[3];
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
