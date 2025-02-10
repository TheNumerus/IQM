/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_FLIP_H
#define IQM_BIN_FLIP_H

#include <IQM/flip.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct FLIPResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgColormap = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgColormapMemory = VK_NULL_HANDLE;
        vk::raii::Buffer meanBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory meanMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 2x R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesFloatTemp;

        // 8x RGBA f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesColorTemp;

        // 2x RGBA f32 filters,
        std::shared_ptr<VulkanImage> imageColorFilter;
        std::shared_ptr<VulkanImage> imageFeatureFilter;

        // RGBA f32 colormap
        std::shared_ptr<VulkanImage> imageColorMap;

        // RGBA f32 output image
        std::shared_ptr<VulkanImage> imageOut;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct FLIPResult {
        std::vector<float> imageData;
        float meanFlip;
    };

    void flip_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void flip_run_single(const IQM::ProfileArgs& args, const IQM::VulkanInstance& instance, IQM::FLIP& flip, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref);

    FLIPResources flip_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance, unsigned spatialKernelSize, unsigned featureKernelSize);
    void flip_upload(const IQM::VulkanInstance& instance, const FLIPResources& res);
    FLIPResult flip_copy_back(const IQM::VulkanInstance& instance, const FLIPResources& res, Timestamps &timestamps);
}

#endif //IQM_BIN_FLIP_H
