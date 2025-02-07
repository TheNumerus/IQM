/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_SSIM_H
#define IQM_BIN_SSIM_H

#include <IQM/ssim.h>
#include "../vulkan_instance.h"
#include "../file_matcher.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"

namespace IQM::Bin {
    struct SSIMResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer mssimBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory mssimMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 5x R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesBlurred;

        // R f32 output image
        std::shared_ptr<VulkanImage> imageOut;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct SSIMResult {
        std::vector<float> imageData;
        float mssim;
    };

    void ssim_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches);

    SSIMResources ssim_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance& instance);
    void ssim_upload(const VulkanInstance& instance, const SSIMResources& res);
    SSIMResult ssim_copy_back(const VulkanInstance& instance, const SSIMResources& res, Timestamps &timestamps, uint32_t kernelSize);
}

#endif //IQM_BIN_SSIM_H
