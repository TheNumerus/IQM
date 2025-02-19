/*
* Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_PSNR_H
#define IQM_BIN_PSNR_H

#include <IQM/psnr.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct PSNRResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer sumBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory sumMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct PSNRResult {
        float db;
    };

    void psnr_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void psnr_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::PSNR& psnr, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref);

    PSNRResources psnr_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance);
    void psnr_upload(const IQM::VulkanInstance& instance, const PSNRResources& res);
    PSNRResult psnr_copy_back(const IQM::VulkanInstance& instance, const PSNRResources& res, Timestamps &timestamps);
}

#endif //IQM_BIN_PSNR_H
