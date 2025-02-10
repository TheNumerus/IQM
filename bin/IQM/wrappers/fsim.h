/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_FSIM_H
#define IQM_BIN_FSIM_H

#include <IQM/fsim.h>
#include "../vulkan_instance.h"
#include "../file_matcher.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"

namespace IQM::Bin {
    struct FSIMResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 16x R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesFloat;
        // 8x RG f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesRg;
        // 2x RGBA f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesColor;

        vk::raii::Buffer bufFft = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memFft = VK_NULL_HANDLE;
        vk::raii::Buffer bufIfft = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memIfft = VK_NULL_HANDLE;
        vk::raii::Buffer bufSort = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memSort = VK_NULL_HANDLE;
        vk::raii::Buffer bufSortTemp = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memSortTemp = VK_NULL_HANDLE;
        vk::raii::Buffer bufSortHist = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memSortHist = VK_NULL_HANDLE;
        vk::raii::Buffer bufNoiseLevels = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memNoiseLevels = VK_NULL_HANDLE;
        vk::raii::Buffer bufNoisePowers = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memNoisePowers = VK_NULL_HANDLE;
        vk::raii::Buffer bufSum = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memSum = VK_NULL_HANDLE;
        std::vector<vk::raii::Buffer> bufEnergy;
        std::vector<vk::raii::DeviceMemory> memEnergy;
        vk::raii::Buffer bufOut = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memOut = VK_NULL_HANDLE;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;

        // FFT
        vk::raii::Fence fftFence = VK_NULL_HANDLE;
        vk::raii::Fence ifftFence = VK_NULL_HANDLE;
    };

    struct FSIMResult {
        float fsim;
        float fsimc;
    };

    void fsim_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches);

    FSIMResources fsim_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance& instance, unsigned dWidth, unsigned dHeight, unsigned histBufSize);
    void fsim_upload(const VulkanInstance& instance, const FSIMResources& res);
    FSIMResult fsim_copy_back(const VulkanInstance& instance, const FSIMResources& res, Timestamps &timestamps);
}

#endif //IQM_BIN_FSIM_H
