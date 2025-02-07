/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_VULKAN_INSTANCE_H
#define IQM_BIN_VULKAN_INSTANCE_H

#include <string>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

namespace IQM::Bin {
    const std::string LAYER_VALIDATION = "VK_LAYER_KHRONOS_validation";

    class VulkanInstance {
    public:
        VulkanInstance();

        std::string selectedDevice;

        vk::raii::Context context;
        vk::raii::Instance instance = VK_NULL_HANDLE;
        vk::raii::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
        vk::raii::Device device = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::Queue> queue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex;
        std::shared_ptr<vk::raii::Queue> transferQueue = VK_NULL_HANDLE;
        uint32_t transferQueueFamilyIndex;
        std::shared_ptr<vk::raii::CommandPool> commandPool = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandPool> commandPoolTransfer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> cmd_buffer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> cmd_bufferTransfer = VK_NULL_HANDLE;

        static void waitForFence(const vk::raii::Device &device, const vk::raii::Fence &fence);
    private:
        void initQueues();
        static std::vector<const char *> getLayers();
    };
}

#endif //IQM_BIN_VULKAN_INSTANCE_H
