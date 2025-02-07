/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_VULKAN_RESOURCE_H
#define IQM_VULKAN_RESOURCE_H

#include <vulkan/vulkan_raii.hpp>

namespace IQM::Bin {
    class VulkanImage {
    public:
        vk::raii::DeviceMemory memory = VK_NULL_HANDLE;
        vk::raii::Image image = VK_NULL_HANDLE;
        vk::raii::ImageView imageView = VK_NULL_HANDLE;

        uint32_t width = 0;
        uint32_t height = 0;
    };

    class VulkanResource {
    public:
        [[nodiscard]] static std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
            const vk::raii::Device &device,
            const vk::raii::PhysicalDevice &physicalDevice,
            unsigned bufferSize,
            vk::BufferUsageFlags bufferFlags,
            vk::MemoryPropertyFlags memoryFlags);
        [[nodiscard]] static VulkanImage createImage(
            const vk::raii::Device &device,
            const vk::raii::PhysicalDevice &physicalDevice,
            const vk::ImageCreateInfo &imageInfo);
        static void initImages(const vk::raii::CommandBuffer &cmd_buf, const std::vector<std::shared_ptr<VulkanImage>> &images);

    private:
        static uint32_t findMemoryType(vk::PhysicalDeviceMemoryProperties const &memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask);
    };
}

#endif //IQM_VULKAN_RESOURCE_H
