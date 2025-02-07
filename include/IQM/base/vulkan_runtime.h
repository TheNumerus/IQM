/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef VULKANRUNTIME_H
#define VULKANRUNTIME_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "vulkan_image.h"

namespace IQM::GPU {
    class VulkanRuntime {
    public:
        VulkanRuntime();
        [[nodiscard]] static vk::raii::ShaderModule createShaderModule(
            const vk::raii::Device &device,
            const std::vector<uint32_t> &spvCode);
        [[nodiscard]] static vk::raii::PipelineLayout createPipelineLayout(
            const vk::raii::Device &device,
            const std::vector<vk::DescriptorSetLayout> &layouts,
            const std::vector<vk::PushConstantRange> &ranges);
        [[nodiscard]] static vk::raii::Pipeline createComputePipeline(
            const vk::raii::Device &device,
            const vk::raii::ShaderModule &shader,
            const vk::raii::PipelineLayout &layout);
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
        [[nodiscard]] static vk::raii::DescriptorPool createDescPool(
            const vk::raii::Device &device,
            uint32_t maxSets,
            std::vector<vk::DescriptorPoolSize> poolSizes);
        [[nodiscard]] static vk::raii::DescriptorSetLayout createDescLayout(
            const vk::raii::Device &device,
            const std::vector<std::pair<vk::DescriptorType, uint32_t>> &stub);
        static void initImages(const std::shared_ptr<vk::raii::CommandBuffer> &cmd_buf, const std::vector<std::shared_ptr<VulkanImage>> &images);
        static std::vector<vk::PushConstantRange> createPushConstantRange(unsigned size);
        static std::vector<vk::DescriptorImageInfo> createImageInfos(const std::vector<std::shared_ptr<VulkanImage>> &images);
        static std::vector<vk::DescriptorImageInfo> createImageInfos(const std::vector<const vk::raii::ImageView *> &images);
        static std::pair<uint32_t, uint32_t> compute2DGroupCounts(const int width, const unsigned height, const unsigned tileSize) {
            auto groupsX = width / tileSize;
            if (width % tileSize != 0) {
                groupsX++;
            }
            auto groupsY = height / tileSize;
            if (height % tileSize != 0) {
                groupsY++;
            }

            return std::make_pair(groupsX, groupsY);
        }
        void waitForFence(const vk::raii::Fence&) const;

        static vk::WriteDescriptorSet createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorImageInfo> &imgInfos);
        static vk::WriteDescriptorSet createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorBufferInfo> &bufInfos);

        std::string selectedDevice;

        vk::raii::Context _context;
        // assigned VK_NULL_HANDLE to sidestep accidental usage of deleted constructor
        vk::raii::Instance _instance = VK_NULL_HANDLE;
        vk::raii::PhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        vk::raii::Device _device = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::Queue> _queue = VK_NULL_HANDLE;
        uint32_t _queueFamilyIndex;
        std::shared_ptr<vk::raii::Queue> _transferQueue = VK_NULL_HANDLE;
        uint32_t _transferQueueFamilyIndex;
        std::shared_ptr<vk::raii::CommandPool> _commandPool = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandPool> _commandPoolTransfer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> _cmd_buffer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> _cmd_bufferTransfer = VK_NULL_HANDLE;

#ifdef PROFILE
        void createSwapchain(vk::SurfaceKHR surface);
        unsigned acquire();
        void present(unsigned index);
        vk::raii::SwapchainKHR swapchain = VK_NULL_HANDLE;
        vk::raii::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        vk::raii::Semaphore renderFinishedSemaphore = VK_NULL_HANDLE;
        vk::raii::Fence swapchainFence = VK_NULL_HANDLE;
#endif
    private:
        void initQueues();
        void initDescriptors();
        static std::vector<const char *> getLayers();
    };
}

#endif //VULKANRUNTIME_H
