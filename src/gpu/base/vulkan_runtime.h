#ifndef VULKANRUNTIME_H
#define VULKANRUNTIME_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace IQM::GPU {
    class VulkanRuntime {
    public:
        VulkanRuntime();
        [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::string& path) const;
        [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout(const vk::PipelineLayoutCreateInfo &pipelineLayoutCreateInfo) const;
        [[nodiscard]] vk::raii::Pipeline createComputePipeline(const vk::raii::ShaderModule &shader, const vk::raii::PipelineLayout &layout) const;
        [[nodiscard]] std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(unsigned bufferSize, vk::BufferUsageFlagBits bufferFlags, vk::MemoryPropertyFlags memoryFlags) const;
        [[nodiscard]] std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(const vk::ImageCreateInfo &imageInfo) const;
        [[nodiscard]] vk::raii::ImageView createImageView(const vk::raii::Image &image) const;
        void setImageLayout(const vk::raii::Image &image, vk::ImageLayout srcLayout, vk::ImageLayout targetLayout) const;

        vk::raii::Context _context;
        // assigned VK_NULL_HANDLE to sidestep accidental usage of deleted constructor
        vk::raii::Instance _instance = VK_NULL_HANDLE;
        vk::raii::PhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        vk::raii::Device _device = VK_NULL_HANDLE;
        vk::raii::Queue _queue = VK_NULL_HANDLE;
        vk::raii::CommandPool _commandPool = VK_NULL_HANDLE;
        vk::raii::CommandBuffer _cmd_buffer = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorPool _descPool = VK_NULL_HANDLE;
    };
}

#endif //VULKANRUNTIME_H
