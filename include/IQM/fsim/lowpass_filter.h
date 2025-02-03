/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_LOWPASS_FILTER_H
#define FSIM_LOWPASS_FILTER_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM::GPU {
    class FSIMLowpassFilter {
    public:
        explicit FSIMLowpassFilter(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void constructFilter(const VulkanRuntime &runtime, int width, int height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageLowpassFilter;
    private:
        void prepareImageStorage(const VulkanRuntime & runtime, int width, int height);
    };
}

#endif //FSIM_LOWPASS_FILTER_H
