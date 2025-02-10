/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_PHASE_CONGRUENCY_H
#define FSIM_PHASE_CONGRUENCY_H
#include <memory>

#include <IQM/base/vulkan_runtime.h>
#include <IQM/base/vulkan_image.h>

namespace IQM {
    struct FSIMInput;

    class FSIMPhaseCongruency {
    public:
        explicit FSIMPhaseCongruency(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput& input) const;
        void compute(const FSIMInput &input, unsigned width, unsigned height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
    };
}

#endif //FSIM_PHASE_CONGRUENCY_H
