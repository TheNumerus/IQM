/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/lowpass_filter.h>

#include "IQM/fsim.h"

static std::vector<uint32_t> src =
#include <fsim/fsim_lowpassfilter.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMLowpassFilter::FSIMLowpassFilter(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smLowpass = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector layouts = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);

    // 1x float - cutoff, 1x int - order
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) + sizeof(float));

    this->layout = VulkanRuntime::createPipelineLayout(device, layouts, ranges);
    this->pipeline = VulkanRuntime::createComputePipeline(device, smLowpass, this->layout);
}

void IQM::FSIMLowpassFilter::setUpDescriptors(const FSIMInput &input) const {
    auto imageInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[0]});

    const auto writeSet = VulkanRuntime::createWriteSet(
        descSet,
        0,
        imageInfos
    );

    const std::vector writes = {
        writeSet
    };

    input.device->updateDescriptorSets(writes, nullptr);
}

void IQM::FSIMLowpassFilter::constructFilter(const FSIMInput &input, const int width, const int height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    int order = 15;
    float cutoff = 0.45;

    input.cmdBuf->pushConstants<float>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, cutoff);
    input.cmdBuf->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(float), order);

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);
}
