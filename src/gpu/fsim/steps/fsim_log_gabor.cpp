/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_log_gabor.h"

IQM::GPU::FSIMLogGabor::FSIMLogGabor(const VulkanRuntime &runtime, const unsigned scales) : scales(scales) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_log_gabor.spv");

    std::vector<vk::DescriptorSetLayout> totalLayouts;
    for (unsigned i = 0; i < scales; i++) {
        totalLayouts.push_back(*runtime._descLayoutTwoImage);
    }

    const std::vector layout = {
        *runtime._descLayoutTwoImage,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(totalLayouts.size()),
        .pSetLayouts = totalLayouts.data()
    };

    this->descSets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};

    // 1x int - scale
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layout = runtime.createPipelineLayout(layout, ranges);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->imageLogGaborFilters = std::vector<std::shared_ptr<VulkanImage>>(this->scales);
}

void IQM::GPU::FSIMLogGabor::constructFilter(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height) {
    this->prepareImageStorage(runtime, lowpass, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);


    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    for (unsigned scale = 0; scale < this->scales; scale++) {
        runtime.setImageLayout(this->imageLogGaborFilters[scale]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSets[scale]}, {});
        runtime._cmd_buffer.pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, scale);

        runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);
    }

    runtime._cmd_buffer.end();

    const std::vector cmdBufs = {
        &*runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue.submit(submitInfo, *fence);
    runtime._device.waitIdle();
}

void IQM::GPU::FSIMLogGabor::prepareImageStorage(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D(width, height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    for (unsigned i = 0; i < this->scales; i++) {
        this->imageLogGaborFilters[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));

        auto imageInfosLowpass = VulkanRuntime::createImageInfos({lowpass});
        auto imageInfos = VulkanRuntime::createImageInfos({this->imageLogGaborFilters[i]});

        const vk::WriteDescriptorSet writeSetLowpass{
            .dstSet = this->descSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = imageInfosLowpass.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        const vk::WriteDescriptorSet writeSet{
            .dstSet = this->descSets[i],
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = imageInfos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        const std::vector writes = {
            writeSetLowpass, writeSet,
        };

        runtime._device.updateDescriptorSets(writes, nullptr);
    }
}