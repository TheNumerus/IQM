/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_sum_filter_responses.h"

#include <fsim.h>

IQM::GPU::FSIMSumFilterResponses::FSIMSumFilterResponses(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_sum_filter_responses.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = FSIM_ORIENTATIONS,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = FSIM_ORIENTATIONS,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }));

    const std::vector layouts = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);

    // 1x float - orientation
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layout = runtime.createPipelineLayout(layouts, ranges);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->filterResponsesInput = std::vector<std::shared_ptr<VulkanImage>>(FSIM_ORIENTATIONS);
    this->filterResponsesRef = std::vector<std::shared_ptr<VulkanImage>>(FSIM_ORIENTATIONS);
}

void IQM::GPU::FSIMSumFilterResponses::computeSums(const VulkanRuntime &runtime, const vk::raii::Buffer &filters, int width, int height) {
    this->prepareImageStorage(runtime, filters, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    // create only one barrier for all images
    auto images = this->filterResponsesInput;
    images.insert(images.end(),this->filterResponsesRef.begin(),this->filterResponsesRef.end());
    VulkanRuntime::initImages(runtime._cmd_buffer, images);

    for (int i = 0; i < FSIM_ORIENTATIONS; i++) {
        runtime._cmd_buffer->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, i);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    }

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime.waitForFence(fence);
}

void IQM::GPU::FSIMSumFilterResponses::prepareImageStorage(const VulkanRuntime &runtime, const vk::raii::Buffer &filters, int width, int height) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32Sfloat,
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

    for (int i = 0; i < FSIM_ORIENTATIONS; i++) {
        this->filterResponsesInput[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
        this->filterResponsesRef[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    }

    auto imageInfosIn = VulkanRuntime::createImageInfos(this->filterResponsesInput);
    auto imageInfosRef = VulkanRuntime::createImageInfos(this->filterResponsesRef);

    const vk::WriteDescriptorSet writeSetIn{
        .dstSet = this->descSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosIn.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetRef{
        .dstSet = this->descSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::DescriptorBufferInfo bufferInfo{
        .buffer = filters,
        .offset = 0,
        .range = sizeof(float) * width * height * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
    };

    const vk::WriteDescriptorSet writeSetBuf{
        .dstSet = this->descSet,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfo,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetIn, writeSetRef, writeSetBuf
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}
