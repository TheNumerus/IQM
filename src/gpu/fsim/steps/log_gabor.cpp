/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/log_gabor.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_log_gabor.inc>
;

IQM::GPU::FSIMLogGabor::FSIMLogGabor(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smLogGabor = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
    }));

    const std::vector layout = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layout.size()),
        .pSetLayouts = layout.data()
    };

    this->descSet = std::move(vk::raii::DescriptorSets{device, descriptorSetAllocateInfo}.front());

    this->layout = VulkanRuntime::createPipelineLayout(device, layout, {});
    this->pipeline = VulkanRuntime::createComputePipeline(device, smLogGabor, this->layout);

    this->imageLogGaborFilters = std::vector<std::shared_ptr<VulkanImage>>(FSIM_SCALES);
}

void IQM::GPU::FSIMLogGabor::constructFilter(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height) {
    this->prepareImageStorage(runtime, lowpass, width, height);

    VulkanRuntime::initImages(runtime._cmd_buffer, this->imageLogGaborFilters);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, FSIM_SCALES);
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

    for (unsigned i = 0; i < FSIM_SCALES; i++) {
        this->imageLogGaborFilters[i] = std::make_shared<VulkanImage>(VulkanRuntime::createImage(runtime._device, runtime._physicalDevice, imageInfo));
    }

    auto imageInfosLowpass = VulkanRuntime::createImageInfos({lowpass});
    auto imageInfos = VulkanRuntime::createImageInfos(this->imageLogGaborFilters);

    const auto writeSetLowpass = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfosLowpass
    );

    const auto writeSet = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        imageInfos
    );

    const std::vector writes = {
        writeSetLowpass, writeSet,
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}
