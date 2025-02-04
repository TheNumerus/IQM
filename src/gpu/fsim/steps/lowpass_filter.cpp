/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/lowpass_filter.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_lowpassfilter.inc>
;

IQM::GPU::FSIMLowpassFilter::FSIMLowpassFilter(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
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


void IQM::GPU::FSIMLowpassFilter::constructFilter(const VulkanRuntime &runtime, const int width, const int height) {
    this->prepareImageStorage(runtime, width, height);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    int order = 15;
    float cutoff = 0.45;

    runtime._cmd_buffer->pushConstants<float>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, cutoff);
    runtime._cmd_buffer->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(float), order);

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FSIMLowpassFilter::prepareImageStorage(const VulkanRuntime &runtime, int width, int height) {
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

    this->imageLowpassFilter = std::make_shared<VulkanImage>(VulkanRuntime::createImage(runtime._device, runtime._physicalDevice, imageInfo));

    VulkanRuntime::initImages(runtime._cmd_buffer, {this->imageLowpassFilter});

    auto imageInfos = VulkanRuntime::createImageInfos({
        this->imageLowpassFilter,
    });

    const auto writeSet = VulkanRuntime::createWriteSet(
        descSet,
        0,
        imageInfos
    );

    const std::vector writes = {
        writeSet
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}
