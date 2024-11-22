#include "fsim.h"
#include "../img_params.h"

IQM::GPU::FSIM::FSIM(const VulkanRuntime &runtime) {
    this->downscaleKernel = runtime.createShaderModule("../shaders_out/fsim_downsample.spv");
    this->lowpassFilterKernel = runtime.createShaderModule("../shaders_out/fsim_lowpassfilter.spv");
    this->kernelGradientMap = runtime.createShaderModule("../shaders_out/fsim_gradientmap.spv");

    const std::vector layout_2 = {
        *runtime._descLayoutTwoImage,
    };

    const std::vector layout_1 = {
        *runtime._descLayoutOneImage,
    };

    const std::vector allLayouts = {
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutOneImage,
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allLayouts.size()),
        .pSetLayouts = allLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSetDownscaleIn = std::move(sets[0]);
    this->descSetDownscaleRef = std::move(sets[1]);
    this->descSetLowpassFilter = std::move(sets[2]);
    this->descSetGradientMapIn = std::move(sets[3]);
    this->descSetGradientMapRef = std::move(sets[4]);

    // 2x int - kernel size, retyped bool
    const auto downsampleRanges = VulkanRuntime::createPushConstantRange(sizeof(int) * 2);
    // 1x float - cutoff, 1x int - order
    const auto lowpassRanges = VulkanRuntime::createPushConstantRange(sizeof(int) + sizeof(float));

    this->layoutDownscale = runtime.createPipelineLayout(layout_2, downsampleRanges);
    this->pipelineDownscale = runtime.createComputePipeline(this->downscaleKernel, this->layoutDownscale);
    this->layoutLowpassFilter = runtime.createPipelineLayout(layout_1, lowpassRanges);
    this->pipelineLowpassFilter = runtime.createComputePipeline(this->lowpassFilterKernel, this->layoutLowpassFilter);

    this->layoutGradientMap = runtime.createPipelineLayout(layout_2, {});
    this->pipelineGradientMap = runtime.createComputePipeline(this->kernelGradientMap, this->layoutGradientMap);
}

IQM::GPU::FSIMResult IQM::GPU::FSIM::computeMetric(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref) {
    assert(image.rows == ref.rows);
    assert(image.cols == ref.cols);

    FSIMResult result;

    const int F = computeDownscaleFactor(image.cols, image.rows);

    result.timestamps.mark("downscale factor computed");

    this->sendImagesToGpu(runtime, image, ref);

    result.timestamps.mark("images sent to gpu");

    const auto widthDownscale = static_cast<int>(std::round(static_cast<float>(image.cols) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<int>(std::round(static_cast<float>(image.rows) / static_cast<float>(F)));

    this->createDownscaledImages(runtime, widthDownscale, heightDownscale);
    this->computeDownscaledImages(runtime, F, widthDownscale, heightDownscale);

    result.timestamps.mark("images downscaled");

    this->createLowpassFilter(runtime, widthDownscale, heightDownscale);

    result.timestamps.mark("lowpass filter computed");

    this->createGradientMap(runtime, widthDownscale, heightDownscale);

    result.timestamps.mark("gradient map computed");

    this->computeFft(runtime, widthDownscale, heightDownscale);

    result.timestamps.mark("fft computed");

    result.image = image;

    result.fsim = 0.5;
    result.fsimc = 0.6;

    return result;
}

int IQM::GPU::FSIM::computeDownscaleFactor(const int cols, const int rows) {
    auto smallerDim = std::min(cols, rows);
    return std::max(1, static_cast<int>(std::round(smallerDim / 256.0)));
}

void IQM::GPU::FSIM::sendImagesToGpu(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref) {
    const auto size = image.rows * image.cols * image.channels();
    auto [stgBuf, stgMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [stgRefBuf, stgRefMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    auto imageParameters = ImageParameters(image.cols, image.rows);

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, image.data, imageParameters.height * imageParameters.width * 4);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, ref.data, imageParameters.height * imageParameters.width * 4);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(image.cols, image.rows, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    this->imageInput = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageRef = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = imageParameters.width,
        .bufferImageHeight = imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{imageParameters.width, imageParameters.height, 1}
    };
    runtime._cmd_buffer.copyBufferToImage(stgBuf, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_buffer.copyBufferToImage(stgRefBuf, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    runtime._cmd_buffer.end();

    const std::vector cmdBufsCopy = {
        &*runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eAllCommands};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue.submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();
}

void IQM::GPU::FSIM::createDownscaledImages(const VulkanRuntime &runtime, int width_downscale, int height_downscale) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = vk::Extent3D(width_downscale, height_downscale, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    vk::ImageCreateInfo imageFloatInfo = {imageInfo};
    imageFloatInfo.format = vk::Format::eR32Sfloat;
    imageFloatInfo.usage = vk::ImageUsageFlagBits::eStorage;

    this->imageInputDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->imageRefDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->imageLowpassFilter = std::make_shared<VulkanImage>(runtime.createImage(imageFloatInfo));
    this->imageGradientMapInput = std::make_shared<VulkanImage>(runtime.createImage(imageFloatInfo));
    this->imageGradientMapRef = std::make_shared<VulkanImage>(runtime.createImage(imageFloatInfo));

    auto imageInfosInput = VulkanRuntime::createImageInfos({
        this->imageInput,
        this->imageInputDownscaled,
    });

    auto imageInfosRef = VulkanRuntime::createImageInfos({
        this->imageRef,
        this->imageRefDownscaled,
    });

    auto imageInfosLPF = VulkanRuntime::createImageInfos({
        this->imageLowpassFilter,
    });

    auto imageInfosGradIn = VulkanRuntime::createImageInfos({
        this->imageInputDownscaled,
        this->imageGradientMapInput,
    });

    auto imageInfosGradRef = VulkanRuntime::createImageInfos({
        this->imageRefDownscaled,
        this->imageGradientMapRef,
    });

    const vk::WriteDescriptorSet writeSetInput{
        .dstSet = this->descSetDownscaleIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosInput.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetRef{
        .dstSet = this->descSetDownscaleRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetLPF{
        .dstSet = this->descSetLowpassFilter,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosLPF.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetGradIn{
        .dstSet = this->descSetGradientMapIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosGradIn.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetGradRef{
        .dstSet = this->descSetGradientMapRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosGradRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetRef, writeSetInput, writeSetLPF, writeSetGradIn, writeSetGradRef
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

void IQM::GPU::FSIM::computeDownscaledImages(const VulkanRuntime &runtime, const int F, const int width, const int height) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInputDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRefDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineDownscale);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleIn}, {});

    int useColor = this->doColorComparison;

    std::array values = {
        F,
        useColor,
    };
    runtime._cmd_buffer.pushConstants<int>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, values);

    //shader works in 8x8 tiles
    constexpr unsigned tileSize = 8;

    auto groupsX = width / tileSize;
    if (width % tileSize != 0) {
        groupsX++;
    }
    auto groupsY = height / tileSize;
    if (height % tileSize != 0) {
        groupsY++;
    }

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleRef}, {});
    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

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

void IQM::GPU::FSIM::createLowpassFilter(const VulkanRuntime &runtime, const int width, const int height) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageLowpassFilter->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineLowpassFilter);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutLowpassFilter, 0, {this->descSetLowpassFilter}, {});

    int order = 15;

    std::array<float, 2> values = {
        0.45,
        *reinterpret_cast<float *>(&order),
    };
    runtime._cmd_buffer.pushConstants<float>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, values);

    //shader works in 16x16 tiles
    constexpr unsigned tileSize = 16;

    auto groupsX = width / tileSize;
    if (width % tileSize != 0) {
        groupsX++;
    }
    auto groupsY = height / tileSize;
    if (height % tileSize != 0) {
        groupsY++;
    }

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

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

void IQM::GPU::FSIM::createGradientMap(const VulkanRuntime &runtime, int width, int height) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageGradientMapInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageGradientMapRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGradientMap);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapIn}, {});

    //shader works in 8x8 tiles
    constexpr unsigned tileSize = 8;

    auto groupsX = width / tileSize;
    if (width % tileSize != 0) {
        groupsX++;
    }
    auto groupsY = height / tileSize;
    if (height % tileSize != 0) {
        groupsY++;
    }

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapRef}, {});

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

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

void IQM::GPU::FSIM::computeFft(const VulkanRuntime &runtime, const int width, const int height) {
    VkFFTApplication fftApp = {};

    VkFFTConfiguration fftConfig = {};
    fftConfig.FFTdim = 2;
    fftConfig.size[0] = width;
    fftConfig.size[1] = height;
    uint64_t bufferSize = width * height * sizeof(float);
    fftConfig.bufferSize = &bufferSize;

    VkDevice deviceRef = *runtime._device;
    VkPhysicalDevice physDeviceRef = *runtime._physicalDevice;
    VkQueue queueRef = *runtime._queue;
    VkCommandPool cmdPoolRef = *runtime._commandPool;
    fftConfig.physicalDevice = &physDeviceRef;
    fftConfig.device = &deviceRef;
    fftConfig.queue = &queueRef;
    fftConfig.commandPool = &cmdPoolRef;

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};
    VkFence fenceRef = *fence;
    fftConfig.fence = &fenceRef;

    if (auto res = initializeVkFFT(&fftApp, fftConfig); res != VKFFT_SUCCESS) {
        throw std::runtime_error("failed to initialize FFT");
    }

    deleteVkFFT(&fftApp);
}