/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "fsim.h"

#include "fft_planner.h"
#include "../img_params.h"

static std::vector<uint32_t> srcDownscale =
#include <fsim/fsim_downsample.inc>
;

static std::vector<uint32_t> srcGradient =
#include <fsim/fsim_gradientmap.inc>
;

static std::vector<uint32_t> srcExtractLuma =
#include <fsim/fsim_extractluma.inc>
;

IQM::GPU::FSIM::FSIM(const vk::raii::Device &device):
descPool(VulkanRuntime::createDescPool(device, 64, {
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 128},
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 32}
})),
lowpassFilter(device, descPool),
logGaborFilter(device, descPool),
angularFilter(device, descPool),
combinations(device, descPool),
sumFilterResponses(device, descPool),
noise_power(device, descPool),
estimateEnergy(device, descPool),
phaseCongruency(device, descPool),
final_multiply(device, descPool)
{
    const auto smDownscale = VulkanRuntime::createShaderModule(device, srcDownscale);
    const auto smGradientMap = VulkanRuntime::createShaderModule(device, srcGradient);
    const auto smExtractLuma = VulkanRuntime::createShaderModule(device, srcExtractLuma);

    this->descSetLayoutImageOp = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->descSetLayoutImBufOp = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allLayouts = {
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImBufOp,
        *this->descSetLayoutImBufOp,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allLayouts.size()),
        .pSetLayouts = allLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetDownscaleIn = std::move(sets[0]);
    this->descSetDownscaleRef = std::move(sets[1]);
    this->descSetGradientMapIn = std::move(sets[2]);
    this->descSetGradientMapRef = std::move(sets[3]);
    this->descSetExtractLumaIn = std::move(sets[4]);
    this->descSetExtractLumaRef = std::move(sets[5]);

    // 1x int - kernel size
    const auto downsampleRanges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layoutDownscale = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImageOp}, downsampleRanges);
    this->pipelineDownscale = VulkanRuntime::createComputePipeline(device, smDownscale, this->layoutDownscale);

    this->layoutGradientMap = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImageOp}, {});
    this->pipelineGradientMap = VulkanRuntime::createComputePipeline(device, smGradientMap, this->layoutGradientMap);

    this->layoutExtractLuma = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImBufOp}, {});
    this->pipelineExtractLuma = VulkanRuntime::createComputePipeline(device, smExtractLuma, this->layoutExtractLuma);
}

IQM::GPU::FSIMResult IQM::GPU::FSIM::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    FSIMResult result;

    const int F = computeDownscaleFactor(image.width, image.height);

    result.timestamps.mark("downscale factor computed");

    this->sendImagesToGpu(runtime, image, ref);

    result.timestamps.mark("images sent to gpu");

    const auto widthDownscale = static_cast<int>(std::round(static_cast<float>(image.width) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<int>(std::round(static_cast<float>(image.height) / static_cast<float>(F)));

    this->initFftLibrary(runtime, widthDownscale, heightDownscale);
    result.timestamps.mark("FFT library initialized");

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    this->createDownscaledImages(runtime, widthDownscale, heightDownscale);
    this->computeDownscaledImages(runtime, F, widthDownscale, heightDownscale);
    this->lowpassFilter.constructFilter(runtime, widthDownscale, heightDownscale);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        nullptr,
        nullptr
    );

    this->createGradientMap(runtime, widthDownscale, heightDownscale);
    this->logGaborFilter.constructFilter(runtime, this->lowpassFilter.imageLowpassFilter, widthDownscale, heightDownscale);
    this->angularFilter.constructFilter(runtime, widthDownscale, heightDownscale);

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        nullptr,
        nullptr
    );

    this->computeFft(runtime, widthDownscale, heightDownscale);
    this->combinations.combineFilters(runtime, this->angularFilter, this->logGaborFilter, this->bufferFft, widthDownscale, heightDownscale);
    this->computeMassInverseFft(runtime, this->combinations.fftBuffer);
    this->sumFilterResponses.computeSums(runtime, this->combinations.fftBuffer, widthDownscale, heightDownscale);;
    this->noise_power.computeNoisePower(runtime, this->combinations.noiseLevels, this->combinations.fftBuffer, widthDownscale, heightDownscale);
    this->estimateEnergy.estimateEnergy(runtime, this->combinations.fftBuffer, widthDownscale, heightDownscale);
    this->phaseCongruency.compute(runtime, this->noise_power.noisePowers, this->estimateEnergy.energyBuffers, this->sumFilterResponses.filterResponsesInput, this->sumFilterResponses.filterResponsesRef, widthDownscale, heightDownscale);

    auto metrics = this->final_multiply.computeMetrics(
        runtime,
        {this->imageInputDownscaled, this->imageRefDownscaled},
        {this->imageGradientMapInput, this->imageGradientMapRef},
        {this->phaseCongruency.pcInput, this->phaseCongruency.pcRef},
        widthDownscale,
        heightDownscale
    );
    result.timestamps.mark("FSIM, FSIMc computed");

    result.fsim = metrics.first;
    result.fsimc = metrics.second;

    this->teardownFftLibrary();

    return result;
}

int IQM::GPU::FSIM::computeDownscaleFactor(const int width, const int height) {
    auto smallerDim = std::min(width, height);
    return std::max(1, static_cast<int>(std::round(smallerDim / 256.0)));
}

void IQM::GPU::FSIM::sendImagesToGpu(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    const auto size = image.width * image.height * 4;
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

    auto imageParameters = ImageParameters(image.width, image.height);

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, image.data.data(), imageParameters.height * imageParameters.width * 4);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, ref.data.data(), imageParameters.height * imageParameters.width * 4);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(image.width, image.height, 1),
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
    runtime._cmd_bufferTransfer->begin(beginInfo);

    runtime.setImageLayout(runtime._cmd_bufferTransfer, this->imageInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_bufferTransfer, this->imageRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = imageParameters.width,
        .bufferImageHeight = imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{imageParameters.width, imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyBufferToImage(stgBuf, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_bufferTransfer->copyBufferToImage(stgRefBuf, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._transferQueue->submit(submitInfoCopy, *fenceCopy);
    runtime.waitForFence(fenceCopy);
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

    vk::ImageCreateInfo imageFftInfo = {imageFloatInfo};
    imageFftInfo.format = vk::Format::eR32G32Sfloat;
    imageFftInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;

    this->imageInputDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->imageRefDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
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

    auto imageInfosGradIn = VulkanRuntime::createImageInfos({
        this->imageInputDownscaled,
        this->imageGradientMapInput,
    });

    auto imageInfosGradRef = VulkanRuntime::createImageInfos({
        this->imageRefDownscaled,
        this->imageGradientMapRef,
    });

    auto writeSetInput = VulkanRuntime::createWriteSet(
        this->descSetDownscaleIn,
        0,
        imageInfosInput
    );

    auto writeSetRef = VulkanRuntime::createWriteSet(
        this->descSetDownscaleRef,
        0,
        imageInfosRef
    );

    auto writeSetGradIn = VulkanRuntime::createWriteSet(
        this->descSetGradientMapIn,
        0,
        imageInfosGradIn
    );

    auto writeSetGradRef = VulkanRuntime::createWriteSet(
        this->descSetGradientMapRef,
        0,
        imageInfosGradRef
    );

    const std::vector writes = {
        writeSetRef, writeSetInput, writeSetGradIn, writeSetGradRef
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

void IQM::GPU::FSIM::computeDownscaledImages(const VulkanRuntime &runtime, const int F, const int width, const int height) {
    runtime.setImageLayout(runtime._cmd_buffer, this->imageInputDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->imageRefDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineDownscale);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleIn}, {});

    runtime._cmd_buffer->pushConstants<int>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, F);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleRef}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FSIM::createGradientMap(const VulkanRuntime &runtime, int width, int height) {
    runtime.setImageLayout(runtime._cmd_buffer, this->imageGradientMapInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->imageGradientMapRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGradientMap);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapIn}, {});

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapRef}, {});

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FSIM::initFftLibrary(const VulkanRuntime &runtime, const int width, const int height) {
    this->fftFence =  vk::raii::Fence{runtime._device, vk::FenceCreateInfo{}};
    this->fftFenceInverse =  vk::raii::Fence{runtime._device, vk::FenceCreateInfo{}};

    this->fftApplication = FftPlanner::initForward(runtime, this->fftFence, width, height);
    this->fftApplicationInverse = FftPlanner::initInverse(runtime, this->fftFenceInverse, width, height);
}

void IQM::GPU::FSIM::teardownFftLibrary() {
    FftPlanner::destroy(this->fftApplication);
    FftPlanner::destroy(this->fftApplicationInverse);
}

void IQM::GPU::FSIM::computeFft(const VulkanRuntime &runtime, const int width, const int height) {
    // image size * 2 float components (complex numbers) * 2 batches
    uint64_t bufferSize = width * height * sizeof(float) * 2 * 2;

    auto [fftBuf, fftMem] = runtime.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    fftBuf.bindMemory(fftMem, 0);

    this->memoryFft = std::move(fftMem);
    this->bufferFft = std::move(fftBuf);

    std::vector bufIn = {
        vk::DescriptorBufferInfo{
            .buffer = this->bufferFft,
            .offset = 0,
            .range = bufferSize / 2,
        }
    };
    auto imInfoInImage = VulkanRuntime::createImageInfos({
        this->imageInputDownscaled,
    });
    auto writeSetInImage = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaIn,
        0,
        imInfoInImage
    );

    auto writeSetInBuf = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaIn,
        1,
        bufIn
    );

    std::vector bufRef = {
        vk::DescriptorBufferInfo{
            .buffer = this->bufferFft,
            .offset = bufferSize / 2,
            .range = bufferSize / 2,
        }
    };

    auto imInfoRefImage = VulkanRuntime::createImageInfos({
        this->imageRefDownscaled,
    });
    auto writeSetRefImage = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaRef,
        0,
        imInfoRefImage
    );

    auto writeSetRefBuf = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaRef,
        1,
        bufRef
    );

    const std::vector writes = {
        writeSetInImage, writeSetInBuf, writeSetRefImage, writeSetRefBuf
    };

    runtime._device.updateDescriptorSets(writes, nullptr);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineExtractLuma);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaIn}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaRef}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {barrier},
        nullptr,
        nullptr
    );

    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **runtime._cmd_buffer;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = *this->bufferFft;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(&this->fftApplication, -1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append FFT: " + std::to_string(res);
        throw std::runtime_error(err);
    }
}

void IQM::GPU::FSIM::computeMassInverseFft(const VulkanRuntime &runtime, const vk::raii::Buffer &buffer) {

    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **runtime._cmd_buffer;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = *buffer;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(&this->fftApplicationInverse, 1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append inverse FFT: " + std::to_string(res);
        throw std::runtime_error(err);
    }
}
