/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "ssim.h"

static std::vector<uint32_t> src =
#include <ssim/ssim.inc>
;

static std::vector<uint32_t> srcLumapack =
#include <ssim/ssim_lumapack.inc>
;

static std::vector<uint32_t> srcGaussHorizontal =
#include <ssim/ssim_gauss_horizontal.inc>
;

static std::vector<uint32_t> srcGauss =
#include <ssim/ssim_gauss.inc>
;

static std::vector<uint32_t> srcMssim =
#include <ssim/mssim_sum.inc>
;

IQM::GPU::SSIM::SSIM(const VulkanRuntime &runtime) {
    const auto smSsim = VulkanRuntime::createShaderModule(runtime._device, src);
    const auto smLumapack = VulkanRuntime::createShaderModule(runtime._device,srcLumapack);
    const auto smGaussHorizontal = VulkanRuntime::createShaderModule(runtime._device, srcGaussHorizontal);
    const auto smGauss = VulkanRuntime::createShaderModule(runtime._device, srcGauss);
    const auto smMssim = VulkanRuntime::createShaderModule(runtime._device, srcMssim);

    this->descSetLayoutLumapack = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 5},
    });

    this->descSetLayoutSsim = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 5},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->descSetLayoutMssim = runtime.createDescLayout({
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allocateLayouts = {
        *this->descSetLayoutLumapack,
        *this->descSetLayoutSsim,
        *this->descSetLayoutSsim,
        *this->descSetLayoutMssim
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allocateLayouts.size()),
        .pSetLayouts = allocateLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSetLumapack = std::move(sets[0]);
    this->descSetGauss = std::move(sets[1]);
    this->descSetSsim = std::move(sets[2]);
    this->descSetMssim = std::move(sets[3]);

    // 1x int - kernel size
    // 3x float - K_1, K_2, sigma
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1 + sizeof(float) * 3);

    // 2x int - kernel size, kernel direction
    // 1x float - sigma
    const auto rangesGauss = VulkanRuntime::createPushConstantRange(2 * sizeof(int) + sizeof(float));

    // 1x int - data size
    const auto rangeMssim = VulkanRuntime::createPushConstantRange( sizeof(int));

    this->layoutLumapack = runtime.createPipelineLayout({*this->descSetLayoutLumapack}, {});
    this->layoutGauss = runtime.createPipelineLayout({*this->descSetLayoutSsim}, rangesGauss);
    this->layoutSsim = runtime.createPipelineLayout({*this->descSetLayoutSsim}, ranges);
    this->layoutMssim = runtime.createPipelineLayout({*this->descSetLayoutMssim}, rangeMssim);

    this->pipelineLumapack = runtime.createComputePipeline(smLumapack, this->layoutLumapack);
    this->pipelineGauss = runtime.createComputePipeline(smGauss, this->layoutGauss);
    this->pipelineGaussHorizontal = runtime.createComputePipeline(smGaussHorizontal, this->layoutGauss);
    this->pipelineSsim = runtime.createComputePipeline(smSsim, this->layoutSsim);
    this->pipelineMssim = runtime.createComputePipeline(smMssim, this->layoutMssim);

    this->uploadDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->computeDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->transferFence = runtime._device.createFence(vk::FenceCreateInfo{});

    this->imagesBlurred = std::vector<std::shared_ptr<VulkanImage>>(5, VK_NULL_HANDLE);
}

IQM::GPU::SSIMResult IQM::GPU::SSIM::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    runtime._device.resetFences({this->transferFence});
    SSIMResult res;

    this->prepareImages(runtime, res.timestamps, image, ref);

    res.timestamps.mark("start GPU pipeline");

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineLumapack);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutLumapack, 0, {this->descSetLumapack}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(this->imageParameters.width, this->imageParameters.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    std::vector<vk::ImageMemoryBarrier> barriers(5);

    for (size_t i = 0; i < barriers.size(); i++) {
        barriers[i] = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = this->imagesBlurred[i]->image,
            .subresourceRange = vk::ImageSubresourceRange {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
    }

    vk::ImageMemoryBarrier barrierTemp = barriers[0];
    barrierTemp.image = this->imageBlurredTemp->image;

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, barriers
    );

    for (int i = 0; i < 5; i++) {
        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGaussHorizontal);
        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGauss, 0, {this->descSetGauss}, {});

        std::array valuesGauss = {
            this->kernelSize,
            *reinterpret_cast<int *>(&this->sigma),
            i,
        };
        runtime._cmd_buffer->pushConstants<int>(this->layoutGauss, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {}, {}, barrierTemp
        );

        valuesGauss = {
            this->kernelSize,
            *reinterpret_cast<int *>(&this->sigma),
            i,
        };

        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGauss);
        runtime._cmd_buffer->pushConstants<int>(this->layoutGauss, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {}, {}, barriers
        );
    }

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSsim);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSsim, 0, {this->descSetSsim}, {});

    std::array values = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->k_1),
        *reinterpret_cast<int *>(&this->k_2),
        *reinterpret_cast<int *>(&this->sigma)
    };
    runtime._cmd_buffer->pushConstants<int>(this->layoutSsim, vk::ShaderStageFlagBits::eCompute, 0, values);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    const auto halfOffset = (this->kernelSize - 1) / 2;
    const auto offset = this->kernelSize - 1;

    vk::BufferImageCopy copyMssimRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width - offset,
        .bufferImageHeight = this->imageParameters.height - offset,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{halfOffset, halfOffset, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width - offset, this->imageParameters.height - offset, 1}
    };
    runtime._cmd_buffer->copyImageToBuffer(this->imageOut->image,  vk::ImageLayout::eGeneral, this->mssimBuf, copyMssimRegion);

    memBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    memBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineMssim);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutMssim, 0, {this->descSetMssim}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = (image.width - offset) * (image.height - offset);
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;

    for (;;) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutMssim, vk::ShaderStageFlagBits::eCompute, 0, size);
        runtime._cmd_buffer->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = this->mssimBuf,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );
        if (groups == 1) {
            break;
        }
        size = groups;
        groups = (groups / sumSize) + 1;
    }

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*this->uploadDone,
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*this->computeDone
    };

    runtime._queue->submit(submitInfo, {});
    res.timestamps.mark("submit compute GPU pipeline");
    runtime.waitForFence(this->transferFence);

    // allocate one float extra for image sum to compute MSSIM
    const auto outSize = (this->imageParameters.height * this->imageParameters.width + 1) * sizeof(float);
    auto [stgBuf, stgMem] = runtime.createBuffer(
        outSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
    );
    stgBuf.bindMemory(stgMem, 0);

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyImageToBuffer(this->imageOut->image,  vk::ImageLayout::eGeneral, stgBuf, copyRegion);
    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = sizeof(float) * this->imageParameters.width * this->imageParameters.height,
        .size = sizeof(float),
    };
    runtime._cmd_bufferTransfer->copyBuffer(this->mssimBuf, stgBuf, bufCopy);

    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*this->computeDone,
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._transferQueue->submit(submitInfoCopy, *fenceCopy);
    res.timestamps.mark("submit copy back GPU pipeline");
    runtime._device.waitIdle();

    res.timestamps.mark("end GPU pipeline");

    std::vector<float> outputData(this->imageParameters.height * this->imageParameters.width);
    void * outBufData = stgMem.mapMemory(0, (this->imageParameters.height * this->imageParameters.width + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, this->imageParameters.height * this->imageParameters.width * sizeof(float));
    res.mssim = (static_cast<float*>(outBufData))[this->imageParameters.height * this->imageParameters.width] / (static_cast<float>(this->imageParameters.width - offset) * static_cast<float>(this->imageParameters.height - offset));

    stgMem.unmapMemory();
    res.timestamps.mark("end copy from GPU");

    res.imageData = std::move(outputData);
    res.height = this->imageParameters.height;
    res.width = this->imageParameters.width;

    return res;
}

void IQM::GPU::SSIM::prepareImages(const VulkanRuntime &runtime, Timestamps& timestamps, const InputImage &image, const InputImage &ref) {
    // always 4 channels on input, with 1B per channel
    const auto size = image.width * image.height * 4;
    const auto sizeTrimmed = (image.width - this->kernelSize + 1) * (image.height - this->kernelSize + 1) * 4;
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
    auto [mssimBuf, mssimMem] = runtime.createBuffer(
        sizeTrimmed,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    this->imageParameters.height = image.height;
    this->imageParameters.width = image.width;

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    mssimBuf.bindMemory(mssimMem, 0);

    timestamps.mark("buffers created");

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, image.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    timestamps.mark("buffers copied");

    this->stgInput = std::move(stgBuf);
    this->stgInputMemory = std::move(stgMem);
    this->stgRef = std::move(stgRefBuf);
    this->stgRefMemory = std::move(stgRefMem);
    this->mssimBuf = std::move(mssimBuf);
    this->mssimMemory = std::move(mssimMem);

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

    vk::ImageCreateInfo lumaImageInfo {srcImageInfo};
    lumaImageInfo.format = vk::Format::eR32G32Sfloat;
    lumaImageInfo.usage = vk::ImageUsageFlagBits::eStorage;

    vk::ImageCreateInfo intermediateImageInfo = {srcImageInfo};
    intermediateImageInfo.usage = vk::ImageUsageFlagBits::eStorage;
    intermediateImageInfo.format = vk::Format::eR32Sfloat;

    vk::ImageCreateInfo dstImageInfo = {srcImageInfo};
    dstImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    dstImageInfo.format = vk::Format::eR32Sfloat;

    this->imageInput = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageRef = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageOut = std::make_shared<VulkanImage>(runtime.createImage(dstImageInfo));
    this->imageBlurredTemp = std::make_shared<VulkanImage>(runtime.createImage(intermediateImageInfo));

    for (int i = 0; i < 5; i++) {
        this->imagesBlurred[i] = std::move(std::make_shared<VulkanImage>(runtime.createImage(intermediateImageInfo)));
    }

    timestamps.mark("images created");

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfo);

    std::vector imagesToInit = {
        this->imageInput,
        this->imageRef,
        this->imageOut,
        this->imageBlurredTemp,
    };

    imagesToInit.insert(imagesToInit.end(), this->imagesBlurred.begin(), this->imagesBlurred.end());

    VulkanRuntime::initImages(runtime._cmd_bufferTransfer, imagesToInit);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyBufferToImage(this->stgInput, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_bufferTransfer->copyBufferToImage(this->stgRef, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    const vk::SubmitInfo submitInfoCopy{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*this->uploadDone
    };

    runtime._transferQueue->submit(submitInfoCopy, this->transferFence);

    timestamps.mark("copy sumbited");

    auto imageInfosIntermediate = VulkanRuntime::createImageInfos(this->imagesBlurred);
    auto imageInfosIntermediateTemp = VulkanRuntime::createImageInfos({this->imageBlurredTemp});

    auto inputImageInfos = VulkanRuntime::createImageInfos({
        this->imageInput,
        this->imageRef,
    });

    auto outImageInfos = VulkanRuntime::createImageInfos({
        this->imageOut,
    });

    std::vector bufInfos = {
        vk::DescriptorBufferInfo{
            .buffer = this->mssimBuf,
            .offset = 0,
            .range = static_cast<unsigned>(sizeTrimmed),
        }
    };

    // input
    auto writeSetLumapackIn = VulkanRuntime::createWriteSet(
        this->descSetLumapack,
        0,
        inputImageInfos
    );

    auto writeSetLumapackOut = VulkanRuntime::createWriteSet(
        this->descSetLumapack,
        1,
        imageInfosIntermediate
    );

    // gauss
    auto writeSetGaussFlip = VulkanRuntime::createWriteSet(
        this->descSetGauss,
        0,
        imageInfosIntermediate
    );

    auto writeSetGaussFlop = VulkanRuntime::createWriteSet(
        this->descSetGauss,
        1,
        imageInfosIntermediateTemp
    );

    // ssim
    auto writeSetSsimIn = VulkanRuntime::createWriteSet(
        this->descSetSsim,
        0,
        imageInfosIntermediate
    );

    auto writeSetSssimOut = VulkanRuntime::createWriteSet(
        this->descSetSsim,
        1,
        outImageInfos
    );

    // mssim
    auto writeSetSum = VulkanRuntime::createWriteSet(
        this->descSetMssim,
        0,
        bufInfos
    );

    runtime._device.updateDescriptorSets({writeSetLumapackIn, writeSetLumapackOut, writeSetGaussFlip, writeSetGaussFlop, writeSetSsimIn, writeSetSssimOut, writeSetSum}, nullptr);
}
