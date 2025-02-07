/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "ssim.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"

using IQM::Bin::InputImage;
using IQM::Bin::VulkanImage;
using IQM::Bin::VulkanInstance;
using IQM::Bin::VulkanResource;

IQM::Bin::SSIMResources IQM::Bin::ssim_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance& instance) {
    // always 4 channels on input, with 1B per channel
    // add 1 float to end so buffer can be reused for writeback from GPU
    const auto size = (test.width * test.height + 1) * 4;
    auto [stgBuf, stgMem] = VulkanResource::createBuffer(
        instance.device,
        instance.physicalDevice,
        size,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached
    );
    auto [stgRefBuf, stgRefMem] = VulkanResource::createBuffer(
    instance.device,
    instance.physicalDevice,
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [mssimBuf, mssimMem] = VulkanResource::createBuffer(
        instance.device,
        instance.physicalDevice,
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    mssimBuf.bindMemory(mssimMem, 0);

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, test.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(test.width, test.height, 1),
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

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, srcImageInfo));
    auto const imageOut = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, dstImageInfo));
    auto imagesBlurred = std::vector<std::shared_ptr<VulkanImage>>();

    for (int i = 0; i < 5; i++) {
        imagesBlurred.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, intermediateImageInfo)));
    }

    return SSIMResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .mssimBuf = std::move(mssimBuf),
        .mssimMemory = std::move(mssimMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imagesBlurred = imagesBlurred,
        .imageOut = imageOut,
        .uploadDone = instance.device.createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device.createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device.createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::ssim_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches) {
    IQM::SSIM ssim(instance.device);

    int processed = 0;

    for (const auto& match : imageMatches) {
        try {
            Timestamps timestamps;
            auto start = std::chrono::high_resolution_clock::now();

            const auto input = load_image(match.testPath);
            const auto reference = load_image(match.refPath);
            if (input.height != reference.height || input.width != reference.width) {
                throw std::runtime_error("Test and reference images have different sizes");
            }

            timestamps.mark("images loaded");

            initRenderDoc();

            auto res = ssim_init_res(input, reference, instance);
            timestamps.mark("resources allocated");

            ssim_upload(instance, res);

            auto ssimArgs = IQM::SSIMInput {
                .device = &instance.device,
                .cmdBuf = &*instance.cmd_buffer,
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .ivMeanTest = &res.imagesBlurred[0]->imageView,
                .ivMeanRef = &res.imagesBlurred[1]->imageView,
                .ivVarTest = &res.imagesBlurred[2]->imageView,
                .ivVarRef = &res.imagesBlurred[3]->imageView,
                .ivCovar = &res.imagesBlurred[4]->imageView,
                .ivOut = &res.imageOut->imageView,
                .imgOut = &res.imageOut->image,
                .bufMssim = &res.mssimBuf,
                .width = input.width,
                .height = input.height
            };

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmd_buffer->begin(beginInfo);

            ssim.computeMetric(ssimArgs);

            instance.cmd_buffer->end();

            const std::vector cmdBufs = {
                &**instance.cmd_buffer
            };

            auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*res.uploadDone,
                .pWaitDstStageMask = &mask,
                .commandBufferCount = 1,
                .pCommandBuffers = *cmdBufs.data(),
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*res.computeDone
            };

            instance.queue->submit(submitInfo, {});
            timestamps.mark("submit compute GPU pipeline");
            // wait so cmd buffer can be reused for GPU -> CPU transfer
            VulkanInstance::waitForFence(instance.device, res.transferFence);

            auto result = ssim_copy_back(instance, res, timestamps, ssim.kernelSize);

            finishRenderDoc();

            if (match.outPath.has_value()) {
                save_float_image(args.outputPath.value(), result.imageData, input.width, input.height);
            }

            timestamps.mark("output saved");

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.mssim << std::endl;
            if (args.verbose) {
                timestamps.print(start, end);
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to process '" << match.testPath << "': " << e.what() << std::endl;
            continue;
        }

        processed += 1;
    }

    std::cout << "Processed " << processed << "/" << imageMatches.size() <<" images" << std::endl;
}

void IQM::Bin::ssim_upload(const VulkanInstance &instance, const SSIMResources &res) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmd_bufferTransfer->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
        res.imageOut,
    };

    imagesToInit.insert(imagesToInit.end(), res.imagesBlurred.begin(), res.imagesBlurred.end());

    VulkanResource::initImages(*instance.cmd_bufferTransfer, imagesToInit);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = res.imageInput->width,
        .bufferImageHeight = res.imageInput->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
    };
    instance.cmd_bufferTransfer->copyBufferToImage(res.stgInput, res.imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmd_bufferTransfer->copyBufferToImage(res.stgRef, res.imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    instance.cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**instance.cmd_bufferTransfer
    };

    const vk::SubmitInfo submitInfoCopy{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*res.uploadDone
    };

    instance.transferQueue->submit(submitInfoCopy, res.transferFence);
}

IQM::Bin::SSIMResult IQM::Bin::ssim_copy_back(const VulkanInstance &instance, const SSIMResources &res, Timestamps &timestamps, uint32_t kernelSize) {
    SSIMResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmd_bufferTransfer->begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = res.imageOut->width,
        .bufferImageHeight = res.imageOut->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageOut->width, res.imageOut->height, 1}
    };
    instance.cmd_bufferTransfer->copyImageToBuffer(res.imageOut->image,  vk::ImageLayout::eGeneral, res.stgInput, copyRegion);
    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = sizeof(float) * res.imageOut->width * res.imageOut->height,
        .size = sizeof(float),
    };
    instance.cmd_bufferTransfer->copyBuffer(res.mssimBuf, res.stgInput, bufCopy);

    instance.cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**instance.cmd_bufferTransfer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*res.computeDone,
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{instance.device, vk::FenceCreateInfo{}};

    instance.transferQueue->submit(submitInfoCopy, *fenceCopy);
    instance.device.waitIdle();

    timestamps.mark("end GPU work");

    const auto offset = kernelSize - 1;
    std::vector<float> outputData(res.imageOut->height * res.imageOut->width);
    void * outBufData = res.stgInputMemory.mapMemory(0, (res.imageOut->height * res.imageOut->width + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, res.imageOut->height * res.imageOut->width * sizeof(float));
    result.mssim = (static_cast<float*>(outBufData))[res.imageOut->height * res.imageOut->width] / (static_cast<float>(res.imageOut->width - offset) * static_cast<float>(res.imageOut->height - offset));

    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    result.imageData = std::move(outputData);

    return result;
}
