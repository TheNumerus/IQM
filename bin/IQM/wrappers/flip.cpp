/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "flip.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"
#include "IQM/flip/viridis.h"

void IQM::Bin::flip_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches) {
    IQM::FLIP flip(instance.device);

    FLIPArguments flipArgs;
    if (args.options.contains("--flip-width")) {
        flipArgs.monitor_width = std::stof(args.options.at("--flip-width"));
    }
    if (args.options.contains("--flip-res")) {
        flipArgs.monitor_resolution_x = std::stof(args.options.at("--flip-res"));
    }
    if (args.options.contains("--flip-distance")) {
        flipArgs.monitor_distance = std::stof(args.options.at("--flip-distance"));
    }

    if (args.verbose) {
        std::cout << "FLIP monitor resolution: "<< flipArgs.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flipArgs.monitor_distance << std::endl
        << "FLIP monitor width: "<< flipArgs.monitor_width << std::endl;
    }

    unsigned spatialKernelSize = IQM::FLIP::spatialKernelSize(flipArgs);
    unsigned featureKernelSize = IQM::FLIP::featureKernelSize(flipArgs);

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

            auto res = flip_init_res(input, reference, instance, spatialKernelSize, featureKernelSize);
            timestamps.mark("resources allocated");

            flip_upload(instance, res);

            auto flipInput = IQM::FLIPInput {
                .args = flipArgs,
                .device = &instance.device,
                .cmdBuf = &*instance.cmd_buffer,
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .ivOut = &res.imageOut->imageView,
                .ivColorMap = &res.imageColorMap->imageView,
                .ivFeatErr = &res.imagesFloatTemp[0]->imageView,
                .ivColorErr = &res.imagesFloatTemp[1]->imageView,
                .ivFeatFilter = &res.imageFeatureFilter->imageView,
                .ivColorFilter = &res.imageColorFilter->imageView,
                .ivTemp = {
                    &res.imagesColorTemp[0]->imageView,
                    &res.imagesColorTemp[1]->imageView,
                    &res.imagesColorTemp[2]->imageView,
                    &res.imagesColorTemp[3]->imageView,
                    &res.imagesColorTemp[4]->imageView,
                    &res.imagesColorTemp[5]->imageView,
                    &res.imagesColorTemp[6]->imageView,
                    &res.imagesColorTemp[7]->imageView,
                },
                .imgOut = &res.imageOut->image,
                .bufMean = &res.meanBuf,
                .width = input.width,
                .height = input.height
            };

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmd_buffer->begin(beginInfo);

            flip.computeMetric(flipInput);

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

            auto result = flip_copy_back(instance, res, timestamps);

            finishRenderDoc();

            if (match.outPath.has_value()) {
                save_float_color_image(args.outputPath.value(), result.imageData, input.width, input.height);
            }

            timestamps.mark("output saved");

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.meanFlip << std::endl;
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

IQM::Bin::FLIPResources IQM::Bin::flip_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance &instance, unsigned spatialKernelSize, unsigned featureKernelSize) {
    // always 4 channels on input, with 1B per channel
    // add 1 float to end so buffer can be reused for writeback from GPU
    const auto outSize = ((test.width * test.height * 4) + 1) * sizeof(float);
    const auto size = (test.width * test.height ) * sizeof(float);
    const auto colormapSize = 256 * 4 * sizeof(float);
    auto [stgBuf, stgMem] = VulkanResource::createBuffer(
        instance.device,
        instance.physicalDevice,
        outSize,
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
    auto [cmBuf, cmMem] = VulkanResource::createBuffer(
        instance.device,
        instance.physicalDevice,
        colormapSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [meanBuf, meanMem] = VulkanResource::createBuffer(
        instance.device,
        instance.physicalDevice,
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    cmBuf.bindMemory(cmMem, 0);
    meanBuf.bindMemory(meanMem, 0);

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, test.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    inBufData = cmMem.mapMemory(0, colormapSize, {});
    memcpy(inBufData, viridis, colormapSize);
    cmMem.unmapMemory();

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

    vk::ImageCreateInfo floatImageInfo = {srcImageInfo};
    floatImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    floatImageInfo.format = vk::Format::eR32Sfloat;

    vk::ImageCreateInfo colorImageInfo = {srcImageInfo};
    colorImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    colorImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo spatialImageInfo = {srcImageInfo};
    spatialImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    spatialImageInfo.extent = vk::Extent3D(spatialKernelSize, spatialKernelSize, 1);
    spatialImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo featureImageInfo = {srcImageInfo};
    featureImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    featureImageInfo.extent = vk::Extent3D(featureKernelSize, 1, 1);
    featureImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo colorMapImageInfo = {srcImageInfo};
    colorMapImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
    colorMapImageInfo.extent = vk::Extent3D(256, 1, 1);
    colorMapImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, srcImageInfo));
    auto const imageOut = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, colorImageInfo));
    auto const imageColorFilter = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, spatialImageInfo));
    auto const imageFeatureFilter = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, featureImageInfo));
    auto const imageColorMap = std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, colorMapImageInfo));
    auto imagesColorTemp = std::vector<std::shared_ptr<VulkanImage>>();
    auto imagesFloatTemp = std::vector<std::shared_ptr<VulkanImage>>();

    for (int i = 0; i < 8; i++) {
        imagesColorTemp.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, colorImageInfo)));
    }

    for (int i = 0; i < 2; i++) {
        imagesFloatTemp.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(instance.device, instance.physicalDevice, floatImageInfo)));
    }

    return FLIPResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .stgColormap = std::move(cmBuf),
        .stgColormapMemory = std::move(cmMem),
        .meanBuf = std::move(meanBuf),
        .meanMemory = std::move(meanMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imagesFloatTemp = imagesFloatTemp,
        .imagesColorTemp = imagesColorTemp,
        .imageColorFilter = imageColorFilter,
        .imageFeatureFilter = imageFeatureFilter,
        .imageColorMap = imageColorMap,
        .imageOut = imageOut,
        .uploadDone = instance.device.createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device.createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device.createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::flip_upload(const VulkanInstance &instance, const FLIPResources &res) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmd_bufferTransfer->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
        res.imageOut,
        res.imageColorFilter,
        res.imageFeatureFilter,
        res.imageColorMap,
        res.imageOut,
    };

    imagesToInit.insert(imagesToInit.end(), res.imagesColorTemp.begin(), res.imagesColorTemp.end());
    imagesToInit.insert(imagesToInit.end(), res.imagesFloatTemp.begin(), res.imagesFloatTemp.end());

    VulkanResource::initImages(*instance.cmd_bufferTransfer, imagesToInit);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = res.imageInput->width,
        .bufferImageHeight = res.imageInput->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
    };
    vk::BufferImageCopy copyColorMapRegion{
        .bufferOffset = 0,
        .bufferRowLength = 256,
        .bufferImageHeight = 1,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{256, 1, 1}
    };
    instance.cmd_bufferTransfer->copyBufferToImage(res.stgInput, res.imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmd_bufferTransfer->copyBufferToImage(res.stgRef, res.imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmd_bufferTransfer->copyBufferToImage(res.stgColormap, res.imageColorMap->image,  vk::ImageLayout::eGeneral, copyColorMapRegion);

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

IQM::Bin::FLIPResult IQM::Bin::flip_copy_back(const VulkanInstance &instance, const FLIPResources &res, Timestamps &timestamps) {
    FLIPResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmd_bufferTransfer->begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = (res.imageOut->width),
        .bufferImageHeight = res.imageOut->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageOut->width, res.imageOut->height, 1}
    };
    instance.cmd_bufferTransfer->copyImageToBuffer(res.imageOut->image,  vk::ImageLayout::eGeneral, res.stgInput, copyRegion);
    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = sizeof(float) * (res.imageOut->width * res.imageOut->height * 4),
        .size = sizeof(float),
    };
    instance.cmd_bufferTransfer->copyBuffer(res.meanBuf, res.stgInput, bufCopy);

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

    std::vector<float> outputData(res.imageOut->height * res.imageOut->width * 4);
    void * outBufData = res.stgInputMemory.mapMemory(0, ((res.imageOut->height * res.imageOut->width * 4) + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, res.imageOut->height * res.imageOut->width * 4 * sizeof(float));
    result.meanFlip = (static_cast<float*>(outBufData))[res.imageOut->height * res.imageOut->width] / (static_cast<float>(res.imageOut->width) * static_cast<float>(res.imageOut->height));

    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    result.imageData = std::move(outputData);

    return result;
}
