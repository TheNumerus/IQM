/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "fsim_noise_power.h"

#include <fsim.h>

#include <execution>
#include <algorithm>

IQM::GPU::FSIMNoisePower::FSIMNoisePower(const VulkanRuntime &runtime) {
    auto [buf, mem] = runtime.createBuffer(
        2 * FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    buf.bindMemory(mem, 0);
    this->noisePowers = std::move(buf);
    this->noisePowersMemory = std::move(mem);

    this->kernel = runtime.createShaderModule("../shaders_out/fsim_sort.spv");

    this->descSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
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

    // 1x uint - buffer size
    const auto sumRanges = VulkanRuntime::createPushConstantRange(sizeof(uint));

    this->layout = runtime.createPipelineLayout({this->descSetLayout}, {sumRanges});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

void IQM::GPU::FSIMNoisePower::copyBackToGpu(const VulkanRuntime &runtime, const vk::raii::Buffer& stgBuf) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = 2 * FSIM_ORIENTATIONS * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(stgBuf, this->noisePowers, {region});

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

void IQM::GPU::FSIMNoisePower::computeNoisePower(const VulkanRuntime &runtime, const vk::raii::Buffer& filterSums, const vk::raii::Buffer& fftBuffer, int width, int height) {
    int size = width * height;

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});
    runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, size);

    auto groups = (size / 128) + 1;
    runtime._cmd_buffer->dispatch(groups, 1, 1);

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


    /*auto [stgBuf, stgMem] = runtime.createBuffer(
        2 * FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);
    auto * bufData = static_cast<float*>(stgMem.mapMemory(0,  2 * FSIM_ORIENTATIONS * sizeof(float), {}));

    auto [filterSumsCpuBuf, filterSumsCpuMem] = runtime.createBuffer(
        FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    filterSumsCpuBuf.bindMemory(filterSumsCpuMem, 0);
    this->copyFilterSumsToCpu(runtime, filterSums, filterSumsCpuBuf);

    auto * filterSumsCpu = static_cast<float*>(filterSumsCpuMem.mapMemory(0,  FSIM_ORIENTATIONS * sizeof(float), {}));

    uint64_t largeBufSize = width * height * sizeof(float) * 2;
    auto [stgBufLarge, stgMemLarge] = runtime.createBuffer(
        largeBufSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBufLarge.bindMemory(stgMemLarge, 0);
    auto * bufDataLarge = static_cast<float*>(stgMemLarge.mapMemory(0, largeBufSize, {}));
    std::vector<float> sortBuf(width * height);

    for (int i = 0; i < FSIM_ORIENTATIONS * 2; i++) {
        this->copyFilterToCpu(runtime, fftBuffer, stgBufLarge, width, height, i);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float real = bufDataLarge[(x + width * y) * 2];
                float imag = bufDataLarge[(x + width * y) * 2 + 1];
                sortBuf[x + width * y] = real * real + imag * imag;
            }
        }
        std::sort(std::execution::par_unseq, sortBuf.begin(), sortBuf.end());
        float median = sortBuf[sortBuf.size() / 2];
        float mean = -median / std::log(0.5);

        bufData[i] = mean / filterSumsCpu[i % FSIM_ORIENTATIONS];
    }

    copyBackToGpu(runtime, stgBuf);

    stgMemLarge.unmapMemory();
    stgMem.unmapMemory();*/
}

void IQM::GPU::FSIMNoisePower::copyFilterToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer& fftBuf, const vk::raii::Buffer& target, int width, int height, int index) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    uint64_t filterSize = width * height * sizeof(float) * 2;

    vk::BufferCopy region {
        .srcOffset = filterSize * (FSIM_ORIENTATIONS * FSIM_SCALES + index * FSIM_SCALES),
        .dstOffset = 0,
        .size = filterSize,
    };
    runtime._cmd_buffer->copyBuffer(fftBuf, target, {region});

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

void IQM::GPU::FSIMNoisePower::copyFilterSumsToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer &gpuSrc, const vk::raii::Buffer &cpuTarget) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = FSIM_ORIENTATIONS * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(gpuSrc, cpuTarget, {region});

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
