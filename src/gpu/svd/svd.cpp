/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/svd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <execution>

static std::vector<uint32_t> srcReduce =
#include <svd/svd_reduce.inc>
;

static std::vector<uint32_t> srcSort =
#include <lib/multi_radixsort.inc>
;

static std::vector<uint32_t> srcSortHistogram =
#include <lib/multi_radixsort_histograms.inc>
;

static std::vector<uint32_t> srcSum =
#include <svd/msvd_sum.inc>
;

IQM::GPU::SVD::SVD(const vk::raii::Device &device) {
    const auto smReduce = VulkanRuntime::createShaderModule(device,  srcReduce);
    const auto smSort = VulkanRuntime::createShaderModule(device, srcSort);
    const auto smSortHistogram = VulkanRuntime::createShaderModule(device, srcSortHistogram);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);

    this->descPool = VulkanRuntime::createDescPool(device, 8, {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 24},
    });

    this->descSetLayoutReduce = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    this->descSetLayoutSort = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    const std::vector layouts = {
        *this->descSetLayoutReduce,
        *this->descSetLayoutSort,
        *this->descSetLayoutSort,
        *this->descSetLayoutReduce,
        *this->descSetLayoutReduce,
        *this->descSetLayoutReduce,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto createdLayouts = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetReduce = std::move(createdLayouts[0]);
    this->descSetSortEven = std::move(createdLayouts[1]);
    this->descSetSortOdd = std::move(createdLayouts[2]);
    this->descSetSortHistogramEven = std::move(createdLayouts[3]);
    this->descSetSortHistogramOdd = std::move(createdLayouts[4]);
    this->descSetSum = std::move(createdLayouts[5]);

    // 1x int - buffer size
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1);
    const auto rangesSum = VulkanRuntime::createPushConstantRange(2 * sizeof(uint32_t));
    const auto rangesSort = VulkanRuntime::createPushConstantRange(4 * sizeof(uint32_t));

    this->layoutReduce = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, ranges);
    this->layoutSort = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {rangesSort});
    this->layoutSortHistogram = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, {rangesSort});
    this->layoutSum = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, {rangesSum});

    this->pipelineReduce = VulkanRuntime::createComputePipeline(device, smReduce, this->layoutReduce);
    this->pipelineSort = VulkanRuntime::createComputePipeline(device, smSort, this->layoutSort);
    this->pipelineSortHistogram = VulkanRuntime::createComputePipeline(device, smSortHistogram, this->layoutSortHistogram);
    this->pipelineSum = VulkanRuntime::createComputePipeline(device, smSum, this->layoutSum);
}

IQM::GPU::SVDResult IQM::GPU::SVD::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    SVDResult res;

    auto bufSize = 2 * 8 * (image.width / 8) * (image.height / 8);
    uint32_t outBufSize = (image.width / 8) * (image.height / 8);

    uint32_t sortGlobalInvocationSize = outBufSize / 32;
    uint32_t remainder = outBufSize % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    auto histBufSize = nSortWorkgroups * 256 * sizeof(uint32_t);
    uint32_t nBlocksPerWorkgroup = 32;

    std::vector<float> data(bufSize);

    this->prepareBuffers(runtime, bufSize * sizeof(float), outBufSize * sizeof(float), histBufSize);

    res.timestamps.mark("buffers prepared");

    cv::Mat inputColor(image.data);
    inputColor = inputColor.reshape(4, image.height);
    cv::Mat refColor(ref.data);
    refColor = refColor.reshape(4, image.height);

    cv::Mat greyInput;
    cv::cvtColor(inputColor, greyInput, cv::COLOR_RGB2GRAY);
    cv::Mat greyRef;
    cv::cvtColor(refColor, greyRef, cv::COLOR_RGB2GRAY);
    cv::Mat inputFloat;
    greyInput.convertTo(inputFloat, CV_32F);
    cv::Mat refFloat;
    greyRef.convertTo(refFloat, CV_32F);

    res.timestamps.mark("image converted");

    // create parallel iterator
    std::vector<int> nums;
    for (int y = 0; (y + 8) < image.height; y+=8) {
        nums.push_back(y);
    }

    std::for_each(std::execution::par, nums.begin(), nums.end(), [&](const int &y) {
        // only process full 8x8 blocks
        for (int x = 0; (x + 8) < image.width; x+=8) {
            cv::Rect crop(x, y, 8, 8);
            cv::Mat srcCrop = inputFloat(crop);
            cv::Mat refCrop = refFloat(crop);

            auto srcSvd = cv::SVD(srcCrop, cv::SVD::NO_UV).w;
            auto refSvd = cv::SVD(refCrop, cv::SVD::NO_UV).w;

            auto start = ((y / 8) * (image.width / 8) + (x / 8)) * 2;

            memcpy(data.data() + (start) * 8, srcSvd.data, 8 * sizeof(float));
            memcpy(data.data() + (start + 1) * 8, refSvd.data, 8 * sizeof(float));
        }
    });

    res.timestamps.mark("end SVD compute");

    void * inBufData = this->stgMemory.mapMemory(0, bufSize * sizeof(float), {});
    memcpy(inBufData, data.data(), bufSize * sizeof(float));
    this->stgMemory.unmapMemory();

    this->copyToGpu(runtime, bufSize * sizeof(float), outBufSize * sizeof(float), histBufSize);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    this->reduceSingularValues(runtime, bufSize);

    auto bufCopy = vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = outBufSize * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(this->outBuffer, this->outSortBuffer, {bufCopy});

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    // need max and median for later steps
    this->sortBlocks(runtime, outBufSize, nSortWorkgroups, nBlocksPerWorkgroup, sortGlobalInvocationSize);

    // after the sorting is done, reuse it's temp buffer to compute M-SVD
    bufCopy = vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = outBufSize * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(this->outBuffer, this->outSortTempBuffer, {bufCopy});

    barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    // now parallel sum
    this->computeMsvd(runtime, outBufSize);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime._device.waitIdle();

    res.timestamps.mark("GPU sum computed");

    this->copyFromGpu(runtime, outBufSize * sizeof(float));

    std::vector<float> outputData(outBufSize);
    void * outBufData = this->stgMemory.mapMemory(0, (outBufSize + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, outBufSize * sizeof(float));

    res.height = image.height / 8;
    res.width = image.width / 8;
    res.msvd = static_cast<float *>(outBufData)[outBufSize] / outBufSize;
    res.imageData = std::move(outputData);

    this->stgMemory.unmapMemory();

    res.timestamps.mark("end GPU writeback");

    return res;
}

void IQM::GPU::SVD::prepareBuffers(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput, size_t histBufInput) {
    assert(sizeInput > sizeOutput);

    // one staging buffer should be enough
    auto [stgBuf, stgMem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        sizeInput,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);
    this->stgBuffer = std::move(stgBuf);
    this->stgMemory = std::move(stgMem);

    auto [buf, mem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        sizeInput,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    buf.bindMemory(mem, 0);
    this->inputBuffer = std::move(buf);
    this->inputMemory = std::move(mem);

    auto [outBuf, outMem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        sizeOutput,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outBuf.bindMemory(outMem, 0);
    this->outBuffer = std::move(outBuf);
    this->outMemory = std::move(outMem);

    auto [outSortBuf, outSortMem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        sizeOutput,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outSortBuf.bindMemory(outSortMem, 0);
    this->outSortBuffer = std::move(outSortBuf);
    this->outSortMemory = std::move(outSortMem);

    auto [outSortTempBuf, outSortTempMem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        sizeOutput,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outSortTempBuf.bindMemory(outSortTempMem, 0);
    this->outSortTempBuffer = std::move(outSortTempBuf);
    this->outSortTempMemory = std::move(outSortTempMem);

    auto [outSortHistBuf, outSortHistMem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        histBufInput,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outSortHistBuf.bindMemory(outSortHistMem, 0);
    this->outSortHistBuffer = std::move(outSortHistBuf);
    this->outSortHistMemory = std::move(outSortHistMem);
}

void IQM::GPU::SVD::reduceSingularValues(const VulkanRuntime &runtime, uint32_t valueCount) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineReduce);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutReduce, 0, {this->descSetReduce}, {});
    runtime._cmd_buffer->pushConstants<int>(this->layoutReduce, vk::ShaderStageFlagBits::eCompute, 0, valueCount);

    // group takes 128 values, reduces to 8 values
    auto groupsX = (valueCount / 128) + 1;
    runtime._cmd_buffer->dispatch(groupsX, 1, 1);

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );
}

void IQM::GPU::SVD::sortBlocks(const VulkanRuntime &runtime, uint32_t nValues, uint32_t nSortWorkgroups, uint32_t nBlocksPerWorkgroup, uint32_t sortGlobalInvocationSize) {
    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    for (unsigned j = 0; j < 4; j++) {
        auto activeSet = j % 2 == 0 ? &this->descSetSortHistogramEven : &this->descSetSortHistogramOdd;
        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSortHistogram);
        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSortHistogram, 0, {*activeSet}, {});

        std::array values = {
            nValues,
            j * 8,
            nSortWorkgroups,
            nBlocksPerWorkgroup,
        };
        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutSortHistogram, vk::ShaderStageFlagBits::eCompute, 0, values);

        runtime._cmd_buffer->dispatch(sortGlobalInvocationSize, 1, 1);

        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );

        activeSet = j % 2 == 0 ? &this->descSetSortEven : &this->descSetSortOdd;
        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSort);
        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSort, 0, {*activeSet}, {});

        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutSort, vk::ShaderStageFlagBits::eCompute, 0, values);

        runtime._cmd_buffer->dispatch(sortGlobalInvocationSize, 1, 1);

        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );
    }
}

void IQM::GPU::SVD::computeMsvd(const VulkanRuntime &runtime, uint32_t nValues) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSum);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSum, 0, {this->descSetSum}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = nValues;
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;
    uint32_t doDiff = 1;

    for (;;) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, 0, size);
        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t), doDiff);
        runtime._cmd_buffer->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = this->outSortTempBuffer,
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
        doDiff = 0;
    }
}

void IQM::GPU::SVD::copyToGpu(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput, size_t histBufInput) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeInput,
    };
    runtime._cmd_buffer->copyBuffer(this->stgBuffer, this->inputBuffer, copyRegion);

    runtime._cmd_buffer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eAllCommands};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();

    std::vector bufInfos = {
        vk::DescriptorBufferInfo {
            .buffer = this->inputBuffer,
            .offset = 0,
            .range = sizeInput,
        },
        vk::DescriptorBufferInfo {
            .buffer = this->outBuffer,
            .offset = 0,
            .range = sizeOutput,
        },
    };

    auto bufInfoSort = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->outSortBuffer,
            .offset = 0,
            .range = sizeOutput,
        }
    };

    auto bufInfoSortTemp = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->outSortTempBuffer,
            .offset = 0,
            .range = sizeOutput,
        }
    };

    auto bufInfoSortHist = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->outSortHistBuffer,
            .offset = 0,
            .range = histBufInput,
        }
    };

    const auto writes = {
        // reduce pass
        VulkanRuntime::createWriteSet(
            this->descSetReduce,
            0,
            bufInfos
        ),
        // sort even
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            0,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            1,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            2,
            bufInfoSortHist
        ),
        // sort odd
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            1,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            2,
            bufInfoSortHist
        ),
        // sort histogram even
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramEven,
            0,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramEven,
            1,
            bufInfoSortHist
        ),
        // sort histogram odd
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramOdd,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramOdd,
            1,
            bufInfoSortHist
        ),
        // sum
        VulkanRuntime::createWriteSet(
            this->descSetSum,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSum,
            1,
            bufInfoSort
        ),
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

void IQM::GPU::SVD::copyFromGpu(const VulkanRuntime &runtime, size_t sizeOutput) {
    runtime._cmd_buffer->reset();
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfoCopy);

    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeOutput,
    };
    runtime._cmd_buffer->copyBuffer(this->outBuffer, this->stgBuffer, copyRegion);

    vk::BufferCopy copyMsvd{
        .srcOffset = 0,
        .dstOffset = sizeOutput,
        .size = sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(this->outSortTempBuffer, this->stgBuffer, copyMsvd);

    runtime._cmd_buffer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfoCopy, *fenceCopy);
    runtime.waitForFence(fenceCopy);
}
