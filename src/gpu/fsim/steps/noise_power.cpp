/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/noise_power.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_pack_for_median.inc>
;

static std::vector<uint32_t> srcSort =
#include <lib/multi_radixsort.inc>
;

static std::vector<uint32_t> srcSortHistogram =
#include <lib/multi_radixsort_histograms.inc>
;

static std::vector<uint32_t> srcOut =
#include <fsim/fsim_noise_power.inc>
;

IQM::GPU::FSIMNoisePower::FSIMNoisePower(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smPack = VulkanRuntime::createShaderModule(device, src);
    const auto smSort = VulkanRuntime::createShaderModule(device, srcSort);
    const auto smSortHistogram = VulkanRuntime::createShaderModule(device, srcSortHistogram);
    const auto smNoisePower = VulkanRuntime::createShaderModule(device, srcOut);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->descSetLayoutSort = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector layouts = {
        *this->descSetLayout,
        *this->descSetLayoutSort,
        *this->descSetLayoutSort,
        *this->descSetLayout,
        *this->descSetLayout,
        *this->descSetLayoutSort,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto createdLayouts = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(createdLayouts[0]);
    this->descSetSortEven = std::move(createdLayouts[1]);
    this->descSetSortOdd = std::move(createdLayouts[2]);
    this->descSetSortHistogramEven = std::move(createdLayouts[3]);
    this->descSetSortHistogramOdd = std::move(createdLayouts[4]);
    this->descSetNoisePower = std::move(createdLayouts[5]);

    // 2x uint - buffer size, index
    const auto ranges = VulkanRuntime::createPushConstantRange(2 * sizeof(uint32_t));
    const auto rangesSort = VulkanRuntime::createPushConstantRange(4 * sizeof(uint32_t));

    this->layout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {ranges});
    this->layoutSort = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {rangesSort});
    this->layoutSortHistogram = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {rangesSort});
    this->layoutNoisePower = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {ranges});

    this->pipeline = VulkanRuntime::createComputePipeline(device, smPack, this->layout);
    this->pipelineSort = VulkanRuntime::createComputePipeline(device, smSort, this->layoutSort);
    this->pipelineSortHistogram = VulkanRuntime::createComputePipeline(device, smSortHistogram, this->layoutSortHistogram);
    this->pipelineNoisePower = VulkanRuntime::createComputePipeline(device, smNoisePower, this->layoutNoisePower);
}

void IQM::GPU::FSIMNoisePower::computeNoisePower(const VulkanRuntime &runtime, const vk::raii::Buffer& filterSums, const vk::raii::Buffer& fftBuffer, int width, int height) {
    uint32_t sortGlobalInvocationSize = (width * height) / 32;
    uint32_t remainder = (width * height) % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    auto histBufSize = nSortWorkgroups * 256 * sizeof(uint32_t);
    uint32_t nBlocksPerWorkgroup = 32;

    this->prepareStorage(runtime, fftBuffer, filterSums, width * height, histBufSize);

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    for (int i = 0; i < FSIM_ORIENTATIONS * 2; i++) {
        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, width * height);
        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t),i);

        auto groups = (width * height) / 256 + 1;

        runtime._cmd_buffer->dispatch(groups, 1, 1);

        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );

        for (unsigned j = 0; j < 4; j++) {
            auto activeSet = j % 2 == 0 ? &this->descSetSortHistogramEven : &this->descSetSortHistogramOdd;
            runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSortHistogram);
            runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSortHistogram, 0, {*activeSet}, {});

            std::array values = {
                static_cast<unsigned>(width * height),
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

        runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineNoisePower);
        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutNoisePower, 0, {this->descSetNoisePower}, {});

        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutNoisePower, vk::ShaderStageFlagBits::eCompute, 0, width * height);
        runtime._cmd_buffer->pushConstants<unsigned>(this->layoutNoisePower, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t),i);

        runtime._cmd_buffer->dispatch(1, 1, 1);

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

void IQM::GPU::FSIMNoisePower::prepareStorage(const VulkanRuntime &runtime, const vk::raii::Buffer& fftBuffer, const vk::raii::Buffer& filterSums, unsigned size, unsigned histBufSize) {
    auto [buf, mem] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        2 * FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    buf.bindMemory(mem, 0);
    this->noisePowers = std::move(buf);
    this->noisePowersMemory = std::move(mem);

    auto [bufSort, memSort] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        2 * FSIM_ORIENTATIONS * size * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    auto [bufTemp, memTemp] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        2 * FSIM_ORIENTATIONS * size * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    auto [bufHist, memHist] = VulkanRuntime::createBuffer(
        runtime._device,
        runtime._physicalDevice,
        histBufSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    bufSort.bindMemory(memSort, 0);
    bufTemp.bindMemory(memTemp, 0);
    bufHist.bindMemory(memHist, 0);

    this->noisePowersSortBuf = std::move(bufSort);
    this->noisePowersSortMemory = std::move(memSort);
    this->noisePowersTempBuf = std::move(bufTemp);
    this->noisePowersTempMemory = std::move(memTemp);
    this->noisePowersSortHistogramBuf = std::move(bufHist);
    this->noisePowersSortHistogramMemory = std::move(memHist);

    auto bufInfoIn = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = fftBuffer,
            .offset = 0,
            .range = 2 * size * sizeof(float) * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    auto bufInfoOut = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->noisePowersSortBuf,
            .offset = 0,
            .range = size * sizeof(float) * FSIM_ORIENTATIONS * 2,
        }
    };

    auto bufInfoSortTemp = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->noisePowersTempBuf,
            .offset = 0,
            .range = size * sizeof(float) * FSIM_ORIENTATIONS * 2,
        }
    };

    auto bufInfoSortHist = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->noisePowersSortHistogramBuf,
            .offset = 0,
            .range = histBufSize,
        }
    };

    auto bufInfoFilterSums = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = filterSums,
            .offset = 0,
            .range = FSIM_ORIENTATIONS * sizeof(float),
        }
    };

    auto bufInfoNoisePower = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = this->noisePowers,
            .offset = 0,
            .range = 2 * FSIM_ORIENTATIONS * sizeof(float),
        }
    };

    const auto writes = {
        // copy pass
        VulkanRuntime::createWriteSet(
            this->descSet,
            0,
            bufInfoIn
        ),
        VulkanRuntime::createWriteSet(
            this->descSet,
            1,
            bufInfoOut
        ),
        // sort even
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            0,
            bufInfoOut
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
            bufInfoOut
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
            bufInfoOut
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
        // noise power
        VulkanRuntime::createWriteSet(
            this->descSetNoisePower,
            0,
            bufInfoOut
        ),
        VulkanRuntime::createWriteSet(
            this->descSetNoisePower,
            1,
            bufInfoFilterSums
        ),
        VulkanRuntime::createWriteSet(
            this->descSetNoisePower,
            2,
            bufInfoNoisePower
        ),
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}
