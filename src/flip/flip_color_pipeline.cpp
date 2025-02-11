/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/flip/color_pipeline.h>
#include <IQM/flip.h>

static std::vector<uint32_t> srcHorizontal =
#include <flip/spatial_prefilter_horizontal.inc>
;

static std::vector<uint32_t> srcPrefilter =
#include <flip/spatial_prefilter.inc>
;

static std::vector<uint32_t> srcDetect =
#include <flip/spatial_detection.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FLIPColorPipeline::FLIPColorPipeline(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smCsfPrefilterHorizontal = VulkanRuntime::createShaderModule(device, srcHorizontal);
    const auto smCsfPrefilter = VulkanRuntime::createShaderModule(device, srcPrefilter);
    const auto smSpatialDetect = VulkanRuntime::createShaderModule(device, srcDetect);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector allDescLayouts = {
        *this->descSetLayout,
        *this->descSetLayout,
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->csfPrefilterHorizontalDescSet = std::move(sets[0]);
    this->csfPrefilterDescSet = std::move(sets[1]);
    this->spatialDetectDescSet = std::move(sets[2]);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float) + sizeof(uint32_t));

    this->csfPrefilterLayout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, ranges);
    this->csfPrefilterHorizontalPipeline = VulkanRuntime::createComputePipeline(device, smCsfPrefilterHorizontal, this->csfPrefilterLayout);
    this->csfPrefilterPipeline = VulkanRuntime::createComputePipeline(device, smCsfPrefilter, this->csfPrefilterLayout);

    this->spatialDetectLayout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {});
    this->spatialDetectPipeline = VulkanRuntime::createComputePipeline(device, smSpatialDetect, this->spatialDetectLayout);
}

void IQM::FLIPColorPipeline::prefilter(const FLIPInput& input, float pixels_per_degree) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterHorizontalPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterHorizontalDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);
    input.cmdBuf->pushConstants<uint32_t>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, sizeof(float), 0u);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);
    input.cmdBuf->pushConstants<uint32_t>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, sizeof(float), 0u);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterHorizontalPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterHorizontalDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);
    input.cmdBuf->pushConstants<uint32_t>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, sizeof(float), 1u);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);
    input.cmdBuf->pushConstants<uint32_t>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, sizeof(float), 1u);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);
}

void IQM::FLIPColorPipeline::computeErrorMap(const FLIPInput& input) {
    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->spatialDetectPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->spatialDetectLayout, 0, {this->spatialDetectDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);
}

void IQM::FLIPColorPipeline::setUpDescriptors(const FLIPInput& input) {
    auto imageInfosPrefilterInput = VulkanRuntime::createImageInfos({
        input.ivTemp[0],
        input.ivTemp[1],
    });

    auto imageInfosPrefilterOutput = VulkanRuntime::createImageInfos({
        input.ivTemp[0],
        input.ivTemp[1],
    });

    auto imageInfosPrefilterTemp = VulkanRuntime::createImageInfos({
        input.ivTemp[2],
    });

    auto imageInfosOutput = VulkanRuntime::createImageInfos({
        input.ivColorErr,
    });

    auto writeSetPrefilterHorInput = VulkanRuntime::createWriteSet(
        this->csfPrefilterHorizontalDescSet,
        0,
        imageInfosPrefilterInput
    );

    auto writeSetPrefilterHorOutput = VulkanRuntime::createWriteSet(
        this->csfPrefilterHorizontalDescSet,
        1,
        imageInfosPrefilterTemp
    );

    auto writeSetPrefilterVertInput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        1,
        imageInfosPrefilterTemp
    );

    auto writeSetPrefilterVertOutput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        0,
        imageInfosPrefilterOutput
    );

    auto writeSetDetectInput = VulkanRuntime::createWriteSet(
        this->spatialDetectDescSet,
        0,
        imageInfosPrefilterOutput
    );

    auto writeSetDetectOutput = VulkanRuntime::createWriteSet(
        this->spatialDetectDescSet,
        1,
        imageInfosOutput
    );

    input.device->updateDescriptorSets({writeSetPrefilterHorInput, writeSetPrefilterHorOutput, writeSetPrefilterVertInput, writeSetPrefilterVertOutput, writeSetDetectInput, writeSetDetectOutput}, nullptr);
}
