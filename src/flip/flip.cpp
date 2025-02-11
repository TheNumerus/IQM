/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <cmath>
#include <IQM/flip.h>

static std::vector<uint32_t> srcInputConvert =
#include <flip/srgb_to_ycxcz.inc>
;

static std::vector<uint32_t> srcFeatureFilterCreate =
#include <flip/feature_filter.inc>
;

static std::vector<uint32_t> srcFeatureFilterNormalize =
#include <flip/feature_filter_normalize.inc>
;

static std::vector<uint32_t> srcFeatureFilterHorizontal =
#include <flip/feature_filter_horizontal.inc>
;

static std::vector<uint32_t> srcFeatureDetect =
#include <flip/feature_detection.inc>
;

static std::vector<uint32_t> srcErrCombine =
#include <flip/combine_error_maps.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FLIP::FLIP(const vk::raii::Device &device):
descPool(VulkanRuntime::createDescPool(device, 64, {
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 128},
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 32}
})),
colorPipeline(device, descPool)
{
    const auto smInputConvert = VulkanRuntime::createShaderModule(device, srcInputConvert);
    const auto smFeatureFilterCreate = VulkanRuntime::createShaderModule(device, srcFeatureFilterCreate);
    const auto smFeatureFilterNormalize = VulkanRuntime::createShaderModule(device, srcFeatureFilterNormalize);
    const auto smFeatureFilterHorizontal = VulkanRuntime::createShaderModule(device, srcFeatureFilterHorizontal);
    const auto smFeatureDetect = VulkanRuntime::createShaderModule(device, srcFeatureDetect);
    const auto smErrorCombine = VulkanRuntime::createShaderModule(device, srcErrCombine);

    this->inputConvertDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
    });

    this->featureFilterCreateDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->featureFilterHorizontalDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->errorCombineDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector allDescLayouts = {
        *this->inputConvertDescSetLayout,
        *this->featureFilterCreateDescSetLayout,
        *this->featureFilterHorizontalDescSetLayout,
        *this->errorCombineDescSetLayout,
        *this->errorCombineDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->inputConvertDescSet = std::move(sets[0]);
    this->featureFilterCreateDescSet = std::move(sets[1]);
    this->featureFilterHorizontalDescSet = std::move(sets[2]);
    this->featureDetectDescSet = std::move(sets[3]);
    this->errorCombineDescSet = std::move(sets[4]);

    this->inputConvertLayout = VulkanRuntime::createPipelineLayout(device, {this->inputConvertDescSetLayout}, {});
    this->inputConvertPipeline = VulkanRuntime::createComputePipeline(device, smInputConvert, this->inputConvertLayout);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float));
    this->featureFilterCreateLayout = VulkanRuntime::createPipelineLayout(device, {this->featureFilterCreateDescSetLayout}, ranges);
    this->featureFilterCreatePipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterCreate, this->featureFilterCreateLayout);
    this->featureFilterNormalizePipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterNormalize, this->featureFilterCreateLayout);

    this->featureFilterHorizontalLayout = VulkanRuntime::createPipelineLayout(device, {this->featureFilterHorizontalDescSetLayout}, {});
    this->featureFilterHorizontalPipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterHorizontal, this->featureFilterHorizontalLayout);

    this->featureDetectLayout = VulkanRuntime::createPipelineLayout(device, {this->errorCombineDescSetLayout}, {});
    this->featureDetectPipeline = VulkanRuntime::createComputePipeline(device, smFeatureDetect, this->featureDetectLayout);

    this->errorCombineLayout = VulkanRuntime::createPipelineLayout(device, {this->errorCombineDescSetLayout}, {});
    this->errorCombinePipeline = VulkanRuntime::createComputePipeline(device, smErrorCombine, this->errorCombineLayout);
}

void IQM::FLIP::computeMetric(const FLIPInput &input) {
    float pixelsPerDegree = FLIP::pixelsPerDegree(input.args);

    this->setUpDescriptors(input);
    this->colorPipeline.setUpDescriptors(input);
    this->convertToYCxCz(input);
    this->createFeatureFilters(input);
    this->computeFeatureErrorMap(input);
    this->colorPipeline.prefilter(input, pixelsPerDegree);
    this->colorPipeline.computeErrorMap(input);
    this->computeFinalErrorMap(input);
}

float IQM::FLIP::pixelsPerDegree(const FLIPArguments &args) {
    return args.monitor_distance * (args.monitor_resolution_x / args.monitor_width) * (std::numbers::pi / 180.0);
}

unsigned IQM::FLIP::spatialKernelSize(const FLIPArguments &args) {
    return 2 * static_cast<int>(std::ceil(3 * std::sqrt(0.04 / (2.0 * std::pow(std::numbers::pi, 2.0))) * pixelsPerDegree(args))) + 1;
}

unsigned IQM::FLIP::featureKernelSize(const FLIPArguments &args) {
    return 2 * static_cast<int>(std::ceil(3 * 0.5 * 0.082 * pixelsPerDegree(args))) + 1;
}

void IQM::FLIP::convertToYCxCz(const FLIPInput& input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->inputConvertPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->inputConvertLayout, 0, {this->inputConvertDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);
}

void IQM::FLIP::createFeatureFilters(const FLIPInput& input) {
    float pixelsPerDegree = FLIP::pixelsPerDegree(input.args);
    int gaussianKernelSize = FLIP::featureKernelSize(input.args);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterCreatePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixelsPerDegree);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(gaussianKernelSize, gaussianKernelSize, 16);

    input.cmdBuf->dispatch(groupsX, 1, 2);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterNormalizePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixelsPerDegree);

    input.cmdBuf->dispatch(groupsX, 1, 2);
}

void IQM::FLIP::computeFeatureErrorMap(const FLIPInput& input) {
    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterHorizontalPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterHorizontalLayout, 0, {this->featureFilterHorizontalDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureDetectPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureDetectLayout, 0, {this->featureDetectDescSet}, {});

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::FLIP::computeFinalErrorMap(const FLIPInput& input) {
    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->errorCombinePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->errorCombineLayout, 0, {this->errorCombineDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::FLIP::setUpDescriptors(const FLIPInput& input) {
    auto imageInfos = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    auto yccOutImageInfos = VulkanRuntime::createImageInfos({
        input.ivTemp[0],
        input.ivTemp[1],
    });

    auto featureFilterImageInfos = VulkanRuntime::createImageInfos({
        input.ivFeatFilter,
    });

    auto tempFeatureFilterImageInfos = VulkanRuntime::createImageInfos({
        input.ivTemp[2],
        input.ivOut,
    });

    auto outFeatureImageInfos = VulkanRuntime::createImageInfos({
        input.ivFeatErr,
    });

    auto colorMapImageInfos = VulkanRuntime::createImageInfos({
        input.ivColorMap
    });

    auto outImageInfos = VulkanRuntime::createImageInfos({
        input.ivOut
    });

    auto errorImageInfos = VulkanRuntime::createImageInfos({
        input.ivFeatErr,
        input.ivColorErr,
    });

    auto writeSetConvertInput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        0,
        imageInfos
    );

    auto writeSetConvertOutput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        1,
        yccOutImageInfos
    );

    auto writeSetFeatureFilter = VulkanRuntime::createWriteSet(
        this->featureFilterCreateDescSet,
        0,
        featureFilterImageInfos
    );

    auto writeSetHorizontalInput = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        0,
        yccOutImageInfos
    );

    auto writeSetHorizontalOutput = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        1,
        tempFeatureFilterImageInfos
    );

    auto writeSetHorizontalFilters = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        2,
        featureFilterImageInfos
    );

    auto writeSetDetectInput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        0,
        tempFeatureFilterImageInfos
    );

    auto writeSetDetectOutput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        1,
        outFeatureImageInfos
    );

    auto writeSetDetectFilters = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        2,
        featureFilterImageInfos
    );

    auto writeSetFinalIn = VulkanRuntime::createWriteSet(
        this->errorCombineDescSet,
        0,
        errorImageInfos
    );

    auto writeSetFinalColorMap = VulkanRuntime::createWriteSet(
        this->errorCombineDescSet,
        1,
        colorMapImageInfos
    );

    auto writeSetFinalOut = VulkanRuntime::createWriteSet(
        this->errorCombineDescSet,
        2,
        outImageInfos
    );

    input.device->updateDescriptorSets({
        writeSetConvertInput, writeSetConvertOutput,
        writeSetFeatureFilter,
        writeSetHorizontalInput, writeSetHorizontalFilters, writeSetHorizontalOutput,
        writeSetDetectInput, writeSetDetectFilters, writeSetDetectOutput,
        writeSetFinalIn, writeSetFinalColorMap, writeSetFinalOut
    }, nullptr);
}
