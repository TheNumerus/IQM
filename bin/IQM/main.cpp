/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "args.h"
#include "file_matcher.h"
#include "vulkan_instance.h"

#if COMPILE_SSIM
#include "wrappers/ssim.h"
#endif

#if COMPILE_SVD
#include <IQM/svd.h>
#endif

#if COMPILE_FSIM
#include <IQM/fsim.h>
#endif

#if COMPILE_FLIP
#include <IQM/flip.h>
#endif
/*
void ssim(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const IQM::InputImage& input, const IQM::InputImage& reference) {
#ifdef COMPILE_SSIM
    IQM::SSIM ssim(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    IQM::Timestamps timestamps;

    auto ssimArgs = IQM::SSIMInput {
        .width = input.width,
        .height = input.height,
        .timestamps = &timestamps,
    };

    auto start = std::chrono::high_resolution_clock::now();
    ssim.computeMetric(ssimArgs);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "MSSIM: " << result.mssim << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        IQM::save_float_image(args.outputPath.value(), result.imageData, input.width, input.height);
    }
#else
    throw std::runtime_error("SSIM support was not compiled");
#endif
}
*/
/*
void svd(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage input, const InputImage reference) {
#ifdef COMPILE_SVD
    IQM::GPU::SVD svd(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = svd.computeMetric(vulkan, &input, &reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "M-SVD: " << result.msvd << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        IQM::save_float_image(args.outputPath.value(), result.imageData, result.width, result.height);
    }
#else
    throw std::runtime_error("SVD support was not compiled");
#endif
}

void fsim(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage input, const InputImage reference) {
#ifdef COMPILE_FSIM
    IQM::GPU::FSIM fsim(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = fsim.computeMetric(vulkan, &input, &reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "FSIM: " << result.fsim << std::endl << "FSIMc: " << result.fsimc << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FSIM support was not compiled");
#endif
}

void flip(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage input, const InputImage reference) {
#ifdef COMPILE_FLIP
    IQM::GPU::FLIP flip(vulkan._device);

    auto flip_args = IQM::GPU::FLIPArguments{};
    if (args.options.contains("--flip-width")) {
        flip_args.monitor_width = std::stof(args.options.at("--flip-width"));
    }
    if (args.options.contains("--flip-res")) {
        flip_args.monitor_resolution_x = std::stof(args.options.at("--flip-res"));
    }
    if (args.options.contains("--flip-distance")) {
        flip_args.monitor_distance = std::stof(args.options.at("--flip-distance"));
    }

    if (args.verbose) {
        std::cout << "FLIP monitor resolution: "<< flip_args.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flip_args.monitor_distance << std::endl
        << "FLIP monitor width: "<< flip_args.monitor_width << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = flip.computeMetric(vulkan, input, reference, flip_args);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FLIP support was not compiled");
#endif
}*/

void printHelp() {
    std::cout << "IQM - Application for computing image quality metrics.\n"
    << "Usage: IQM --method METHOD --input INPUT --ref REF [--output OUTPUT]\n\n"
    << "Arguments:\n"
    << "    --method <METHOD> : selects method to compute, one of SSIM, SVD, FSIM, FLIP\n"
    << "    --input <INPUT>   : path to tested image\n"
    << "    --ref <REF>       : path to reference image\n"
    << "    --output <OUTPUT> : path to output image, optional\n\n"
    << "    -v, --verbose     : enables more detailed output\n"
    << "    -h, --help        : prints help\n\n"
    << "Method specific arguments:\n"
    << "FLIP:\n"
    << "    --flip-width <WIDTH>       : Width of display in meters\n"
    << "    --flip-res <RES>           : Resolution of display in pixels\n"
    << "    --flip-distance <DISTANCE> : Distance to display in meters\n"
    << std::endl;
}

int main(const int argc, const char **argv) {
    std::optional<IQM::Bin::Args> args;

    try {
        args = IQM::Bin::Args(argc, argv);
    } catch (std::exception& e) {
        std::cout << "Error parsing arguments: " << e.what() << std::endl;
        printHelp();
        return -1;
    }

    if (args->printHelp) {
        printHelp();
        return 0;
    }

    if (args->verbose) {
        std::cout << "Selected method: " << IQM::method_name(args->method) << std::endl;
    }

    IQM::Bin::FileMatcher matcher;
    const auto matches = matcher.match(args.value());

    const IQM::Bin::VulkanInstance vulkan;

    if (args->verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    try {
        switch (args->method) {
            case IQM::Method::SSIM:
#ifdef COMPILE_SSIM
                IQM::Bin::ssim_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("SSIM support is not compiled");
#endif
                break;
            case IQM::Method::CW_SSIM_CPU:
                throw std::runtime_error("CW-SSIM is not implemented");
            /*case IQM::Method::SVD:
                svd(args.value(), vulkan, InputImage{.data = input.data, .width = (int)input.width, .height = (int)input.height}, InputImage{.data = reference.data, .width = (int)reference.width, .height = (int)reference.height});
                break;
            case IQM::Method::FSIM:
                fsim(args.value(), vulkan, InputImage{.data = input.data, .width = (int)input.width, .height = (int)input.height}, InputImage{.data = reference.data, .width = (int)reference.width, .height = (int)reference.height});
                break;
            case IQM::Method::FLIP:
                flip(args.value(), vulkan, InputImage{.data = input.data, .width = (int)input.width, .height = (int)input.height}, InputImage{.data = reference.data, .width = (int)reference.width, .height = (int)reference.height});
                break;*/
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
