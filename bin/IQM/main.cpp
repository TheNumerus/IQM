/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include <chrono>

#include <IQM/base/vulkan_runtime.h>
#include <IQM/input_image.h>

#include "args.h"
#include "../shared/debug_utils.h"
#include "../shared/io.h"

#if COMPILE_SSIM
#include <IQM/ssim.h>
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

void ssim(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage& input, const InputImage& reference) {
#ifdef COMPILE_SSIM
    IQM::GPU::SSIM ssim(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = ssim.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "MSSIM: " << result.mssim << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        save_float_image(args.outputPath.value(), result.imageData, result.width, result.height);
    }
#else
    throw std::runtime_error("SSIM support was not compiled");
#endif
}

void svd(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage& input, const InputImage& reference) {
#ifdef COMPILE_SVD
    IQM::GPU::SVD svd(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = svd.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "M-SVD: " << result.msvd << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        save_float_image(args.outputPath.value(), result.imageData, result.width, result.height);
    }
#else
    throw std::runtime_error("SVD support was not compiled");
#endif
}

void fsim(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage& input, const InputImage& reference) {
#ifdef COMPILE_FSIM
    IQM::GPU::FSIM fsim(vulkan._device);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = fsim.computeMetric(vulkan, input, reference);
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

void flip(const IQM::Args& args, const IQM::GPU::VulkanRuntime& vulkan, const InputImage& input, const InputImage& reference) {
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
}

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
    std::optional<IQM::Args> args;

    try {
        args = IQM::Args(argc, argv);
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

    const auto input = load_image(args->inputPath);
    const auto reference = load_image(args->refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    const IQM::GPU::VulkanRuntime vulkan;

    if (args->verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    try {
        switch (args->method) {
            case IQM::Method::SSIM:
                ssim(args.value(), vulkan, input, reference);
                break;
            case IQM::Method::CW_SSIM_CPU:
                throw std::runtime_error("CW-SSIM is not implemented");
            case IQM::Method::SVD:
                svd(args.value(), vulkan, input, reference);
                break;
            case IQM::Method::FSIM:
                fsim(args.value(), vulkan, input, reference);
                break;
            case IQM::Method::FLIP:
                flip(args.value(), vulkan, input, reference);
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
