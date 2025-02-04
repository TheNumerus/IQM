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

#include <GLFW/glfw3.h>

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

void ssim(const IQM::ProfileArgs& args, const IQM::GPU::VulkanRuntime &vulkan, IQM::GPU::SSIM &ssim, const InputImage& input, const InputImage& reference) {
#if COMPILE_SSIM
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
#else
    throw std::runtime_error("SSIM support was not compiled");
#endif
}

void svd(const IQM::ProfileArgs& args, const IQM::GPU::VulkanRuntime &vulkan, IQM::GPU::SVD &svd, const InputImage& input, const InputImage& reference) {
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
}

void fsim(const IQM::ProfileArgs& args, const IQM::GPU::VulkanRuntime &vulkan, IQM::GPU::FSIM &fsim, const InputImage& input, const InputImage& reference) {
#ifdef COMPILE_FSIM
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

void flip(const IQM::ProfileArgs& args, const IQM::GPU::VulkanRuntime &vulkan, IQM::GPU::FLIP &flip, const InputImage& input, const InputImage& reference) {
#ifdef COMPILE_FLIP
    auto flip_args = IQM::GPU::FLIPArguments{};
    if (args.options.contains("FLIP_WIDTH")) {
        flip_args.monitor_width = std::stof(args.options.at("FLIP_WIDTH"));
    }
    if (args.options.contains("FLIP_RES")) {
        flip_args.monitor_resolution_x = std::stof(args.options.at("FLIP_RES"));
    }
    if (args.options.contains("FLIP_DISTANCE")) {
        flip_args.monitor_distance = std::stof(args.options.at("FLIP_DISTANCE"));
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

void ErrorCallback(int, const char* err_str) {
    std::cout << "GLFW Error: " << err_str << std::endl;
}

void printHelp() {
    std::cout << "IQM-profile - Application for profiling methods in IQM.\n"
    << "Usage: IQM-profile --method METHOD --input INPUT --ref REF [--iterations I]\n\n"
    << "Arguments:\n"
    << "    --method <METHOD> : selects method to compute, one of SSIM, SVD, FSIM, FLIP\n"
    << "    --input <INPUT>   : path to tested image\n"
    << "    --ref <REF>       : path to reference image\n"
    << "    --iterations <I>  : number of iterations to compute, unlimited if not set\n"
    << "    -v, --verbose     : enables more detailed output\n"
    << "    -h, --help        : prints help\n\n"
    << "Method specific arguments:\n"
    << "FLIP:\n"
    << "    --flip-width <WIDTH>       : Width of display in meters\n"
    << "    --flip-res <RES>           : Resolution of display in pixels\n"
    << "    --flip-distance <DISTANCE> : Distance to display in meters\n"
    << std::endl;
}

int main(int argc, const char **argv) {
    std::optional<IQM::ProfileArgs> args;

    try {
        args = IQM::ProfileArgs(argc, argv);
    } catch (std::exception& e) {
        std::cout << "Error parsing arguments: " << e.what() << std::endl;
        printHelp();
        return -1;
    }

    if (args->verbose) {
        std::cout << "Selected method: " << IQM::method_name(args->method) << std::endl;
    }

    if (!glfwInit()) {
        return -1;
    }

    glfwSetErrorCallback(ErrorCallback);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "IQM Profile", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    IQM::GPU::VulkanRuntime vulkan;
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(*vulkan._instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    if (args->verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    vulkan.createSwapchain(surface);
    glfwShowWindow(window);

    const auto input = load_image(args->inputPath);
    const auto reference = load_image(args->refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    IQM::GPU::SSIM ssimMethod(vulkan._device);
    IQM::GPU::SVD svdMethod(vulkan._device);
    IQM::GPU::FSIM fsimMethod(vulkan._device);
    IQM::GPU::FLIP flipMethod(vulkan._device);

    unsigned frameIndex = 0;

    while (!glfwWindowShouldClose(window)) {
        try {
            auto index = vulkan.acquire();

            switch (args->method) {
                case IQM::Method::SSIM:
                    ssim(args.value(), vulkan, ssimMethod, input, reference);
                break;
                case IQM::Method::CW_SSIM_CPU:
                    throw std::runtime_error("CW-SSIM is not implemented");
                case IQM::Method::SVD:
                    svd(args.value(), vulkan, svdMethod, input, reference);
                break;
                case IQM::Method::FSIM:
                    fsim(args.value(), vulkan, fsimMethod, input, reference);
                break;
                case IQM::Method::FLIP:
                    flip(args.value(), vulkan, flipMethod, input, reference);
                break;
            }

            vulkan.present(index);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            exit(-1);
        }

        glfwPollEvents();

        if (args->iterations.has_value()) {
            if (args->iterations.value() < frameIndex) {
                break;
            }
        }

        frameIndex += 1;
    }

    vulkan._device.waitIdle();
    vulkan.~VulkanRuntime();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
