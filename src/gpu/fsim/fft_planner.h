/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FFTPLANNER_H
#define FFTPLANNER_H
#include <map>
#include <vkFFT.h>

#include "../base/vulkan_runtime.h"

/**
 * Helper class for creation of VkFFT plans
 * Caches already created kernels for faster computation in (future) batch mode
 */
namespace IQM::GPU {
    class FftPlanner {
    public:
        static VkFFTApplication initForward(const VulkanRuntime &runtime, const vk::raii::Fence &fence, const unsigned width, const unsigned height) {
            // image size * 2 float components (complex numbers) * 2 batches
            uint64_t bufferSize = width * height * sizeof(float) * 2 * 2;

            std::pair dims = {width, height};

            VkFFTApplication fftApp = {};

            VkFFTConfiguration fftConfig = {};
            fftConfig.FFTdim = 2;
            fftConfig.size[0] = width;
            fftConfig.size[1] = height;
            fftConfig.bufferSize = &bufferSize;

            VkDevice deviceRef = *runtime._device;
            VkPhysicalDevice physDeviceRef = *runtime._physicalDevice;
            VkQueue queueRef = **runtime._queue;
            VkCommandPool cmdPoolRef = **runtime._commandPool;
            fftConfig.physicalDevice = &physDeviceRef;
            fftConfig.device = &deviceRef;
            fftConfig.queue = &queueRef;
            fftConfig.commandPool = &cmdPoolRef;
            fftConfig.numberBatches = 2;
            fftConfig.makeForwardPlanOnly = true;

            VkFence fenceRef = *fence;
            fftConfig.fence = &fenceRef;

            if (createdForwardKernels.contains(dims)) {
                fftConfig.loadApplicationFromString = true;
                fftConfig.loadApplicationString = createdForwardKernels.at(dims).data();

                if (initializeVkFFT(&fftApp, fftConfig) != VKFFT_SUCCESS) {
                    throw std::runtime_error("failed to initialize FFT");
                }
            } else {
                fftConfig.saveApplicationToString = true;

                if (initializeVkFFT(&fftApp, fftConfig) != VKFFT_SUCCESS) {
                    throw std::runtime_error("failed to initialize FFT");
                }

                std::vector<char> kernel(fftApp.applicationStringSize);
                memcpy(kernel.data(), fftApp.saveApplicationString, fftApp.applicationStringSize);
                createdForwardKernels.emplace(dims, kernel);
            }

            return fftApp;
        }

        static VkFFTApplication initInverse(const VulkanRuntime &runtime, const vk::raii::Fence &fence, const unsigned width, const unsigned height) {
            // (image size * 2 float components (complex numbers) ) * 16 filters * 3 cases (by itself, times input, times reference)
            uint64_t bufferSizeInverse = width * height * sizeof(float) * 2 * 4 * 4 * 3;

            std::pair dims = {width, height};

            VkFFTApplication fftAppInverse = {};

            VkFFTConfiguration fftConfigInverse = {};
            fftConfigInverse.FFTdim = 2;
            fftConfigInverse.size[0] = width;
            fftConfigInverse.size[1] = height;
            fftConfigInverse.bufferSize = &bufferSizeInverse;

            VkDevice deviceRef = *runtime._device;
            VkPhysicalDevice physDeviceRef = *runtime._physicalDevice;
            VkQueue queueRef = **runtime._queue;
            VkCommandPool cmdPoolRef = **runtime._commandPool;
            fftConfigInverse.physicalDevice = &physDeviceRef;
            fftConfigInverse.device = &deviceRef;
            fftConfigInverse.queue = &queueRef;
            fftConfigInverse.commandPool = &cmdPoolRef;
            fftConfigInverse.numberBatches = 16 * 3;
            fftConfigInverse.makeInversePlanOnly = true;
            fftConfigInverse.normalize = true;

            VkFence fenceRef = *fence;
            fftConfigInverse.fence = &fenceRef;

            if (createdInverseKernels.contains(dims)) {
                fftConfigInverse.loadApplicationFromString = true;
                fftConfigInverse.loadApplicationString = createdInverseKernels.at(dims).data();

                if (initializeVkFFT(&fftAppInverse, fftConfigInverse) != VKFFT_SUCCESS) {
                    throw std::runtime_error("failed to initialize FFT");
                }
            } else {
                fftConfigInverse.saveApplicationToString = true;

                if (initializeVkFFT(&fftAppInverse, fftConfigInverse) != VKFFT_SUCCESS) {
                    throw std::runtime_error("failed to initialize FFT");
                }

                std::vector<char> kernel(fftAppInverse.applicationStringSize);
                memcpy(kernel.data(), fftAppInverse.saveApplicationString, fftAppInverse.applicationStringSize);
                createdInverseKernels.emplace(dims, kernel);
            }

            return fftAppInverse;
        }

        static void destroy(VkFFTApplication &fftApp) {
            deleteVkFFT(&fftApp);
        }

    private:
        inline static std::map<std::pair<unsigned, unsigned>, std::vector<char>> createdForwardKernels;
        inline static std::map<std::pair<unsigned, unsigned>, std::vector<char>> createdInverseKernels;
    };
}



#endif //FFTPLANNER_H
