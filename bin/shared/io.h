/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_IO_H
#define IQM_IO_H

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>

inline InputImage load_image(const std::string &filename) {
    // force all images to always open in RGBA format to prevent issues with separate RGB and RGBA loading
    int x, y, channels;
    unsigned char* data = stbi_load(filename.c_str(), &x, &y, &channels, 4);
    if (data == nullptr) {
        const auto err = stbi_failure_reason();
        const auto msg = std::string("Failed to load image '" + filename + "', reason: " + err);
        throw std::runtime_error(msg);
    }

    std::vector<unsigned char> dataVec(x * y * 4);
    memcpy(dataVec.data(), data, x * y * 4 * sizeof(char));

    stbi_image_free(data);

    return InputImage{
        .width = x,
        .height = y,
        .data = std::move(dataVec)
    };
}

inline std::vector<unsigned char> convertFloatToChar(const std::vector<float>& data) {
    std::vector<unsigned char> result(data.size());

    for (int i = 0; i < data.size(); ++i) {
        result[i] = static_cast<unsigned char>(std::clamp(data[i], 0.0f, 1.0f) * 255.0f);
    }

    return result;
}

inline void save_float_image(const std::string &filename, const std::vector<float> &imageData, unsigned int width, unsigned int height) {
    const auto converted = convertFloatToChar(imageData);

    auto saveResult = stbi_write_png(filename.c_str(), width, height, 1, converted.data(), width * sizeof(unsigned char));
    if (saveResult == 0) {
        throw std::runtime_error("Failed to save output image");
    }
}

#endif //IQM_IO_H
