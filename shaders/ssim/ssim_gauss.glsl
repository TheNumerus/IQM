/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#include "ssim_shared.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform image2D input_img[5];
layout(set = 0, binding = 1, r32f) uniform image2D output_img[5];

layout( push_constant ) uniform constants {
    int kernelSize;
    float sigma;
    int direction;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[z]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    vec2 total = vec2(0.0);
    float totalWeight = 0.0;
    int start = -(push_consts.kernelSize - 1) / 2;
    int end = (push_consts.kernelSize - 1) / 2;

    if (push_consts.direction == 0) {
        // horizontal blur
        for (int xOffset = start; xOffset <= end; xOffset++) {
            int x = pos.x + xOffset;
            int y = pos.y;
            if (x >= size.x || x < 0) {
                continue;
            }
            float weight = gaussWeight(ivec2(xOffset, 0), push_consts.sigma);
            total += imageLoad(input_img[z], ivec2(x, y)).xy * weight;
            totalWeight += weight;
        }
        total /= totalWeight;

        imageStore(output_img[z], pos, vec4(total.x, total.y, 0.0, 0.0));
    } else {
        // vertical blur
        for (int yOffset = start; yOffset <= end; yOffset++) {
            int x = pos.x;
            int y = pos.y + yOffset;
            if (y >= size.y || y < 0) {
                continue;
            }
            float weight = gaussWeight(ivec2(0, yOffset), push_consts.sigma);
            total += imageLoad(output_img[z], ivec2(x, y)).xy * weight;
            totalWeight += weight;
        }
        total /= totalWeight;

        imageStore(input_img[z], pos, vec4(total.x, total.y, 0.0, 0.0));
    }
}
