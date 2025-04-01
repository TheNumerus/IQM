/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
};
layout(set = 0, binding = 1, r32f) uniform writeonly image2D outputImg;

layout( push_constant ) uniform constants {
    uint width0;
    uint height0;
    uint width1;
    uint height1;
    uint width2;
    uint height2;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(outputImg);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float value = 0;
    uint offset = 0;

    int nearestX = int((float(x) / float(size.x)) * float(push_consts.width0));
    int nearestY = int((float(y) / float(size.y)) * float(push_consts.height0));

    int nearestX1 = int((float(x) / float(size.x)) * float(push_consts.width1));
    int nearestY1 = int((float(y) / float(size.y)) * float(push_consts.height1));

    int nearestX2 = int((float(x) / float(size.x)) * float(push_consts.width2));
    int nearestY2 = int((float(y) / float(size.y)) * float(push_consts.height2));

    value += data[nearestX + push_consts.width0 * nearestY];
    offset += push_consts.width0 * push_consts.height0;

    value += data[offset + nearestX1 + push_consts.width1 * nearestY1];
    offset += push_consts.width1 * push_consts.height1;

    value += data[offset + nearestX2 + push_consts.width2 * nearestY2];
    offset += push_consts.width2 * push_consts.height2;

    value += data[offset + nearestX2 + push_consts.width2 * nearestY2];
    offset += push_consts.width2 * push_consts.height2;

    value += data[offset + nearestX2 + push_consts.width2 * nearestY2];

    imageStore(outputImg, pos, vec4(value));
}