/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 8, local_size_y = 8, local_size_z = 16) in;

layout (constant_id = 0) const int KERNEL_SIZE = 5;

const int KERNEL_HALF = KERNEL_SIZE / 2;
const int CACHE_DIM = int(gl_WorkGroupSize.x) + 2*KERNEL_HALF;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float inData[];
};

layout(std430, set = 0, binding = 1) buffer OutBuf {
    float outData[];
};

layout(std430, set = 0, binding = 2) buffer WeightBuf {
    float weights[];
};

layout(std430, set = 0, binding = 3) buffer BiasBuf {
    float biases[];
};

layout( push_constant ) uniform constants {
    uint width;
    uint height;
    uint targetWidth;
    uint targetHeight;
    uint inChannels;
    uint kernelSize;
    uint padding;
    uint stride;
} push_consts;

shared float cache[CACHE_DIM][CACHE_DIM];

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z * gl_WorkGroupSize.z + gl_LocalInvocationID.z;

    float valueTotal = 0.0;

    uint channelSize = push_consts.width * push_consts.height;
    uint targetChannelSize = push_consts.targetWidth * push_consts.targetHeight;

    for (int i = 0; i < push_consts.inChannels; i++) {
        int inChannelOffset = i * int(KERNEL_SIZE * KERNEL_SIZE * gl_NumWorkGroups.z * gl_WorkGroupSize.z);

        if (gl_LocalInvocationID.z == 0) {
            // populate src cache to eliminate redundant accesses
            for (int sy = 0; sy < 2; sy++) {
                for (int sx = 0; sx < 2; sx++) {
                    int piX = int(sx * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
                    int piY = int(sy * gl_WorkGroupSize.y + gl_LocalInvocationID.y);
                    int actX = int((x + sx * gl_WorkGroupSize.x) * push_consts.stride) - int(push_consts.padding);
                    int actY = int((y + sy * gl_WorkGroupSize.y) * push_consts.stride) - int(push_consts.padding);

                    if (piX < CACHE_DIM && piY < CACHE_DIM) {
                        if (0 <= actX && actX < int(push_consts.width) && 0 <= actY && actY < int(push_consts.height)) {
                            float value = inData[channelSize * i + actX + push_consts.width * actY];
                            cache[piX][piY] = value;
                        } else {
                            cache[piX][piY] = 0.0;
                        }
                    }
                }
            }
        }

        memoryBarrierShared();
        barrier();

        for (int ox = 0; ox < KERNEL_SIZE; ox++) {
            for (int oy = 0; oy < KERNEL_SIZE; oy++) {
                int posOffset = (KERNEL_SIZE * (oy) + (ox)) * int(gl_NumWorkGroups.z * gl_WorkGroupSize.z);

                float value = cache[ox + gl_LocalInvocationID.x][oy + gl_LocalInvocationID.y];

                float weight = weights[inChannelOffset + posOffset + z];

                valueTotal += value * weight;
            }
        }

        memoryBarrierShared();
        barrier();
    }

    if (x >= push_consts.targetWidth || y >= push_consts.targetHeight) {
        return;
    }

    float relu = max(0.0, valueTotal + biases[z]);
    outData[targetChannelSize * z + x + push_consts.targetWidth * y] = relu;
}