/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define SUMSIZE 1024

layout (local_size_x = SUMSIZE, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InOutBuf {
    float data[];
} buffers[8];

layout( push_constant ) uniform constants {
    // number of elements to sum
    uint size;
    uint index;
} push_consts;

shared float subSums[SUMSIZE];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + tid;

    subSums[tid] = mix(buffers[push_consts.index].data[i], 0.0, i >= push_consts.size);

    memoryBarrierShared();
    barrier();

    for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            subSums[tid] += subSums[tid + s];
        }

        memoryBarrierShared();
        barrier();
    }

    if (tid == 0) {
        buffers[push_consts.index].data[gl_WorkGroupID.x] = subSums[0];
    }
}