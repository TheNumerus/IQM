/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InFFTBuf {
    float inData[];
};

layout( push_constant ) uniform constants {
    uint size;
} push_consts;

shared uint sharedVars[4];
shared float pivot;
shared bool indices[128];
shared bool breakInner;

void swap(uint x, uint y) {
    float value = inData[x];
    float valueOther = inData[y];

    inData[x] = valueOther;
    inData[y] = value;
}

void swapOptional(uint x, uint y) {
    float value = inData[x];
    float valueOther = inData[y];

    inData[x] = min(value, valueOther);
    inData[y] = max(value, valueOther);
}

uint partitionQS(uint left, uint right, uint pivotIndex) {
    /**
    pivotValue := list[pivotIndex]
    swap list[pivotIndex] and list[right]  // Move pivot to end
    storeIndex := left
    for i from left to right − 1 do
        if list[i] < pivotValue then
            swap list[storeIndex] and list[i]
            increment storeIndex
    swap list[right] and list[storeIndex]  // Move pivot to its final place
    return storeIndex
    */
    pivot = inData[pivotIndex];
    if (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x == 0) {
        swap(pivotIndex, right);
    }
    memoryBarrierShared();
    memoryBarrierBuffer();
    barrier();
    uint storeIndex = left;

    uint start = left + gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint end = right - (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
    // clamp
    if (right < (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x)) {
        end = 0;
    }
    // only first half
    if (start < end) {
        swapOptional(start, end);
    }
    memoryBarrierBuffer();
    barrier();

    if (gl_WorkGroupID.x == 0) {
        for(uint i = 0; breakInner; i++) {
            indices[gl_LocalInvocationID.x] = inData[storeIndex + gl_LocalInvocationID.x] > pivot;
            memoryBarrierShared();
            barrier();

            if (gl_LocalInvocationID.x == 0) {
                breakInner = false;
                for (uint y = 0; y < gl_WorkGroupSize.x; y++) {
                    if (indices[y]) {
                        storeIndex = left + i * gl_WorkGroupSize.x + y;
                        breakInner = true;
                        break;
                    }
                }
            }
            memoryBarrierShared();
            barrier();
        }
    }

    memoryBarrierShared();
    memoryBarrierBuffer();
    barrier();

    if (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x == 0) {
        swap(right, storeIndex);
    }
    memoryBarrierBuffer();
    barrier();

    return storeIndex;
}

void main() {
    uint x = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) * 2;
    if (x >= push_consts.size) {
        return;
    }

    uint target = push_consts.size / 2;
    //float pivot = inData[push_consts.size / 2];
    uint pivotIndex = push_consts.size / 2;
    uint left = 0;
    uint right = push_consts.size;

    // horrible quickselect
    for(;;) {
        if (left == right) {
            //return inData[left];
            break;
        }
        pivotIndex = (right - left) / 2 + left;
        pivotIndex = partitionQS(left, right, pivotIndex);
        if (target == pivotIndex) {
            //return inData[target];
            break;
        } else if (target < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }



    // horrible odd-even sort
    /*for (uint i = 0; i < push_consts.size; i++) {
        x = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) * 2;

        if (i % 2 == 0) {
            swap(x);
        } else {
            x += 1;

            swap(x);
        }

        memoryBarrierBuffer();
        barrier();
    }*/
}