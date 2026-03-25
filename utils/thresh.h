#pragma once

#include "../types.h"

// Convert uint8 ROI to float and produce a binary flame mask.
// flame_cover_mask[i] = 0 where pixel > 254 (saturated), 1 otherwise.
inline void threshold(
    const uint8_t* roi, float* flame_cover_mask, float* float_roi, bool& threshold_exceeded)
{
    threshold_exceeded = false;

    #pragma omp parallel
    {
        bool local_exceeded = false;

        #pragma omp for simd nowait
        for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
            float_roi[i] = static_cast<float>(roi[i]);
            if (roi[i] > 254) {
                flame_cover_mask[i] = 0.f;
                local_exceeded = true;
            } else {
                flame_cover_mask[i] = 1.f;
            }
        }
        if (local_exceeded) {
            #pragma omp atomic write
            threshold_exceeded = true;
        }
    }
}
