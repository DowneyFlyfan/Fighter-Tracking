#pragma once

#include "../types.h"

// Convert uint8 ROI to float and produce a binary flame mask.
// flame_cover_mask[i] = 0 where pixel > 254 (saturated), 1 otherwise.
inline void threshold(const uint8_t *roi, float *flame_cover_mask,
                      float *float_roi, bool &threshold_exceeded) {
    int exceeded = 0;

#pragma GCC ivdep
    for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
        float val = static_cast<float>(roi[i]);
        float_roi[i] = val;
        int is_flame = roi[i] > 254;
        flame_cover_mask[i] = 1.f - static_cast<float>(is_flame);
        exceeded |= is_flame;
    }

    threshold_exceeded = exceeded;
}
