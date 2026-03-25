#pragma once

#include <algorithm>
#include <cstring>
#include <tuple>

#include "../types.h"

// Extract ROI region from raw uint8 pointer into pre-allocated contiguous buffer.
inline std::tuple<int, int> GetROI(uint8_t* roi_ptr, const uint8_t* img_ptr,
                                   const std::array<float, 4>& tgt_xywh) {
    int ctr_coords_x =
        static_cast<int>(tgt_xywh[0] + static_cast<int>(tgt_xywh[2]) / 2);
    int ctr_coords_y =
        static_cast<int>(tgt_xywh[1] + static_cast<int>(tgt_xywh[3]) / 2);

    int roi_topleft_x = std::min(std::max(0, ctr_coords_x - ROI_SIZE / 2),
                                 static_cast<int>(IMG_WIDTH) - ROI_SIZE);
    int roi_topleft_y = std::min(std::max(0, ctr_coords_y - ROI_SIZE / 2),
                                 static_cast<int>(IMG_HEIGHT) - ROI_SIZE);

    for (int y = 0; y < ROI_SIZE; ++y) {
        const uint8_t* src_row =
            img_ptr + (roi_topleft_y + y) * static_cast<int>(IMG_WIDTH) +
            roi_topleft_x;
        uint8_t* dst_row = roi_ptr + y * ROI_SIZE;
        std::memcpy(dst_row, src_row, ROI_SIZE);
    }

    return {roi_topleft_x, roi_topleft_y};
}
