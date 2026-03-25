#pragma once

#include <array>
#include <cstring>

#include "types.h"

// Extract 40 patches of 25x25 from a raw float pointer (row-major, ROI_SIZE width)
inline void extract_all_patches(
    std::array<std::array<std::array<float, 25>, 25>, 40>& selected_patch,
    const float* src_image,
    const std::array<std::array<float, 2>, 40>& coords) {

    size_t row_bytes = 25 * sizeof(float);
    for (int p = 0; p < 40; ++p) {
        int center_row = static_cast<int>(coords[p][0]);
        int center_col = static_cast<int>(coords[p][1]);

        // Clamp center so the 25x25 patch stays within [0, ROI_SIZE)
        center_row = std::max(12, std::min(center_row, ROI_SIZE - 13));
        center_col = std::max(12, std::min(center_col, ROI_SIZE - 13));

        for (int r_p = 0; r_p < 25; ++r_p) {
            int src_row = center_row - 12 + r_p;
            int src_col = center_col - 12;

            const float* src_ptr = src_image + src_row * ROI_SIZE + src_col;
            float* dst_ptr = selected_patch[p][r_p].data();
            memcpy(dst_ptr, src_ptr, row_bytes);
        }
    }
}
