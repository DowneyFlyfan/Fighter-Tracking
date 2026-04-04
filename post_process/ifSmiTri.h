#pragma once

#include "../types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

inline constexpr float SPREAD_THRESH = 1.0f;
inline constexpr float SHIFT_THRESH = 1.0f;

struct SmiTriCheck {
    bool apply;
    std::vector<Point> src_points;
    std::vector<Point> dst_points;
};

// Determine whether to apply similar triangle (SmiTri) correction
// based on extremal keypoint spread in both source and destination frames.
// Returns up to 4 extremal points (xmax, xmin, ymax, ymin) that pass
// spread and shift criteria.
inline SmiTriCheck CheckSmiTri(const std::vector<Point> &src_pts,
                               const std::vector<Point> &dst_pts) {
    SmiTriCheck result;
    result.apply = false;

    // Find indices of extremal points (max/min x and y)
    int idx_xmax = 0, idx_xmin = 0, idx_ymax = 0, idx_ymin = 0;
    for (int i = 1; i < static_cast<int>(src_pts.size()); ++i) {
        if (src_pts[i][0] > src_pts[idx_xmax][0])
            idx_xmax = i;
        if (src_pts[i][0] < src_pts[idx_xmin][0])
            idx_xmin = i;
        if (src_pts[i][1] > src_pts[idx_ymax][1])
            idx_ymax = i;
        if (src_pts[i][1] < src_pts[idx_ymin][1])
            idx_ymin = i;
    }

    // Gather extremal points from both frames (order: xmax, xmin, ymax, ymin)
    std::array<int, 4> ext_idx = {idx_xmax, idx_xmin, idx_ymax, idx_ymin};
    std::array<Point, 4> src_ext, dst_ext;
    for (int i = 0; i < 4; ++i) {
        src_ext[i] = src_pts[ext_idx[i]];
        dst_ext[i] = dst_pts[ext_idx[i]];
    }

    // Spreads: how wide the extremal points span in each axis
    float src_x_spread = std::abs(src_ext[0][0] - src_ext[1][0]);
    float src_y_spread = std::abs(src_ext[2][1] - src_ext[3][1]);
    float dst_x_spread = std::abs(dst_ext[0][0] - dst_ext[1][0]);
    float dst_y_spread = std::abs(dst_ext[2][1] - dst_ext[3][1]);

    // Shifts: how far each y-extremal point sticks out in y
    // relative to both x-extremal points (take the min)
    float src_ymax_shift =
        std::min(std::abs(src_ext[2][1] - src_ext[0][1]),  // ymax.y vs xmax.y
                 std::abs(src_ext[2][1] - src_ext[1][1])); // ymax.y vs xmin.y
    float src_ymin_shift =
        std::min(std::abs(src_ext[3][1] - src_ext[0][1]),  // ymin.y vs xmax.y
                 std::abs(src_ext[3][1] - src_ext[1][1])); // ymin.y vs xmin.y
    float dst_ymax_shift = std::min(std::abs(dst_ext[2][1] - dst_ext[0][1]),
                                    std::abs(dst_ext[2][1] - dst_ext[1][1]));
    float dst_ymin_shift = std::min(std::abs(dst_ext[3][1] - dst_ext[0][1]),
                                    std::abs(dst_ext[3][1] - dst_ext[1][1]));

    // Shifts: how far each x-extremal point sticks out in x
    // relative to both y-extremal points (take the min)
    float src_xmax_shift =
        std::min(std::abs(src_ext[0][0] - src_ext[2][0]),  // xmax.x vs ymax.x
                 std::abs(src_ext[0][0] - src_ext[3][0])); // xmax.x vs ymin.x
    float src_xmin_shift = std::min(std::abs(src_ext[1][0] - src_ext[2][0]),
                                    std::abs(src_ext[1][0] - src_ext[3][0]));
    float dst_xmax_shift = std::min(std::abs(dst_ext[0][0] - dst_ext[2][0]),
                                    std::abs(dst_ext[0][0] - dst_ext[3][0]));
    float dst_xmin_shift = std::min(std::abs(dst_ext[1][0] - dst_ext[2][0]),
                                    std::abs(dst_ext[1][0] - dst_ext[3][0]));

    // Check if spreads are wide enough in both frames
    bool x_wide =
        src_x_spread >= SPREAD_THRESH && dst_x_spread >= SPREAD_THRESH;
    bool y_wide =
        src_y_spread >= SPREAD_THRESH && dst_y_spread >= SPREAD_THRESH;

    // Check if shifts are significant in both frames
    bool ymax_shifted =
        src_ymax_shift >= SHIFT_THRESH && dst_ymax_shift >= SHIFT_THRESH;
    bool ymin_shifted =
        src_ymin_shift >= SHIFT_THRESH && dst_ymin_shift >= SHIFT_THRESH;
    bool xmax_shifted =
        src_xmax_shift >= SHIFT_THRESH && dst_xmax_shift >= SHIFT_THRESH;
    bool xmin_shifted =
        src_xmin_shift >= SHIFT_THRESH && dst_xmin_shift >= SHIFT_THRESH;

    // Which axis dominates?
    bool x_dominant = src_x_spread >= src_y_spread;

    // Within the secondary axis, which extremal point shifts more?
    bool ymax_more = src_ymax_shift >= src_ymin_shift;
    bool xmax_more = src_xmax_shift >= src_xmin_shift;

    // Decision: x-dominant path needs wide x-spread + any y-shift,
    //           y-dominant path needs wide y-spread + any x-shift
    bool x_path = x_dominant && x_wide;
    bool y_path = !x_dominant && y_wide;
    bool best_y_shifted =
        (ymax_more && ymax_shifted) || (!ymax_more && ymin_shifted);
    bool best_x_shifted =
        (xmax_more && xmax_shifted) || (!xmax_more && xmin_shifted);

    bool apply = (x_path && best_y_shifted) || (y_path && best_x_shifted);
    result.apply = apply;

    // Select which extremal points to output:
    // - Both x-extremals if x-dominant path, or if the specific one is shifted
    // - Both y-extremals if y-dominant path, or if the specific one is shifted
    bool select_xmax =
        (x_path || (y_path && xmax_more && xmax_shifted)) && apply;
    bool select_xmin =
        (x_path || (y_path && !xmax_more && xmin_shifted)) && apply;
    bool select_ymax =
        (y_path || (x_path && ymax_more && ymax_shifted)) && apply;
    bool select_ymin =
        (y_path || (x_path && !ymax_more && ymin_shifted)) && apply;

    bool mask[4] = {select_xmax, select_xmin, select_ymax, select_ymin};
    for (int i = 0; i < 4; ++i) {
        if (mask[i]) {
            result.src_points.push_back(src_ext[i]);
            result.dst_points.push_back(dst_ext[i]);
        }
    }

    return result;
}
