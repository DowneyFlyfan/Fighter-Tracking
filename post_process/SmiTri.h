#pragma once

#include "../utils/types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

struct IsParallelsResult {
    double denominator;
    double x, y, A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y;
    std::array<double, 2> pt0, pt1, pt2;
};

struct SmiTriCheck {
    bool apply;
    std::vector<Point> src_points;
    std::vector<Point> dst_points;
};

inline constexpr float SPREAD_THRESH = 1.0f;
inline constexpr float SHIFT_THRESH = 1.0f;

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

// Find the intersection of two lines (AD and BC) across 3 triangle rotations.
// Returns the intersection with the largest absolute denominator (least
// parallel).
inline IsParallelsResult
is_parallels(const std::array<std::array<double, 2>, 3> &pts_last,
             const std::array<double, 2> &center,
             const std::array<std::array<double, 2>, 3> &pts_curr) {
    std::array<double, 3> a1, b1, c1, a2, b2, c2, denom;

    for (int i = 0; i < 3; ++i) {
        const auto &A = pts_last[i];
        const auto &B = pts_last[(i + 1) % 3];
        const auto &C = pts_last[(i + 2) % 3];
        const auto &D = center;

        a1[i] = D[1] - A[1];
        b1[i] = A[0] - D[0];
        c1[i] = D[0] * A[1] - A[0] * D[1];

        a2[i] = C[1] - B[1];
        b2[i] = B[0] - C[0];
        c2[i] = C[0] * B[1] - B[0] * C[1];

        denom[i] = a1[i] * b2[i] - a2[i] * b1[i];
    }

    // Select rotation with largest absolute denominator
    int idx = 0;
    if (std::abs(denom[1]) > std::abs(denom[0]))
        idx = 1;
    if (std::abs(denom[2]) > std::abs(denom[idx]))
        idx = 2;

    double denom_val = denom[idx];
    double x = (b1[idx] * c2[idx] - b2[idx] * c1[idx]) / denom_val;
    double y = (a2[idx] * c1[idx] - a1[idx] * c2[idx]) / denom_val;

    return {denom_val,
            x,
            y,
            pts_last[idx][0],
            pts_last[idx][1],
            pts_last[(idx + 1) % 3][0],
            pts_last[(idx + 1) % 3][1],
            pts_last[(idx + 2) % 3][0],
            pts_last[(idx + 2) % 3][1],
            center[0],
            center[1],
            pts_curr[idx % 3],
            pts_curr[(idx + 1) % 3],
            pts_curr[(idx + 2) % 3]};
}

// Compute transformed center point using similar triangle rule.
// E is the intersection point of AD and BC
inline std::array<double, 2>
SmiTri(const std::array<std::array<double, 2>, 3> &pts_last,
       const std::array<std::array<double, 2>, 3> &pts_current,
       const std::array<double, 2> &center_pt_last) {
    auto [denominator, x, y, A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y, current_A,
          current_B, current_C] =
        is_parallels(pts_last, center_pt_last, pts_current);

    // AD and AE vectors
    double AD_dx = D_x - A_x;
    double AD_dy = D_y - A_y;
    double AE_dx = x - A_x;
    double AE_dy = y - A_y;

    double AD_sq = AD_dx * AD_dx + AD_dy * AD_dy;
    double AE_sq = AE_dx * AE_dx + AE_dy * AE_dy;

    double AD_AE_sign = (AE_dx * AD_dx >= 0 && AE_dy * AD_dy >= 0) ? 1.0 : -1.0;
    double AD_AE_ratio = AE_sq != 0 ? std::sqrt(AD_sq / AE_sq) : 0.0;

    // BC and BE vectors
    double BC_dx = C_x - B_x;
    double BC_dy = C_y - B_y;
    double BE_dx = x - B_x;
    double BE_dy = y - B_y;

    double BC_sq = BC_dx * BC_dx + BC_dy * BC_dy;
    double BE_sq = BE_dx * BE_dx + BE_dy * BE_dy;

    double BE_BC_sign = (BE_dx * BC_dx >= 0 && BE_dy * BC_dy >= 0) ? 1.0 : -1.0;
    double BE_BC_ratio = BC_sq != 0 ? std::sqrt(BE_sq / BC_sq) : 0.0;

    // Map to current frame
    double BC_curr_dx = current_C[0] - current_B[0];
    double BC_curr_dy = current_C[1] - current_B[1];

    double curr_E_x = current_B[0] + BE_BC_sign * BC_curr_dx * BE_BC_ratio;
    double curr_E_y = current_B[1] + BE_BC_sign * BC_curr_dy * BE_BC_ratio;

    double AE_curr_dx = curr_E_x - current_A[0];
    double AE_curr_dy = curr_E_y - current_A[1];

    return {current_A[0] + AD_AE_sign * AE_curr_dx * AD_AE_ratio,
            current_A[1] + AD_AE_sign * AE_curr_dy * AD_AE_ratio};
}
