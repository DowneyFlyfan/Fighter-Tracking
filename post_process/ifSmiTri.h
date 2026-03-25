#pragma once

#include "../types.h"

#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <cmath>

inline constexpr float big_diff = 1.0f;
inline constexpr float small_diff = 1.0f;

struct if_SmiTri_result {
    bool choose;
    std::vector<Point> src_points;
    std::vector<Point> dst_points;
};

// Determine whether to apply similar triangle (SmiTri) correction
// based on extremal keypoint spread in both source and destination frames
inline if_SmiTri_result IfSmiTri(const std::vector<Point>& src_pts, const std::vector<Point>& dst_pts)
{
    if_SmiTri_result ifsmiTriresult;
    ifsmiTriresult.choose = false;

    // Find indices of extremal points (max/min x and y)
    int idx_max_x = 0, idx_min_x = 0, idx_max_y = 0, idx_min_y = 0;
    for (size_t i = 1; i < src_pts.size(); ++i) {
        if (src_pts[i][0] > src_pts[idx_max_x][0]) idx_max_x = i;
        if (src_pts[i][0] < src_pts[idx_min_x][0]) idx_min_x = i;
        if (src_pts[i][1] > src_pts[idx_max_y][1]) idx_max_y = i;
        if (src_pts[i][1] < src_pts[idx_min_y][1]) idx_min_y = i;
    }

    // Build extremal point set (order: xmax, xmin, ymax, ymin)
    std::array<int, 4> xxyy_idx = {idx_max_x, idx_min_x, idx_max_y, idx_min_y};
    std::array<Point, 4> src_xxyy_pts, dst_xxyy_pts;
    for (int i = 0; i < 4; ++i) {
        src_xxyy_pts[i] = src_pts[xxyy_idx[i]];
        dst_xxyy_pts[i] = dst_pts[xxyy_idx[i]];
    }

    // Extract extremal coordinates
    float src_max_x = src_xxyy_pts[0][0], src_min_x = src_xxyy_pts[1][0];
    float src_max_y = src_xxyy_pts[2][1], src_min_y = src_xxyy_pts[3][1];
    float dst_max_x = dst_xxyy_pts[0][0], dst_min_x = dst_xxyy_pts[1][0];
    float dst_max_y = dst_xxyy_pts[2][1], dst_min_y = dst_xxyy_pts[3][1];

    float src_xmax_y = src_xxyy_pts[0][1], src_xmin_y = src_xxyy_pts[1][1];
    float src_ymax_x = src_xxyy_pts[2][0], src_ymin_x = src_xxyy_pts[3][0];
    float dst_xmax_y = dst_xxyy_pts[0][1], dst_xmin_y = dst_xxyy_pts[1][1];
    float dst_ymax_x = dst_xxyy_pts[2][0], dst_ymin_x = dst_xxyy_pts[3][0];

    // Build difference arrays (a - b)
    float a[20] = {
        src_max_x, src_max_y, dst_max_x, dst_max_y,
        src_max_y, src_min_y, dst_max_y, dst_min_y,
        src_max_y, src_min_y, dst_max_y, dst_min_y,
        src_max_x, src_min_x, dst_max_x, dst_min_x,
        src_max_x, src_min_x, dst_max_x, dst_min_x
    };
    float b[20] = {
        src_min_x, src_min_y, dst_min_x, dst_min_y,
        src_xmin_y, src_xmin_y, dst_xmin_y, dst_xmin_y,
        src_xmax_y, src_xmax_y, dst_xmax_y, dst_xmax_y,
        src_ymax_x, src_ymax_x, dst_ymax_x, dst_ymax_x,
        src_ymin_x, src_ymin_x, dst_ymin_x, dst_ymin_x
    };

    float diff[20];
    std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(diff),
        [](float x, float y) { return std::abs(x - y); });

    // Compute y-direction and x-direction shifts
    float src_ymax_shift = std::min(diff[4], diff[8]);
    float src_ymin_shift = std::min(diff[5], diff[9]);
    float dst_ymax_shift = std::min(diff[6], diff[10]);
    float dst_ymin_shift = std::min(diff[7], diff[11]);

    float src_xmax_shift = std::min(diff[12], diff[16]);
    float src_xmin_shift = std::min(diff[13], diff[17]);
    float dst_xmax_shift = std::min(diff[14], diff[18]);
    float dst_xmin_shift = std::min(diff[15], diff[19]);

    // Condition checks
    bool x_bigger_4 = diff[0] >= big_diff && diff[2] >= big_diff;
    bool y_bigger_4 = diff[1] >= big_diff && diff[3] >= big_diff;
    bool ymax_x_bigger_2 = src_ymax_shift >= small_diff && dst_ymax_shift >= small_diff;
    bool ymin_x_bigger_2 = src_ymin_shift >= small_diff && dst_ymin_shift >= small_diff;
    bool xmax_y_bigger_2 = src_xmax_shift >= small_diff && dst_xmax_shift >= small_diff;
    bool xmin_y_bigger_2 = src_xmin_shift >= small_diff && dst_xmin_shift >= small_diff;

    bool x_bigger_y = diff[0] >= diff[1];
    bool ymax_shift_bigger = src_ymax_shift >= src_ymin_shift;
    bool xmax_shift_bigger = src_xmax_shift >= src_xmin_shift;

    // Compound conditions (condition_a AND condition_b)
    bool condition_a[6] = {x_bigger_y, ymax_shift_bigger, !ymax_shift_bigger, !x_bigger_y, xmax_shift_bigger, !xmax_shift_bigger};
    bool condition_b[6] = {x_bigger_4, ymax_x_bigger_2, ymin_x_bigger_2, y_bigger_4, xmax_y_bigger_2, xmin_y_bigger_2};

    bool condition_result[6];
    for (int i = 0; i < 6; ++i)
        condition_result[i] = condition_a[i] && condition_b[i];

    // Final selection logic
    bool choose = (condition_result[0] && (condition_result[1] || condition_result[2])) ||
                  (condition_result[3] && (condition_result[4] || condition_result[5]));

    ifsmiTriresult.choose = choose;

    // Build mask: determine which extremal points to output
    bool mask[4] = {
        (condition_result[0] || (condition_result[3] && condition_result[4])) && choose,
        (condition_result[0] || (condition_result[3] && condition_result[5])) && choose,
        (condition_result[3] || (condition_result[0] && condition_result[1])) && choose,
        (condition_result[3] || (condition_result[0] && condition_result[2])) && choose
    };

    // Collect selected points
    for (int i = 0; i < 4; ++i) {
        if (mask[i]) {
            ifsmiTriresult.src_points.push_back(src_xxyy_pts[i]);
            ifsmiTriresult.dst_points.push_back(dst_xxyy_pts[i]);
        }
    }

    return ifsmiTriresult;
}
