#pragma once

#include <array>
#include <cmath>

#include "is_parallel.h"

// Compute transformed center point using similar triangle rule.
inline std::array<double, 2> SmiTri(
    const std::array<std::array<double, 2>, 3>& pts_last,
    const std::array<std::array<double, 2>, 3>& pts_current,
    const std::array<double, 2>& center_pt_last)
{
    std::array<double, 3> A_xs = { pts_last[0][0], pts_last[1][0], pts_last[2][0] };
    std::array<double, 3> A_ys = { pts_last[0][1], pts_last[1][1], pts_last[2][1] };
    std::array<double, 3> B_xs = { pts_last[1][0], pts_last[2][0], pts_last[0][0] };
    std::array<double, 3> B_ys = { pts_last[1][1], pts_last[2][1], pts_last[0][1] };
    std::array<double, 3> C_xs = { pts_last[2][0], pts_last[0][0], pts_last[1][0] };
    std::array<double, 3> C_ys = { pts_last[2][1], pts_last[0][1], pts_last[1][1] };
    std::array<double, 3> D_xs = { center_pt_last[0], center_pt_last[0], center_pt_last[0] };
    std::array<double, 3> D_ys = { center_pt_last[1], center_pt_last[1], center_pt_last[1] };

    auto [denominator, x, y, A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y,
        current_A, current_B, current_C] =
        is_parallels(A_xs, A_ys, B_xs, B_ys, C_xs, C_ys, D_xs, D_ys, pts_current);

    // AD and A-Dprime vectors
    double AD_dx = D_x - A_x;
    double AD_dy = D_y - A_y;
    double ADp_dx = x - A_x;
    double ADp_dy = y - A_y;

    double AD_sq = AD_dx * AD_dx + AD_dy * AD_dy;
    double ADp_sq = ADp_dx * ADp_dx + ADp_dy * ADp_dy;

    double AD_ADp_sign = (ADp_dx * AD_dx >= 0 && ADp_dy * AD_dy >= 0) ? 1.0 : -1.0;
    double AD_ADp_ratio = ADp_sq != 0 ? std::sqrt(AD_sq / ADp_sq) : 0.0;

    // BC and B-Dprime vectors
    double BC_dx = C_x - B_x;
    double BC_dy = C_y - B_y;
    double BDp_dx = x - B_x;
    double BDp_dy = y - B_y;

    double BC_sq = BC_dx * BC_dx + BC_dy * BC_dy;
    double BDp_sq = BDp_dx * BDp_dx + BDp_dy * BDp_dy;

    double BDp_BC_sign = (BDp_dx * BC_dx >= 0 && BDp_dy * BC_dy >= 0) ? 1.0 : -1.0;
    double BDp_BC_ratio = BC_sq != 0 ? std::sqrt(BDp_sq / BC_sq) : 0.0;

    // Map to current frame
    double BC_curr_dx = current_C[0] - current_B[0];
    double BC_curr_dy = current_C[1] - current_B[1];

    double curr_Dp_x = current_B[0] + BDp_BC_sign * BC_curr_dx * BDp_BC_ratio;
    double curr_Dp_y = current_B[1] + BDp_BC_sign * BC_curr_dy * BDp_BC_ratio;

    double ADp_curr_dx = curr_Dp_x - current_A[0];
    double ADp_curr_dy = curr_Dp_y - current_A[1];

    return {
        current_A[0] + AD_ADp_sign * ADp_curr_dx * AD_ADp_ratio,
        current_A[1] + AD_ADp_sign * ADp_curr_dy * AD_ADp_ratio
    };
}
