#pragma once

#include <array>
#include <cmath>

#include "is_parallel.h"

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
