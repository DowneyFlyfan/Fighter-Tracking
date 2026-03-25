#pragma once

#include <array>
#include <cmath>

struct IsParallelsResult {
    double denominator;
    double x, y, A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y;
    std::array<double, 2> pt0, pt1, pt2;
};

// Find the intersection of two lines (AD and BC) across 3 triangle permutations.
// Returns the intersection with the largest absolute denominator (least parallel).
inline IsParallelsResult is_parallels(
    const std::array<double, 3>& A_xs, const std::array<double, 3>& A_ys,
    const std::array<double, 3>& B_xs, const std::array<double, 3>& B_ys,
    const std::array<double, 3>& C_xs, const std::array<double, 3>& C_ys,
    const std::array<double, 3>& D_xs, const std::array<double, 3>& D_ys,
    const std::array<std::array<double, 2>, 3>& pts_curr)
{
    std::array<double, 3> a1, b1, c1, a2, b2, c2, denom;

    for (int i = 0; i < 3; ++i) {
        a1[i] = D_ys[i] - A_ys[i];
        b1[i] = A_xs[i] - D_xs[i];
        c1[i] = D_xs[i] * A_ys[i] - A_xs[i] * D_ys[i];

        a2[i] = C_ys[i] - B_ys[i];
        b2[i] = B_xs[i] - C_xs[i];
        c2[i] = C_xs[i] * B_ys[i] - B_xs[i] * C_ys[i];

        denom[i] = a1[i] * b2[i] - a2[i] * b1[i];
    }

    // Select permutation with largest absolute denominator
    int idx = 0;
    if (std::abs(denom[1]) > std::abs(denom[0])) idx = 1;
    if (std::abs(denom[2]) > std::abs(denom[idx])) idx = 2;

    double denom_val = denom[idx];
    double x = (b1[idx] * c2[idx] - b2[idx] * c1[idx]) / denom_val;
    double y = (a2[idx] * c1[idx] - a1[idx] * c2[idx]) / denom_val;

    return {
        denom_val, x, y,
        A_xs[idx], A_ys[idx],
        B_xs[idx], B_ys[idx],
        C_xs[idx], C_ys[idx],
        D_xs[idx], D_ys[idx],
        pts_curr[idx % 3],
        pts_curr[(idx + 1) % 3],
        pts_curr[(idx + 2) % 3]
    };
}
