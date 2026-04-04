#pragma once

#include <array>
#include <cmath>

struct IsParallelsResult {
    double denominator;
    double x, y, A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y;
    std::array<double, 2> pt0, pt1, pt2;
};

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
