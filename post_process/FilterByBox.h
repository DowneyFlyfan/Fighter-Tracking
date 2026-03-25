#pragma once

#include "../types.h"

#include <vector>
#include <array>
#include <limits>

struct FilterByBoxResult {
    std::vector<Point> kp1_boxfiltered;
    std::vector<Point> kp2_boxfiltered;
    std::vector<Descriptor> dscrp2_boxfiltered;
};

// Filter matched keypoints to keep only those within the expected box region.
// The box boundaries are computed as the average of the target box edges
// and the keypoint coordinate extrema.
inline FilterByBoxResult FilterByBox(const std::vector<Point>& kp1, const std::vector<Point>& kp2,
                                const std::vector<Descriptor>& dscrp2, const Box& tgt_xywh_last)
{
    float x = tgt_xywh_last[0];
    float y = tgt_xywh_last[1];
    float w = tgt_xywh_last[2];
    float h = tgt_xywh_last[3];

    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();

    // Find min/max coordinates of kp2
    for (const auto& pt : kp2) {
        if (pt[0] < x_min) x_min = pt[0];
        if (pt[0] > x_max) x_max = pt[0];
        if (pt[1] < y_min) y_min = pt[1];
        if (pt[1] > y_max) y_max = pt[1];
    }

    // Compute filter boundaries (average of box edge and keypoint extrema)
    float x_lower = (x - w * 1 + x_min) / 2.0f;
    float x_upper = (x + w * 2 + x_max) / 2.0f;
    float y_lower = (y - h * 1 + y_min) / 2.0f;
    float y_upper = (y + h * 2 + y_max) / 2.0f;

    FilterByBoxResult filterbyboxresult;

    size_t max_size = kp2.size();
    filterbyboxresult.kp1_boxfiltered.reserve(max_size);
    filterbyboxresult.kp2_boxfiltered.reserve(max_size);
    filterbyboxresult.dscrp2_boxfiltered.reserve(max_size);

    // Keep points within filter boundaries
    for (size_t i = 0; i < kp2.size(); ++i) {
        float px = kp2[i][0];
        float py = kp2[i][1];

        if (px >= x_lower && px <= x_upper && py >= y_lower && py <= y_upper) {
            filterbyboxresult.kp1_boxfiltered.push_back(kp1[i]);
            filterbyboxresult.kp2_boxfiltered.push_back(kp2[i]);
            filterbyboxresult.dscrp2_boxfiltered.push_back(dscrp2[i]);
        }
    }

    return filterbyboxresult;
}
