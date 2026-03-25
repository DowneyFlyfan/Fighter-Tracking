#pragma once

#include "../types.h"

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>

inline constexpr int N = 40;
inline constexpr int keep = 26;

struct FilterKptsResult {
    std::vector<Point> src_pts;
    std::vector<Point> dst_pts;
    std::vector<Descriptor> dst_dscrp;
};

// Filter keypoint matches by 70th percentile of normalized displacement norm
inline FilterKptsResult FilterKpts(const std::array<Match, 40>& matches,
                const std::array<Point, 40>& kp_last,
                const std::array<Point, 40>& kp_curr,
                const std::array<Descriptor, 40>& dscrp_curr)
{
    // 1. Sort matches by distance (ascending), using index-based sorting
    std::vector<int> indices(40);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return matches[a][2] < matches[b][2];
    });

    // 2. Select top K matches
    std::array<Match, 26> final_matches;
    for (int i = 0; i < 26; ++i) {
        final_matches[i] = matches[indices[i]];
    }

    // 3. Extract keypoint coordinates and descriptors (swap y,x to x,y)
    std::vector<Point> src_pts_matched;
    std::vector<Point> dst_pts_matched;
    std::vector<Descriptor> dst_dscrp;

    for (const auto& m : final_matches) {
        int src_idx = static_cast<int>(m[0]);
        int dst_idx = static_cast<int>(m[1]);
        src_pts_matched.push_back({kp_last[src_idx][1], kp_last[src_idx][0]});
        dst_pts_matched.push_back({kp_curr[dst_idx][1], kp_curr[dst_idx][0]});
        dst_dscrp.push_back(dscrp_curr[dst_idx]);
    }

    // 4. Compute displacement norms (normalized by 255)
    std::vector<float> norms;
    for (size_t i = 0; i < 26; ++i) {
        float dx = (dst_pts_matched[i][0] - src_pts_matched[i][0]) / 255.0f;
        float dy = (dst_pts_matched[i][1] - src_pts_matched[i][1]) / 255.0f;
        float norm = std::sqrt(dx * dx + dy * dy);
        norms.push_back(norm);
    }

    // 5. Compute 70th percentile (26 * 0.7 = 18.2, interpolate between index 17 and 18)
    std::vector<float> sorted_norms = norms;
    std::sort(sorted_norms.begin(), sorted_norms.end());
    float q3 = 0.5f * sorted_norms[17] + 0.5f * sorted_norms[18];

    // 6. Filter out points exceeding the 70th percentile
    FilterKptsResult filterkptsresult;
    for (size_t i = 0; i < norms.size(); ++i) {
        if (norms[i] <= q3) {
            filterkptsresult.src_pts.push_back(src_pts_matched[i]);
            filterkptsresult.dst_pts.push_back(dst_pts_matched[i]);
            filterkptsresult.dst_dscrp.push_back(dst_dscrp[i]);
        }
    }

    return filterkptsresult;
}

// Filter keypoint matches using mode-based outlier rejection (within +/-2 of mode)
inline FilterKptsResult FilterKptsMode(const std::array<Match, 40>& matches,
                const std::array<Point, 40>& kp_last,
                const std::array<Point, 40>& kp_curr,
                const std::array<Descriptor, 40>& dscrp_curr)
{
    // 1. Sort matches by distance (ascending)
    std::vector<int> indices(40);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return matches[a][2] < matches[b][2];
    });

    // 2. Select top K matches
    std::array<Match, 26> final_matches;
    for (int i = 0; i < 26; ++i) {
        final_matches[i] = matches[indices[i]];
    }

    // 3. Extract keypoint coordinates and descriptors (swap y,x to x,y)
    std::vector<Point> src_pts_matched;
    std::vector<Point> dst_pts_matched;
    std::vector<Descriptor> dst_dscrp;

    for (const auto& m : final_matches) {
        int src_idx = static_cast<int>(m[0]);
        int dst_idx = static_cast<int>(m[1]);
        src_pts_matched.push_back({kp_last[src_idx][1], kp_last[src_idx][0]});
        dst_pts_matched.push_back({kp_curr[dst_idx][1], kp_curr[dst_idx][0]});
        dst_dscrp.push_back(dscrp_curr[dst_idx]);
    }

    // 4. Compute displacement norms (raw, not normalized)
    std::vector<float> norms;
    for (size_t i = 0; i < 26; ++i) {
        float dx = (dst_pts_matched[i][0] - src_pts_matched[i][0]);
        float dy = (dst_pts_matched[i][1] - src_pts_matched[i][1]);
        float norm = std::sqrt(dx * dx + dy * dy);
        norms.push_back(norm);
    }

    // 5. Find mode using histogram frequency count
    std::unordered_map<float, int> freqMap;
    for (float val : norms) {
        freqMap[val]++;
    }

    float mode = norms[0];
    int maxCount = 0;
    for (const auto& [val, count] : freqMap) {
        if (count > maxCount) {
            maxCount = count;
            mode = val;
        }
    }

    // 6. Keep points within +/-2.0 of the mode
    FilterKptsResult filterkptsresult;
    for (size_t i = 0; i < 26; ++i) {
        if (norms[i] >= mode - 2.0f && norms[i] <= mode + 2.0f) {
            filterkptsresult.src_pts.push_back(src_pts_matched[i]);
            filterkptsresult.dst_pts.push_back(dst_pts_matched[i]);
            filterkptsresult.dst_dscrp.push_back(dst_dscrp[i]);
        }
    }

    return filterkptsresult;
}
