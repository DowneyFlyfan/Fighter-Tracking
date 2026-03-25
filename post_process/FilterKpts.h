#pragma once

#include "../utils/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

inline constexpr int NUM_KPTS = 40;
inline constexpr int TOP_K = 26;
inline constexpr float NORM_SCALE = 255.0f;
inline constexpr float PERCENTILE_70_LOW = 0.5f;
inline constexpr float PERCENTILE_70_HIGH = 0.5f;
inline constexpr int PERCENTILE_IDX_LO = 17;
inline constexpr int PERCENTILE_IDX_HI = 18;
inline constexpr float MODE_RANGE = 2.0f;

struct FilterKptsResult {
    std::vector<Point> src_pts;
    std::vector<Point> dst_pts;
    std::vector<Descriptor> dst_dscrp;
};

struct TopKMatches {
    std::array<Point, TOP_K> src_pts;
    std::array<Point, TOP_K> dst_pts;
    std::array<Descriptor, TOP_K> dst_dscrp;
};

// Sort matches by Hamming distance, return top-K matched keypoints
inline TopKMatches
ExtractTopKMatches(const std::array<Match, NUM_KPTS> &matches,
                   const std::array<Point, NUM_KPTS> &kp_last,
                   const std::array<Point, NUM_KPTS> &kp_curr,
                   const std::array<Descriptor, NUM_KPTS> &dscrp_curr) {
    std::array<int, NUM_KPTS> indices;
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return matches[a][2] < matches[b][2]; });

    TopKMatches top_k;
    for (int i = 0; i < TOP_K; ++i) {
        const auto &m = matches[indices[i]];
        int dst_idx = static_cast<int>(m[1]);
        top_k.src_pts[i] = kp_last[m[0]];
        top_k.dst_pts[i] = kp_curr[dst_idx];
        top_k.dst_dscrp[i] = dscrp_curr[dst_idx];
    }
    return top_k;
}

// Compute displacement norms (Euclidean Distance) between matched keypoint
// pairs
inline std::array<float, TOP_K>
ComputeNorms(const std::array<Point, TOP_K> &src_pts,
             const std::array<Point, TOP_K> &dst_pts, float scale) {
    std::array<float, TOP_K> norms;
    for (int i = 0; i < TOP_K; ++i) {
        float dx = (dst_pts[i][0] - src_pts[i][0]) / scale;
        float dy = (dst_pts[i][1] - src_pts[i][1]) / scale;
        norms[i] = std::sqrt(dx * dx + dy * dy);
    }
    return norms;
}

// Filter keypoint matches by 70th percentile of normalized displacement norm
inline FilterKptsResult
FilterKpts(const std::array<Match, NUM_KPTS> &matches,
           const std::array<Point, NUM_KPTS> &kp_last,
           const std::array<Point, NUM_KPTS> &kp_curr,
           const std::array<Descriptor, NUM_KPTS> &dscrp_curr) {
    auto top_k = ExtractTopKMatches(matches, kp_last, kp_curr, dscrp_curr);
    auto norms = ComputeNorms(top_k.src_pts, top_k.dst_pts, NORM_SCALE);

    // Compute 70th percentile threshold
    std::array<float, TOP_K> sorted_norms = norms;
    std::sort(sorted_norms.begin(), sorted_norms.end());
    float threshold = PERCENTILE_70_LOW * sorted_norms[PERCENTILE_IDX_LO] +
                      PERCENTILE_70_HIGH * sorted_norms[PERCENTILE_IDX_HI];

    // Keep points within the percentile
    FilterKptsResult result;
    for (int i = 0; i < TOP_K; ++i) {
        if (norms[i] <= threshold) {
            result.src_pts.push_back(top_k.src_pts[i]);
            result.dst_pts.push_back(top_k.dst_pts[i]);
            result.dst_dscrp.push_back(top_k.dst_dscrp[i]);
        }
    }
    return result;
}

// Filter keypoint matches using mode-based outlier rejection (within +/-
// MODE_RANGE of mode)
inline FilterKptsResult
FilterKptsMode(const std::array<Match, NUM_KPTS> &matches,
               const std::array<Point, NUM_KPTS> &kp_last,
               const std::array<Point, NUM_KPTS> &kp_curr,
               const std::array<Descriptor, NUM_KPTS> &dscrp_curr) {
    auto top_k = ExtractTopKMatches(matches, kp_last, kp_curr, dscrp_curr);

    // Raw norms (scale = 1.0)
    auto norms = ComputeNorms(top_k.src_pts, top_k.dst_pts, 1.0f);

    // Find mode via frequency count (round to nearest integer to bin floats)
    std::array<int, TOP_K> binned;
    for (int i = 0; i < TOP_K; ++i) {
        binned[i] = static_cast<int>(std::round(norms[i]));
    }

    std::unordered_map<int, int> freq_map;
    for (int val : binned) {
        freq_map[val]++;
    }

    int mode_bin = binned[0];
    int max_count = 0;
    for (const auto &[val, count] : freq_map) {
        if (count > max_count) {
            max_count = count;
            mode_bin = val;
        }
    }
    float mode = static_cast<float>(mode_bin);

    // Keep points within +/- MODE_RANGE of the mode
    FilterKptsResult result;
    for (int i = 0; i < TOP_K; ++i) {
        if (norms[i] >= mode - MODE_RANGE && norms[i] <= mode + MODE_RANGE) {
            result.src_pts.push_back(top_k.src_pts[i]);
            result.dst_pts.push_back(top_k.dst_pts[i]);
            result.dst_dscrp.push_back(top_k.dst_dscrp[i]);
        }
    }
    return result;
}
