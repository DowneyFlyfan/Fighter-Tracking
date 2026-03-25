#pragma once

#include "../utils/types.h"

#include <array>
#include <limits>
#include <vector>

struct MatchKptsCorrectResult {
    Box tgt_xywh_curr_orb;
    std::vector<Point> sel_spts_orb;
    std::vector<Point> sel_dpts_orb;
};

// Correct bounding box using ORB (Oriented FAST and Rotated BRIEF) keypoint
// matching. Finds extremal points (xmin, xmax, ymin, ymax) in matched keypoints
// and uses their displacement to update the target box.
inline MatchKptsCorrectResult
MatchKptsCorrect(const std::vector<Point> &kp1_matched,
                 const std::vector<Point> &kp2_matched,
                 const Box &tgt_xywh_last) {
    MatchKptsCorrectResult OrbMatchResult;
    float x = tgt_xywh_last[0];
    float y = tgt_xywh_last[1];
    float w = tgt_xywh_last[2];
    float h = tgt_xywh_last[3];

    // 1. Find extremal point indices in kp1_matched
    int idx_xmin = 0, idx_xmax = 0, idx_ymin = 0, idx_ymax = 0;
    for (int i = 1; i < kp1_matched.size(); ++i) {
        if (kp1_matched[i][0] < kp1_matched[idx_xmin][0])
            idx_xmin = i;
        if (kp1_matched[i][0] > kp1_matched[idx_xmax][0])
            idx_xmax = i;
        if (kp1_matched[i][1] < kp1_matched[idx_ymin][1])
            idx_ymin = i;
        if (kp1_matched[i][1] > kp1_matched[idx_ymax][1])
            idx_ymax = i;
    }

    // 2. Select corresponding points
    std::vector<int> idxs = {idx_xmin, idx_xmax, idx_ymin, idx_ymax};
    for (auto idx : idxs) {
        OrbMatchResult.sel_spts_orb.push_back(kp1_matched[idx]);
        OrbMatchResult.sel_dpts_orb.push_back(kp2_matched[idx]);
    }

    // 3. Compute current boundaries from keypoint displacements
    float current_x_min = OrbMatchResult.sel_dpts_orb[0][0] -
                          OrbMatchResult.sel_spts_orb[0][0] + x;
    float current_y_min = OrbMatchResult.sel_dpts_orb[2][1] -
                          OrbMatchResult.sel_spts_orb[2][1] + y;
    float current_x_max = OrbMatchResult.sel_dpts_orb[1][0] + x + w -
                          OrbMatchResult.sel_spts_orb[1][0];
    float current_y_max = OrbMatchResult.sel_dpts_orb[3][1] + y + h -
                          OrbMatchResult.sel_spts_orb[3][1];
    float current_w = current_x_max - current_x_min;
    float current_h = current_y_max - current_y_min;

    Box tgt_xywh_current = {current_x_min, current_y_min, current_w, current_h};

    // 4. Find extremal indices in destination points
    int dst_idx_xmin = 0, dst_idx_xmax = 0, dst_idx_ymin = 0, dst_idx_ymax = 0;
    for (int i = 1; i < OrbMatchResult.sel_dpts_orb.size(); ++i) {
        if (OrbMatchResult.sel_dpts_orb[i][0] <
            OrbMatchResult.sel_dpts_orb[dst_idx_xmin][0])
            dst_idx_xmin = i;
        if (OrbMatchResult.sel_dpts_orb[i][0] >
            OrbMatchResult.sel_dpts_orb[dst_idx_xmax][0])
            dst_idx_xmax = i;
        if (OrbMatchResult.sel_dpts_orb[i][1] <
            OrbMatchResult.sel_dpts_orb[dst_idx_ymin][1])
            dst_idx_ymin = i;
        if (OrbMatchResult.sel_dpts_orb[i][1] >
            OrbMatchResult.sel_dpts_orb[dst_idx_ymax][1])
            dst_idx_ymax = i;
    }

    // 5. Final correction: expand bbox to encompass destination extrema
    OrbMatchResult.tgt_xywh_curr_orb[0] = std::min(
        tgt_xywh_current[0], OrbMatchResult.sel_dpts_orb[dst_idx_xmin][0]);
    OrbMatchResult.tgt_xywh_curr_orb[1] = std::min(
        tgt_xywh_current[1], OrbMatchResult.sel_dpts_orb[dst_idx_ymin][1]);
    OrbMatchResult.tgt_xywh_curr_orb[2] = std::max(
        tgt_xywh_current[2], OrbMatchResult.sel_dpts_orb[dst_idx_xmax][0] -
                                 OrbMatchResult.sel_dpts_orb[dst_idx_xmin][0]);
    OrbMatchResult.tgt_xywh_curr_orb[3] = std::max(
        tgt_xywh_current[3], OrbMatchResult.sel_dpts_orb[dst_idx_ymax][1] -
                                 OrbMatchResult.sel_dpts_orb[dst_idx_ymin][1]);

    return OrbMatchResult;
}
