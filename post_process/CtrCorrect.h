#pragma once

#include "../types.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <array>

// Correct the bounding box center based on the predicted center point,
// Frangi response bounding box, and destination keypoints.
// Returns corrected bbox in {x, y, w, h} format.
inline Box CtrCorrect(const Point& current_ctr_xy,
               const Box& frangi_xyxy,
               const Box& bbox_per5_last,
               const std::vector<Point>& per5_dst_pts)
{
    float left_dis = std::abs(current_ctr_xy[0] - frangi_xyxy[0]);
    float right_dis = std::abs(current_ctr_xy[0] - frangi_xyxy[2]);
    float top_dis = std::abs(current_ctr_xy[1] - frangi_xyxy[1]);
    float bottom_dis = std::abs(current_ctr_xy[1] - frangi_xyxy[3]);

    bool lr_adjust = (left_dis > bbox_per5_last[2]) || (right_dis > bbox_per5_last[2]);
    bool l_bigger_r = (left_dis > right_dis);
    bool tb_adjust = (top_dis > bbox_per5_last[3]) || (bottom_dis > bbox_per5_last[3]);
    bool t_bigger_b = (top_dis > bottom_dis);

    Box adjust_xyxy = {
        current_ctr_xy[0] - right_dis,
        current_ctr_xy[1] - bottom_dis,
        current_ctr_xy[0] + left_dis,
        current_ctr_xy[1] + top_dis
    };

    Box correct_xyxy = frangi_xyxy;

    if (lr_adjust) {
        if (l_bigger_r) correct_xyxy[0] = adjust_xyxy[0];
        else            correct_xyxy[2] = adjust_xyxy[2];
    }
    if (tb_adjust) {
        if (t_bigger_b) correct_xyxy[1] = adjust_xyxy[1];
        else            correct_xyxy[3] = adjust_xyxy[3];
    }

    // Find min/max coordinates of destination keypoints
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& pt : per5_dst_pts) {
        if (pt[0] < min_x) min_x = pt[0];
        if (pt[0] > max_x) max_x = pt[0];
        if (pt[1] < min_y) min_y = pt[1];
        if (pt[1] > max_y) max_y = pt[1];
    }

    // Expand bbox to encompass all keypoints
    float corrected_x1 = std::min(correct_xyxy[0], min_x);
    float corrected_y1 = std::min(correct_xyxy[1], min_y);
    float corrected_x2 = std::max(correct_xyxy[2], max_x);
    float corrected_y2 = std::max(correct_xyxy[3], max_y);

    // Convert from xyxy to xywh format
    Box corrected_bbox = {
        corrected_x1,
        corrected_y1,
        corrected_x2 - corrected_x1,
        corrected_y2 - corrected_y1
    };

    return corrected_bbox;
}
