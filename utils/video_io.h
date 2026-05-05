#pragma once

#include "types.h"

#include <opencv2/opencv.hpp>
#include <string>

// CPU video I/O helpers. Synchronous, no threads. The pipeline_threads.h
// + BoundedQueue design was removed: on CPU, the bottleneck is ORT +
// post-processing, so it's better to give every core to those stages
// in turn (fork-join inside each stage) than to give 1 core each to
// decode + encode for the whole run. Encode/decode are <1 ms here.

// HUD-text masking: tile the row immediately below each mask up through
// the mask's height. Adapts to local background intensity, so the patch
// blends in instead of leaving a black box.
inline void mask_hud_inplace(cv::Mat &m) {
    auto fill_from_below = [&](int x, int y, int w, int h) {
        cv::Mat src_row = m(cv::Rect(x, y + h, w, 1));
        cv::Mat dst = m(cv::Rect(x, y, w, h));
        cv::repeat(src_row, h, 1, dst);
    };
    fill_from_below(LABEL_MASK_L1_X, LABEL_MASK_L1_Y,
                    LABEL_MASK_L1_W, LABEL_MASK_L1_H);
    fill_from_below(LABEL_MASK_L2_X, LABEL_MASK_L2_Y,
                    LABEL_MASK_L2_W, LABEL_MASK_L2_H);
    fill_from_below(m.cols - LABEL_MASK_R_W, 0,
                    LABEL_MASK_R_W, LABEL_MASK_R_H);
}

// Read next frame, force grayscale, apply HUD mask. Returns false on EOF.
inline bool read_masked_frame(cv::VideoCapture &cap, cv::Mat &gray_out) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) return false;
    if (frame.channels() == 1) {
        gray_out = frame;
    } else {
        cv::cvtColor(frame, gray_out, cv::COLOR_BGR2GRAY);
    }
    mask_hud_inplace(gray_out);
    return true;
}
