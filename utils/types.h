#pragma once

#include <array>
#include <cstdint>

// Shared type aliases used across the tracking pipeline
using Match = std::array<float, 3>;         // {src_idx, dst_idx, distance}
using Point = std::array<float, 2>;         // {x, y}
using Descriptor = std::array<uint64_t, 4>; // 256-bit packed binary descriptor
using Box = std::array<float, 4>;           // {x, y, w, h}

// Image and ROI (Region of Interest) constants
inline constexpr int ROI_SIZE = 256;
inline constexpr float IMG_WIDTH = 640.0f;
inline constexpr float IMG_HEIGHT = 512.0f;

// Number of keypoints extracted per frame. Drives both the top-K selection
// width in parallel_topk and the array sizes of all per-keypoint pipelines
// (matches, descriptors, patches).
inline constexpr int NUM_KPTS = 40;
inline constexpr int TOPK_THREADS = 4;
inline constexpr int TOPK_TOTAL = NUM_KPTS * TOPK_THREADS;

// Side length of each square patch around a keypoint, used for ORB descriptor
// extraction. PATCH_HALF places the keypoint at the patch center.
inline constexpr int PATCH_SIZE = 25;
inline constexpr int PATCH_HALF = PATCH_SIZE / 2;

// ORB descriptor width in bits (also the number of intensity comparisons).
inline constexpr int DESCRIPTOR_BITS = 256;

// HUD (Heads-Up Display) text overlays burned into Anti-UAV-RGBT videos.
// Three separate rectangles instead of one big top strip, because the
// second-line "方位 X 俯仰 X" text shares a y-range with high-altitude
// drones in some sequences (e.g. test20 has the drone at y=58, x=215);
// keeping the line-2 mask narrow in x preserves such targets.
//
// LABEL_MASK_L1: top-left line 1 ("前端 红外"), y in roughly [0, 33]
// LABEL_MASK_L2: top-left line 2 ("方位 X.XX 俯仰 X.XX"), y in [55, 90]
// LABEL_MASK_R : top-right timestamp ("YYYY-MM-DD HH:MM:SS"), y in [0, 35]
inline constexpr int LABEL_MASK_L1_X = 0;
inline constexpr int LABEL_MASK_L1_Y = 0;
inline constexpr int LABEL_MASK_L1_W = 150;
inline constexpr int LABEL_MASK_L1_H = 38;

// Width 220 covers test1's line-2 text ending at x=217. This clips ~5 px
// off the left edge of test20's drone (GT left edge at x=215); accept that
// loss to remove the text pollution from the Frangi response.
inline constexpr int LABEL_MASK_L2_X = 0;
inline constexpr int LABEL_MASK_L2_Y = 55;
inline constexpr int LABEL_MASK_L2_W = 220;
inline constexpr int LABEL_MASK_L2_H = 40;

inline constexpr int LABEL_MASK_R_W = 260;
inline constexpr int LABEL_MASK_R_H = 38;
