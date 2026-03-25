#pragma once

#include <array>
#include <cstdint>

// Shared type aliases used across the tracking pipeline
using Match = std::array<float, 3>;         // {src_idx, dst_idx, distance}
using Point = std::array<float, 2>;         // {x, y}
using Descriptor = std::array<uint64_t, 4>; // 256-bit packed binary descriptor
using Box = std::array<float, 4>;           // {x, y, w, h}

// Image and ROI (Region of Interest) constants
inline constexpr int ROI_SIZE = 480;
inline constexpr float IMG_WIDTH = 1920.0f;
inline constexpr float IMG_HEIGHT = 1080.0f;

inline constexpr int TOPK = 40;
inline constexpr int TOPK_THREADS = 4;
inline constexpr int TOPK_TOTAL = TOPK * TOPK_THREADS;
