#pragma once

#include "types.h"

#include <array>
#include <omp.h>

#include "bit_pattern_21.h"

typedef std::array<std::array<std::array<float, 25>, 25>, 40> PatchArray;

// Clamp (x,y) to [0,24] and return the pixel value from the 25x25 patch
inline float sample_pixel(const std::array<std::array<float, 25>, 25> &patch,
                          int x, int y) {
    x = std::max(0, std::min(24, x));
    y = std::max(0, std::min(24, y));
    return patch[x][y];
}

// Extract 256-bit packed binary descriptors from 40 patches using ORB
// (Oriented FAST and Rotated BRIEF) bit pattern
inline std::array<Descriptor, 40>
extract_descriptors(const PatchArray &patches) {
    std::array<Descriptor, 40> descriptors{};
#pragma omp parallel for
    for (int i = 0; i < 40; ++i) {
        descriptors[i] = {};
        for (int j = 0; j < 256; ++j) {
            int idx = j * 4;
            int x1 = 10 + bit_pattern_21_[idx + 0];
            int y1 = 10 + bit_pattern_21_[idx + 1];
            int x2 = 10 + bit_pattern_21_[idx + 2];
            int y2 = 10 + bit_pattern_21_[idx + 3];
            float p1 = sample_pixel(patches[i], x1, y1);
            float p2 = sample_pixel(patches[i], x2, y2);
            if (p1 < p2) {
                // Set bit j: word = j/64, bit = j%64
                descriptors[i][j >> 6] |= (1ULL << (j & 63));
            }
        }
    }
    return descriptors;
}

// Compute Hamming distance between two 256-bit packed binary descriptors
// using __builtin_popcountll (Population Count Long Long)
inline int hamming_distance(const Descriptor &a, const Descriptor &b) {
    return __builtin_popcountll(a[0] ^ b[0]) +
           __builtin_popcountll(a[1] ^ b[1]) +
           __builtin_popcountll(a[2] ^ b[2]) +
           __builtin_popcountll(a[3] ^ b[3]);
}

// Brute-force nearest-neighbor matching using Hamming distance.
// Returns array of {query_idx, best_train_idx, best_distance}.
inline std::array<std::array<float, 3>, 40>
match_descriptors(const std::array<Descriptor, 40> &query,
                  const std::array<Descriptor, 40> &train) {
    std::array<std::array<float, 3>, 40> matches;

#pragma omp parallel for
    for (int i = 0; i < 40; ++i) {
        int best_dist = 256;
        int best_idx = 0;
        for (int j = 0; j < 40; ++j) {
            int dist = hamming_distance(query[i], train[j]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        matches[i] = {static_cast<float>(i), static_cast<float>(best_idx),
                      static_cast<float>(best_dist)};
    }

    return matches;
}
