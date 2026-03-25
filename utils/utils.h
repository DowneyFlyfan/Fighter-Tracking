#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "types.h"

namespace fs = std::filesystem;

// Extract the numeric part from filenames like "img_12.png" -> 12
inline int extract_number(const std::string &filename) {
    std::regex re("img_(\\d+)");
    std::smatch match;
    if (std::regex_search(filename, match, re)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

// Return directory entries sorted by the numeric suffix in filenames
inline std::vector<fs::directory_entry>
get_sorted_image_entries(const std::string &folder_path) {
    std::vector<fs::directory_entry> entries;
    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry &a, const fs::directory_entry &b) {
                  int num_a = extract_number(a.path().filename().string());
                  int num_b = extract_number(b.path().filename().string());
                  return num_a < num_b;
              });
    return entries;
}

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Element-wise multiply response by flame mask
inline void multiply(const float *response_mat, const float *flame_cover_mask,
                     float *orb_response) {
#pragma omp parallel for simd
    for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
        orb_response[i] = response_mat[i] * flame_cover_mask[i];
    }
}

// Shifted subtraction: out[i] = in[i + shift] - in[i], then find argmax
inline std::array<float, 4> shift_subtract(const float *x_sum_matrix,
                                           const float *y_sum_matrix, int w,
                                           int h) {
    int x_out_len = ROI_SIZE - w;
    int y_out_len = ROI_SIZE - h;
    std::array<float, ROI_SIZE> x_out{};
    std::array<float, ROI_SIZE> y_out{};

    for (int i = 0; i < x_out_len; ++i) {
        x_out[i] = x_sum_matrix[i + w] - x_sum_matrix[i];
    }
    for (int j = 0; j < y_out_len; ++j) {
        y_out[j] = y_sum_matrix[j + h] - y_sum_matrix[j];
    }

    auto max_x = std::max_element(x_out.begin(), x_out.begin() + x_out_len);
    auto max_y = std::max_element(y_out.begin(), y_out.begin() + y_out_len);
    auto index_x = std::distance(x_out.begin(), max_x);
    auto index_y = std::distance(y_out.begin(), max_y);

    return {static_cast<float>(index_x), static_cast<float>(index_y),
            static_cast<float>(w), static_cast<float>(h)};
}
