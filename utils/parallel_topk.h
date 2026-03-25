#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <vector>

#include "types.h"

// Parallel top-K selection using per-thread sorted arrays, then partial_sort
inline std::array<std::pair<float, int>, TOPK> topk_sorted_parallel(
    const float* data) {
    using Pair = std::pair<float, int>;

    int n = ROI_SIZE * ROI_SIZE;
    int chunk_size = (n + TOPK_THREADS - 1) / TOPK_THREADS;
    std::array<Pair, TOPK_TOTAL> merged;

#pragma omp parallel num_threads(TOPK_THREADS)
    {
        int tid = omp_get_thread_num();
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, n);

        // Stack-allocated sorted array (descending by value)
        std::array<Pair, TOPK> local_topk;
        int count = 0;

        for (int i = start; i < end; ++i) {
            float val = data[i];
            if (count < TOPK) {
                int pos = count;
                while (pos > 0 && val > local_topk[pos - 1].first) {
                    local_topk[pos] = local_topk[pos - 1];
                    pos--;
                }
                local_topk[pos] = {val, i};
                count++;
            } else if (val > local_topk[TOPK - 1].first) {
                int pos = TOPK - 1;
                while (pos > 0 && val > local_topk[pos - 1].first) {
                    local_topk[pos] = local_topk[pos - 1];
                    pos--;
                }
                local_topk[pos] = {val, i};
            }
        }

        int offset = tid * TOPK;
        for (int i = 0; i < count; ++i) merged[offset + i] = local_topk[i];
    }

    // Final merge via partial_sort
    std::partial_sort(
        merged.begin(), merged.begin() + TOPK, merged.end(),
        [](const Pair& a, const Pair& b) { return a.first > b.first; });

    std::array<Pair, TOPK> result;
    std::copy(merged.begin(), merged.begin() + TOPK, result.begin());
    return result;
}
