# SOTA Non-NN UAV Tracker Benchmark

This branch (`SOTA`) compares the existing hand-crafted C++ tracker (`./main` on
the `CPU` branch) against re-implemented latest non-NN UAV CF SOTA methods.

## Reality check (2023-2026 non-NN UAV CF)

User asked for "latest 2023-2026 non-NN UAV SOTA". Honest survey:

- **2022 onwards** non-NN UAV tracking is mostly stale; NN dominates Anti-UAV
  challenges (CVPR 2025 Anti-UAV winner = YOLOv12 + BoT-SORT, all NN).
- 2023-2024 non-NN UAV CF papers (SOCF, SSTCF, SAARCF, SRCF) **publish only
  paper, no public code**. Re-implementation is the only option.
- Latest non-NN UAV CF with public code = AutoTrack (CVPR 2020), MSCF (ICRA
  2021), ReCF (TMM 2021) — all Matlab.

## What we built (Phase D only)

User chose to skip OpenCV / pyCFTrackers / AutoTrack baselines and re-implement
**SOCF** (Wang et al., Expert Systems with Applications 2023) from paper.

Original SOCF has three components:

1. STRCF backbone with ADMM solver (5–10 iter)
2. **Object Saliency-Aware**: saliency mask multiplied into the filter via
   the ADMM optimisation
3. **Spatial Disturbance Suppression**: response history → context map →
   disturbance map → penalise negatives in distractor sub-regions

To stay non-NN and FPGA-friendly, the implementation simplifies as:

- **MOSSE closed-form base** instead of STRCF + ADMM (one matrix operation
  per frame, no iterations). Identical math: A_t = G·conj(F), B_t = |F|²,
  H = A/B, response = IFFT(F·H).
- **Adaptive PATCH = SEARCH_FACTOR × max(init_w, init_h)** so the filter's
  search range exceeds inter-frame drone motion (set to 2× empirically).
- **Spatial disturbance suppression** retained as a spatial mask on the
  Gaussian label (`G_masked = G · suppress`) before each update — toggle via
  `USE_DIST = True`. Empirically a small regression in our 3-video benchmark,
  so disabled by default.
- **Saliency mask DROPPED**: the paper applies it inside the ADMM iterations
  to weight the filter. Multiplying spectral-residual saliency into the
  response (cheap approximation) was empirically harmful (kills the drone
  peak when the drone is dim — the exact failure mode SOCF aims to fix).
  Toggleable via `USE_SALIENCY = True` for inspection.

The simplified version is therefore **closer to "pure MOSSE with adaptive
search-region and optional disturbance suppression"** than to a faithful SOCF
reproduction. Don't claim original-paper numbers; this is a useful
non-NN baseline against the C++ DoH+Kalman+CLAHE tracker on our 3 videos.

## Files

```
SOTA_methods/
  socf/
    saliency.py            # spectral residual (Hou & Zhang 2007), 30 lines
    socf_tracker.py        # MOSSE-base SOCF-flavour tracker, ~170 lines
  run_socf.py              # CLI matching main.cc output format
```

`tools/eval_one.py` was extended with `--cmd` so the existing evaluator can
drive any tracker:

```
python tools/eval_one.py Datasets/Anti-UAV-RGBT/test/test1 \
       --cmd "python SOTA_methods/run_socf.py"
```

## Results (3 videos)

|                 | test1                       | interference2               | interference6               |
|                 | w5/w25/IoU/mean             | w5/w25/IoU/mean             | w5/w25/IoU/mean             |
|-----------------|-----------------------------|-----------------------------|-----------------------------|
| Ours C++ (CPU)  | 49.6 / 89.7 / 0.615 / 14.9  | 57.4 / 96.0 / 0.669 / 6.6   | 12.0 / 18.7 / 0.104 / 268.9 |
| **SOCF (simpl)**| 14.2 / **99.9** / **0.629** / 9.8 | 31.5 / **100.0** / **0.682** / 6.8 | 1.4 / 8.0 / 0.019 / 269.2 |

Trade-off vs C++ DoH+Kalman+CLAHE:

- **MOSSE wins coarse**: w25 ≈ 100% (essentially never loses target on
  test1 / inter2). IoU slightly higher than C++.
- **MOSSE loses fine**: w5 only 14–31% vs C++'s 50–57%. Pure correlation-
  filter peak picking is sub-pixel-noisy whereas C++'s DoH centroid is more
  precise.
- **interference6 fails for both**: drone too dim, low-contrast → both DoH
  and MOSSE struggle. Real fix needs richer features (HOG + colour) or
  detection-aware re-init.

## Future work

- Add OpenCV `TrackerCSRT_create` (CSR-DCF, CVPR 2017) baseline via
  opencv-contrib-python.
- Port pyCFTrackers STRCF/DSST/Staple (CVPR 2018 / BMVC 2014 / CVPR 2016).
- Add C++ CAutoTrack (port of CVPR 2020 SOTA UAV CF) as subprocess wrapper.
- Faithful SOCF: implement STRCF + ADMM solver (5-10 iter), proper saliency-
  in-filter formulation. ~4-6 hr extra.
- SSTCF (MDPI Symmetry 2024) re-implementation — similar simplification path.
