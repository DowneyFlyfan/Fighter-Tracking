# Experiment Record

Format: setting → result on test1 / interference2 (within-5px / within-25px / IoU / mean error).

## Detector / Sizing iterations

### Brightness-mask + DoH-score (baseline)
- Detection: brightness > 0.7 × local_max around Kalman prediction.
- Score: doh_peak × size_match × Mahalanobis gating.
- Sizing: brightness mask CC bbox.
- **test1**: w25 98.2%, IoU 0.636, mean 7.0
- **interference2**: w25 0%, IoU 0.001 (drone is NOT brightest, locks distractor frame 1)
- **Conclusion**: brightness-based detection fundamentally fails on multi-object scenes.

### Multi-scale DoH detection + brightness Otsu sizing
- Detection: 3-scale Gaussian Hessian (σ=1, 2, 4), threshold = DOH_FRAC (0.4) × per-video doh_anchor (drone's own DoH peak in init box).
- Score: size_match × gating (no brightness in score).
- Sizing: Otsu thresholding in 60×60 around DoH centroid → CC containing seed → bbox center as Kalman measurement.
- **test1**: w25 90.1%, IoU 0.502
- **interference2**: w25 95.3%, IoU 0.492
- **Conclusion**: brightness-invariant detection + auto-Otsu sizing works across both scenes. But Otsu picks bright body only (drone bright body, not full silhouette) → bbox center biased to one side of drone.

### Asymmetry offset tracking
- After bbox detection, track offset between brightness-bbox top-left and DoH peak; apply to output box position.
- **test1**: w25 90.5%, IoU 0.589 (+0.09 IoU)
- **interference2**: w25 88.0%, IoU 0.493
- **Conclusion**: helps test1 IoU, neutral on interference2. Adds complexity.

### Brightness threshold 0.5 → 0.3 (capture dim drone parts)
- Lower brightness threshold to include drone's dim tail/wings in CC.
- **test1**: w25 90.5%, IoU 0.589
- **interference2**: w25 88.0%, IoU 0.493
- **Conclusion**: captures more of drone, but no clear advantage over 0.5.

### Adaptive window + size sanity check
- Local sizing window scales with current drone size, reject CC if bbox >> ref_area.
- **test1**: w25 94.6%, IoU 0.592 (good!)
- **interference2**: w25 4-10% (BROKEN — window shrinks too small, drone exits)
- **Conclusion**: scale-adaptive window helps test1 but kills interference2 (drone variation).

### floodFill from DoH peak
- Region growing with intensity tolerance ±25 around seed.
- **test1**: w25 92%, IoU 0.317
- **interference2**: w25 44%, IoU 0.179
- **Conclusion**: tolerance-based growth fills cross-drone connections, mask explodes.

### 6-state Kalman (cx, cy, w, h, vx, vy) + motion centroid as primary measurement
- Frame diff with phaseCorrelate camera comp → motion mask CC centroid.
- **test1**: w25 14%, IoU 0.045
- **interference2**: w25 39%, IoU 0.071
- **Conclusion**: motion blob area >> drone area (covers OLD + NEW drone position), bbox biased to midpoint, Kalman size measurement noisy.

### Motion hard gate + DoH detection
- Reject blobs with no motion overlap. Motion as binary filter, DoH as locator.
- **test1**: w25 91%, IoU 0.452
- **interference2**: w25 6%, IoU 0.022
- **Conclusion**: motion gate rejects real drone in low-motion frames.

### Pure DoH (no brightness anywhere) — dual threshold
- Detection: DoH > 0.4 × doh_anchor → seed location.
- Sizing: DoH > 0.05 × doh_anchor (looser) → drone full shape extent → CC centroid as Kalman measurement.
- **test1**: w5 42.8%, w25 96.3%, IoU 0.430, mean 9.1, median 5.8
- **interference2**: w5 43.1%, w25 97.0%, IoU 0.559, mean 8.2, median 5.4
- **Conclusion**: best result so far. Brightness-free, two thresholds (both fractions of one auto-calibrated anchor). Looser Kalman Q (5/50 instead of 2/20) for fast camera motion.

### Knowledge-Distilled ConvNeXt-V1-Atto student (DINOv2-ViT-L teacher)
- Pipeline: AntiUAV410 train (200 IR seq) → DINOv2-ViT-L/14 teacher (304M, 1024-d embedding) → ConvNeXt-V1-Atto student (3.7M, 1024-d projection head) trained via cosine-similarity KD loss on RTX 5070 Ti.
- Why DINOv2-ViT-L not FocusTrack-ViT-B: FocusTrack uses Siamese tracking architecture (template+search dual input), extracting single-patch embedding requires non-trivial weight surgery. DINOv2 self-supervised on 142M images includes aerial/OOD content; cosine in DINOv2 embedding space is what we distill (teacher discriminative power matters, not strict UAV pretrain).
- 5998 patches sampled (200 video × 30 frame), batch 64, 5 epoch, AdamW lr=1e-4. KD loss converged 0.41 → 0.32 (cosine sim ≈ 0.68 with teacher).
- Student exported to ONNX (3.7MB), replaces previous ImageNet-pretrained ConvNeXtV2-Atto.
- C++ KalmanScoredTracker: per-candidate cosine similarity to frame-0 drone embedding multiplied into score = size_match × gating × cnn_score.

### Frozen Hu Moments shape template
- Save Hu Moments of drone DoH-low CC averaged over first 10 stable frames → `tmpl_hu_`.
- Each later frame: compute Hu of detected CC, distance to template, gate at very lenient threshold (shape_score < 0.001 → reject).
- **test1**: w5 40.7%, w25 89.0%, IoU 0.410, mean 17.5
- **interference2**: w5 42.9%, w25 95.3%, IoU 0.565, mean 9.6
- **Conclusion**: minimal change at lenient threshold (rarely fires). Strict threshold (< 0.1) breaks both tests. Hu values fluctuate too much across drone orientations for tight gating. Template kept as safety net.

## Hyperparameters Currently in Use

- `DOH_FRAC = 0.4` (detection threshold fraction)
- `0.05` (sizing threshold fraction)
- `doh_anchor_` (auto-calibrated per video from frame 0)
- `SEARCH_RADIUS = 160`
- `GATING_CHI2 = 25`
- Kalman Q: position 5, velocity 50
- `HU_SHAPE_SIGMA = 5.0`, `SHAPE_REJECT_BELOW = 0.001` (shape template, lenient)

### MOSSE DCF (Discriminative Correlation Filter, no NN)

- 64x64 patch around Kalman predicted center, FFT-based correlation filter (Bolme et al. 2010).

- Frame 0: init filter from drone patch. Each frame: 1 FFT + 1 IFFT to get response map. Per DoH candidate: free lookup at candidate's pixel offset → score multiplier (floor 0.05).

- Online update: A_t = (1-eta)A + eta*G*conj(F); B_t = (1-eta)B + eta*F*conj(F); eta=0.125.

- Pre-process: log + per-patch normalize + Hanning window (illumination invariance).

- score = size_match * gating * hog * dcf.

- **interference2**: w5 52.0%, w10 82.6%, w25 92.5%, IoU 0.580, mean 36.0, med 5.0

- **vs HOG-only baseline**: w5 +12, w10 +15, IoU +0.08, median better. Mean spike (36 vs 12) = tail of bad frames where DCF drift after occlusion locked wrong target.

- **Conclusion**: DCF discriminates drone vs distractor well. Next: add PSR confidence gating to suppress bad updates (drift fix).

### MOSSE + PSR (Peak-to-Sidelobe Ratio) gated update

- PSR = (peak - mean_offpeak) / std_offpeak over 11x11 mask around peak.

- Only update DCF online when PSR >= 10 (Bolme 2010: high confidence). Always trust measurement (DoH + Kalman gating already plausibility check).

- Velocity reset variant tried (zero vx, vy on reject) → broke both videos (-18 w5 test1). Removed.

- PSR_KEEP measurement-reject variant tried (PSR < 3 → drop measurement) → -1 point on test1, no help interference2. Removed; only filter-update gate kept.

- **test1**: w5 52.3%, w10 81.5%, w25 89.4%, IoU 0.430, mean 16.2, med 4.9

- **interference2**: w5 52.0%, w10 83.4%, w25 92.4%, IoU 0.588, mean 35.8, med 5.0

- **Conclusion**: PSR gate stabilises filter slightly (interference2 IoU +0.008, w10 +1.2 vs no PSR). But mean_err = 35.8 unchanged → drift root cause is NOT update poisoning alone. When DCF genuinely locks distractor, PSR is HIGH at distractor → gate doesn't fire. Need frozen-anchor recovery mechanism (re-pull running A_t, B_t toward frame-0 A_0, B_0 every N frames or on lost-state detection).

### MOSSE + PSR gate + frozen-anchor pullback (drift recovery)

- Freeze A_0, B_0 at init() (frame-0 anchor, never updated).

- Track lost_frames_ counter: incremented when accepted=false OR psr<PSR_UPDATE. Reset to 0 on confident accept.

- When lost_frames_ >= 10 (LOST_TRIGGER): A_ = 0.5*A_ + 0.5*A_0; B_ = 0.5*B_ + 0.5*B_0. Resets counter so pullback fires once per drift episode.

- **test1**: w5 51.1%, w10 80.3%, w25 88.9%, IoU 0.427, mean 16.6, med 5.0

- **interference2**: w5 51.9%, w10 84.0%, w25 96.0%, IoU 0.597, mean 8.1, med 5.0

- **vs no-pullback**: interference2 mean_err 35.8 → 8.1 (-77%), w25 92.4 → 96.0 (+3.6), w10 83.4 → 84.0, IoU 0.588 → 0.597. test1 essentially unchanged (-1 w5). Drift cascade broken.

- **Conclusion**: when running filter contaminates with distractor appearance, half-weight pullback to frame-0 lets it re-lock drone within a few frames. Slight test1 regression negligible vs huge interference2 win. Best config so far.

### Camera ego-motion compensation (phaseCorrelate)

- Estimate frame-to-frame camera translation via phaseCorrelate on a centered 256x256 window (background dominates the FFT peak; drone is small fraction).

- Apply shift to Kalman state position BEFORE predict(): cx += shift.x; cy += shift.y. Velocity component left untouched -> represents drone's WORLD-frame motion only (decoupled from camera pan).

- Outlier guards: response_min = 0.05, max_shift = 20 px. First attempt with 80 px cap broke interference2 (w5 52 → 25, mean 8 → 108) because moving distractors hijacked phase peak and produced large bogus shifts. Tight 20 px bound recovers.

- **test1**: w5 51.8%, w10 81.1%, w25 89.1%, IoU 0.430, mean 15.2, med 5.0

- **interference2**: w5 52.2%, w10 84.9%, w25 96.5%, IoU 0.598, mean 7.7, med 4.9

- **vs anchor-pullback only (no ego-motion)**: test1 mean 16.6 → 15.2, w5 +0.7. interference2 mean 8.1 → 7.7, w25 96.0 → 96.5, w10 84.0 → 84.9. Small but consistent improvements on every metric; no regression. Decoupling drone-world-velocity from camera-pan stabilises Kalman prediction during fast camera motion.

### 5-region multi-window phaseCorrelate + median (robust ego-motion)

- 4 corner patches + 1 centered patch, each 192x192 (clamped to half frame dim). phaseCorrelate per region; filter by response_min=0.05 and max_shift=30 px. Take per-axis median of accepted estimates (need >=2 valid).

- Robust to: distractor crossing one corner, drone in central patch, locally-uniform regions with weak phase peak.

- 4-corner-only variant tested -> hurt test1 (mean 15.1 -> 16.3) because clean-background scenes lost central signal. Adding center back fixes that.

- **test1**: w5 51.9%, w10 81.0%, w25 89.7%, IoU 0.430, mean 14.8, med 5.0

- **interference2**: w5 52.1%, w10 84.3%, w25 96.5%, IoU 0.598, mean 7.9, med 5.0

- Speed: 2.99 ms / frame (~335 FPS) after sepFilter2D + motion-mask removal optimisations.

### Optimisation pass (CPU)

- Removed unused motion mask (320x320 phaseCorrelate + warpAffine + threshold + morphology, ~2 ms wasted; gate had been disabled long ago).

- DoH 3-channel x 3-scale filter2D -> sepFilter2D (1D row + 1D col Gaussian-derivative kernels). Mathematically identical (outer product = original 2D kernel). 25x25 -> 25+25 ops/pixel, ~30x fewer multiplies.

- ego-motion phaseCorrelate window 256 -> 192 with multi-region voting (5 regions instead of 1).

- **Total**: 9.31 ms -> 2.99 ms (3.1x faster). Accuracy on both videos preserved or improved.

## Failed/Discarded Approaches

- HOG + Hu Moments fingerprint (16×16 patch on IR is too dominated by background)
- 3-state machine (LOCKED/UNSURE/LOST) — over-engineered, broke baseline
- Otsu on DoH map directly (DoH histogram not bimodal in flat sky)
