<div align="center">

# :zap: HSpeedTrack

### High-Speed Visual Object Tracker

[![FPS](https://img.shields.io/badge/Speed-694_FPS-brightgreen?style=for-the-badge&logo=nvidia)](.)
[![Platform](https://img.shields.io/badge/Jetson_Orin_Nano-8GB-76B900?style=for-the-badge&logo=nvidia)](.)
[![Resolution](https://img.shields.io/badge/Resolution-1920x1080-blue?style=for-the-badge)](.)
[![Power](https://img.shields.io/badge/TDP-15W-orange?style=for-the-badge&logo=lightning)](.)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus)](.)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-red?style=for-the-badge&logo=nvidia)](.)

*An order of magnitude faster than state-of-the-art trackers -- on an edge device.*

</div>

---

## :trophy: Speed Comparison with State-of-the-Art

### :desktop_computer: Hardware Context

Most trackers report FPS on power-hungry desktop GPUs. HSpeedTrack runs on a **15W edge device**:

| | Platform | TDP | GPU TOPS (FP16) |
|:-:|----------|----:|----------------:|
| :red_circle: | RTX 4070 Ti SUPER | 285 W | ~185 TOPS |
| :red_circle: | RTX 2080 Ti | 250 W | ~107 TOPS |
| :yellow_circle: | Jetson AGX Xavier | 30 W | ~22 TOPS |
| :green_circle: | **Jetson Orin Nano 8GB (Ours)** | **15 W** | **~40 TOPS** |
| :yellow_circle: | Jetson Nano | 10 W | ~0.5 TOPS |

> :bulb: Our device consumes **4.6x less power** than an RTX 4070 Ti and **16.7x less** than an RTX 2080 Ti.

### :bar_chart: FPS Comparison

<table>
<tr><th>Method</th><th>Hardware</th><th>FPS</th><th>Visual</th></tr>
<tr>
  <td>:1st_place_medal: <b>HSpeedTrack (Ours)</b></td>
  <td>:green_circle: Jetson Orin Nano 8GB</td>
  <td><b>694</b></td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-brightgreen?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://github.com/PinataFarms/FEARTracker">FEAR-XS</a> <sub>ECCV'22</sub></td>
  <td>:red_circle: Desktop GPU</td>
  <td>~350</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-yellow?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://arxiv.org/pdf/2305.15896">MixFormerV2-B</a> <sub>NeurIPS'23</sub></td>
  <td>:red_circle: Desktop GPU</td>
  <td>165</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-orange?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://www.mdpi.com/1424-8220/25/23/7359">MemLoTrack</a> <sub>Sensors'25</sub></td>
  <td>:red_circle: RTX 4070 Ti SUPER</td>
  <td>153</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-orange?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0330074&type=printable">ISTD-DETR</a> <sub>PLOS ONE'25</sub></td>
  <td>:red_circle: Desktop GPU</td>
  <td>133</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-orange?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://arxiv.org/html/2506.20381v1">DyHiT</a> <sub>IJCV'25</sub></td>
  <td>:yellow_circle: Jetson AGX Xavier</td>
  <td>111</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-red?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://paperswithcode.com/paper/joint-feature-learning-and-relation-modeling">OSTrack</a> <sub>ECCV'22</sub></td>
  <td>:red_circle: RTX 2080 Ti</td>
  <td>105</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-red?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://www.researchgate.net/publication/375832249">EECF</a> <sub>TPAMI'23</sub></td>
  <td>:red_circle: Desktop GPU</td>
  <td>86</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88-red?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://medium.com/@DeeperAndCheaper/yolov8-jetson-deepstream-benchmark-test-orin-nano-4gb-8gb-nx-tx2-f3993f9c8d2f">YOLOv8n+Tracker</a></td>
  <td>:green_circle: Jetson Orin Nano</td>
  <td>52</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88-critical?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf">SiamRPN++</a> <sub>CVPR'19</sub></td>
  <td>:red_circle: Desktop GPU</td>
  <td>35</td>
  <td><img src="https://img.shields.io/badge/-%E2%96%88-critical?style=flat-square" /></td>
</tr>
<tr>
  <td><a href="https://github.com/HonglinChu/SiamTrackers">LightTrack</a> <sub>CVPR'21</sub></td>
  <td>:yellow_circle: Jetson Nano</td>
  <td>8.3</td>
  <td><img src="https://img.shields.io/badge/-%-critical?style=flat-square" /></td>
</tr>
</table>

### :leaves: FPS-per-Watt Efficiency

The true measure of edge efficiency -- **FPS normalized by power consumption**:

| | Method | FPS | TDP | FPS/W | vs. Ours |
|:-:|--------|----:|----:|------:|---------:|
| :1st_place_medal: | **HSpeedTrack** | **694** | **15W** | **46.3** | -- |
| :2nd_place_medal: | DyHiT | 111 | 30W | 3.7 | 12.5x slower |
| :3rd_place_medal: | YOLOv8n + Tracker | 52 | 15W | 3.5 | 13.2x slower |
| | FEAR-XS | ~350 | ~250W | ~1.4 | 33x slower |
| | LightTrack | 8.3 | 10W | 0.8 | 58x slower |
| | MemLoTrack | 153 | 285W | 0.5 | **92x slower** |
| | OSTrack | 105 | 250W | 0.4 | **116x slower** |

> :fire: **HSpeedTrack is 12.5x more power-efficient than the next best edge tracker (DyHiT)**
> **and 92x more efficient than MemLoTrack running on a 285W desktop GPU.**

---

## :gear: Pipeline Overview

```
Frame 0 (Initialization)
=========================
Full Image (1920x1080)
   |
   v
Resize to 480x480 --> TRT Inference (Frangi Response)
                           |
                           +--> Prefix-Sum on x_max, y_max
                           |        |
                           |        v
                           |    Shift-Subtract --> Target (x, y, w, h)
                           |                           |
                           v                           v
                     Scale to 1920x1080           Crop 480x480 ROI
                                                       |
                                                       v
                                                  Threshold (Flame Mask)
                                                       |
                                                       v
                                              TRT Inference (ROI Response)
                                                       |
                                                       v
                                                  Parallel Top-K (40 pts)
                                                       |
                                                       v
                                             25x25 Patch --> ORB Descriptors


Frame N (Tracking)
===================
Last Target Position
   |
   v
Crop 480x480 ROI from Full Image
   |
   +--> Threshold (Flame Detection + Float Conversion)
   |
   v
TRT Inference ---------------+  (async, GPU)
                              |
Erode Flame Mask -------------+  (parallel, CPU)
                              |
cudaStreamSynchronize --------+
   |
   v
Prefetch response/x_max/y_max to CPU
   |
   +--> Masked Response = Response * Eroded Mask
   |
   +--> Parallel Top-K --> Keypoints + ORB Descriptors
   |
   +--> Descriptor Matching (current vs. last frame)
   |
   +--> Prefix-Sum + Shift-Subtract --> Candidate Box
   |
   v
Post-Processing (Dual Correction Path)
   |
   +--> ORB Mode-Filtered Correction
   |
   +--> Similar-Triangle Geometric Correction
   |
   v
Final Target Position
```

---

## :jigsaw: Key Design Choices

| | Decision | Rationale |
|:-:|----------|-----------|
| :rocket: | TensorRT FP16 inference | Hardware-accelerated Frangi vesselness filter; sub-millisecond latency |
| :abacus: | Prefix-sum + shift-subtract | O(W+H) target localization instead of O(W*H) argmax |
| :thread: | Parallel Top-K (4 threads) | Each thread maintains sorted top-40 over 57,600 elements; merge via `partial_sort` |
| :dna: | ORB binary descriptors | 256-bit Hamming matching; no floating-point distance computation |
| :triangular_ruler: | Similar-triangle correction | Geometric consistency check using 3 matched keypoint pairs |
| :brain: | `cudaMemPrefetchAsync` | Eliminates CUDA Unified Memory page faults on Jetson shared-memory architecture |
| :dart: | `pthread_setaffinity_np` | Pin to core 0; prevents cache invalidation from OS thread migration |
| :fast_forward: | Branchless threshold | SIMD-vectorizable flame mask generation; `#pragma GCC ivdep` |
| :arrows_counterclockwise: | CPU-GPU overlap | `cv::erode` runs on CPU while TRT inference runs on GPU via async stream |

---

## :open_file_folder: Project Structure

```
hspeedtrack_x86/
  |- hspeedtrack.cc             # Production tracker
  |- hspeedtrack_debug.cc       # Debug version with per-stage timing
  |- build.sh                   # Build script (GCC C++20, OpenMP, TRT, CUDA, OpenCV)
  |- types.h                    # Shared type aliases and constants
  |- init_engine.h              # TRT engine loader (deserialize + execution context)
  |
  |- post_process/
  |    |- CtrCorrect.h          # Center-point correction from SmiTri output
  |    |- FilterByBox.h         # Filter keypoints by bounding box proximity
  |    |- FilterKpts.h          # Keypoint filtering by descriptor match quality
  |    |- MatchKptsCorrect.h    # ORB mode-filtered correction
  |    |- SmiTri.h              # Similar-triangle transformation
  |    |- ifSmiTri.h            # Similar-triangle applicability check
  |    |- is_parallel.h         # Parallelism check for triangle edges
  |
  |- utils/
  |    |- parallel_topk.h       # OpenMP parallel Top-K selection (4 threads x 40)
  |    |- descriptor_match.h    # ORB descriptor extraction + Hamming matching
  |    |- get_roi.h             # ROI cropping from full image
  |    |- thresh.h              # Branchless uint8-to-float + flame mask
  |    |- slice.h               # 25x25 patch extraction for descriptors
  |    |- box_size.h            # Per-frame target size lookup
  |    |- bit_pattern_21.h      # ORB BRIEF bit pattern (21x21 sampling)
  |    |- utils.h               # Sorted image loading, CUDA check, multiply, shift-subtract
  |
  |- engine_model/              # TRT engine files (platform-specific)
  |- onnx2trt/                  # ONNX models + trtexec conversion
  |- Datasets/                  # Evaluation datasets
  |- output/                    # Tracking result frames
```

---

## :package: Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| :green_circle: TensorRT | 10+ | Neural network inference (Frangi vesselness filter) |
| :green_circle: CUDA | 12+ | GPU acceleration, unified memory, stream management |
| :blue_circle: OpenCV | 4.x | Image I/O, resize, morphological erosion, video output |
| :orange_circle: OpenMP | 4.5+ | Parallel Top-K, descriptor extraction, SIMD vectorization |
| :purple_circle: GCC | 12+ | C++20 standard, `constexpr`, `std::string_view` |

---

## :hammer_and_wrench: Build

```bash
# Production build
bash build.sh hspeedtrack.cc hspeedtrack

# Debug build (per-stage timing output)
bash build.sh hspeedtrack_debug.cc hspeedtrack_debug
```

---

## :engine: Generate TensorRT Engine

> :warning: TensorRT engine files are **platform-specific** -- regenerate on each target machine.

```bash
cd onnx2trt

trtexec \
  --onnx=./Norm_Grad_Response_Masked_Max_480.onnx \
  --saveEngine=./Norm_Grad_Response_Masked_Max_480.engine \
  --fp16 \
  --builderOptimizationLevel=5 \
  --tilingOptimizationLevel=3 \
  --avgTiming=16 \
  --useCudaGraph \
  --useManagedMemory \
  --exposeDMA \
  --noDataTransfers \
  --timingCacheFile=./timing.cache \
  --separateProfileRun \
  --dumpProfile
```

Then copy the engine:

```bash
cp onnx2trt/Norm_Grad_Response_Masked_Max_480.engine engine_model/
```

<details>
<summary>:mag: Parameter Reference (click to expand)</summary>

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--fp16` | -- | Enable FP16 (half precision) in addition to FP32 |
| `--builderOptimizationLevel` | 5 (max) | Builder spends more time searching for better optimization strategies |
| `--tilingOptimizationLevel` | 3 (max) | Higher tiling optimization search depth |
| `--avgTiming` | 16 | Iterations averaged per kernel timing for tactic selection |
| `--useCudaGraph` | -- | Capture execution into CUDA graph for lower launch overhead |
| `--useManagedMemory` | -- | Use CUDA Unified Memory (matches `cudaMallocManaged` in code) |
| `--timingCacheFile` | `./timing.cache` | Save/load timing cache to speed up subsequent builds |

**Why NOT `--best`?**
`--best` enables all precisions including INT8, which requires a calibrator with representative data.
Without calibration, INT8 tactics are ignored and the builder may select suboptimal strategies,
resulting in larger engine size and slower inference.

</details>

---

## :rocket: Run

```bash
# Place test images in ./test_imgs/ (grayscale, 1920x1080, named img_1.jpg, img_2.jpg, ...)
./hspeedtrack

# Debug mode (prints per-stage timing breakdown)
./hspeedtrack_debug
```

---

## :books: References

- Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF," *ICCV 2011*
- Frangi et al., "Multiscale Vessel Enhancement Filtering," *MICCAI 1998*
- Borsuk et al., "FEAR: Fast, Efficient, Accurate and Robust Visual Tracker," *ECCV 2022*
- Cui et al., "MixFormerV2: Efficient Fully Transformer Tracking," *NeurIPS 2023*
- Ye et al., "OSTrack: Joint Feature Learning and Relation Modeling for Tracking," *ECCV 2022*
- Huang et al., "Anti-UAV410: A Thermal Infrared Benchmark for Tracking Drones in the Wild," *TPAMI 2023*
