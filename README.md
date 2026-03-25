# hspeedtrack_x86

## Generate TensorRT Engine (ONNX to TRT)

TensorRT engine files are **platform-specific** — they must be regenerated on each target machine.

### Optimal trtexec Command (FP16, Max Optimization)

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

Then copy the engine to the expected path:

```bash
cp onnx2trt/Norm_Grad_Response_Masked_Max_480.engine engine_model/
```

### Parameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--fp16` | — | Enable FP16 (half precision) in addition to FP32 |
| `--builderOptimizationLevel` | 5 (max) | Builder spends more time searching for better optimization strategies (default: 3, range: 0-5) |
| `--tilingOptimizationLevel` | 3 (max) | Higher tiling optimization search depth (default: 0, range: 0-3) |
| `--avgTiming` | 16 | Number of iterations averaged per kernel timing for tactic selection (default: 8) |
| `--useCudaGraph` | — | Capture engine execution into a CUDA graph for lower launch overhead |
| `--useManagedMemory` | — | Use CUDA Unified Memory (matches `cudaMallocManaged` in application code) |
| `--exposeDMA` | — | Serialize DMA (Direct Memory Access) transfers for benchmarking |
| `--noDataTransfers` | — | Disable host-device data transfers during benchmarking |
| `--timingCacheFile` | `./timing.cache` | Save/load global timing cache to speed up subsequent builds |
| `--separateProfileRun` | — | Run profiling in a separate pass for accurate e2e (end-to-end) timing |
| `--dumpProfile` | — | Print per-layer profile information |

### Why NOT `--best`

`--best` enables all precisions including INT8, which requires a calibrator with representative data. Without calibration, INT8 tactics are ignored and the builder may select suboptimal strategies, resulting in **larger engine size and slower inference**.
# Fighter-Tracking
