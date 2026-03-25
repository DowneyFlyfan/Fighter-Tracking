"""
Export the NormFrangiNoSqrt vesselness module to ONNX and compare
against the existing .onnx file.

Pipeline:
  1. Normalize input by dividing by 255
  2. Hessian (Dxx, Dxy, Dyy) via grouped conv directly on normalized image
  3. No-sqrt vesselness: disc = ReLU(trace² - 4*det), response = ReLU((|a|+|b|)*(a-b))
  4. Apply 10-pixel binary border mask
  5. Outputs: response [1,1,H,W], x_max [W] (max along rows), y_max [H] (max along cols)

Hardware operations: conv, mul, abs, ReLU only (no sqrt/exp).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormFrangiNoSqrtMaskedMax(nn.Module):
    """
    Hardware-friendly Frangi vesselness for target tracking (no sqrt).

    Uses trace²-4*det discriminant (standard characteristic polynomial
    discriminant = (λ1-λ2)²) instead of trace²-4*det².
    Fixed input size 480x480.

    Pipeline:
      1. image / 255.0
      2. Hessian (Dxx, Dxy, Dyy) via grouped conv directly on normalized image
      3. No-sqrt vesselness: disc = ReLU(trace² - 4*det)
      4. Apply 10-pixel binary border mask
      5. Per-axis max projection
    """

    H = 480
    W = 480
    RESP_BORDER = 10  # border pixels zeroed on response

    def __init__(self):
        super().__init__()

        # Second-order Gaussian derivative kernels [3, 1, 7, 7]
        # Channel 0: g_xx, Channel 1: g_xy, Channel 2: g_yy
        g_xx = torch.tensor(
            [
                [
                    1.5713e-04,
                    7.1784e-04,
                    0.0000e00,
                    -1.7681e-03,
                    0.0000e00,
                    7.1784e-04,
                    1.5713e-04,
                ],
                [
                    1.9142e-03,
                    8.7451e-03,
                    0.0000e00,
                    -2.1539e-02,
                    0.0000e00,
                    8.7451e-03,
                    1.9142e-03,
                ],
                [
                    8.5790e-03,
                    3.9193e-02,
                    0.0000e00,
                    -9.6532e-02,
                    0.0000e00,
                    3.9193e-02,
                    8.5790e-03,
                ],
                [
                    1.4144e-02,
                    6.4618e-02,
                    0.0000e00,
                    -1.5915e-01,
                    0.0000e00,
                    6.4618e-02,
                    1.4144e-02,
                ],
                [
                    8.5790e-03,
                    3.9193e-02,
                    0.0000e00,
                    -9.6532e-02,
                    0.0000e00,
                    3.9193e-02,
                    8.5790e-03,
                ],
                [
                    1.9142e-03,
                    8.7451e-03,
                    0.0000e00,
                    -2.1539e-02,
                    0.0000e00,
                    8.7451e-03,
                    1.9142e-03,
                ],
                [
                    1.5713e-04,
                    7.1784e-04,
                    0.0000e00,
                    -1.7681e-03,
                    0.0000e00,
                    7.1784e-04,
                    1.5713e-04,
                ],
            ]
        )
        g_xy = torch.tensor(
            [
                [
                    2.0000e-04,
                    1.4000e-03,
                    3.2000e-03,
                    -0.0000e00,
                    -3.2000e-03,
                    -1.4000e-03,
                    -2.0000e-04,
                ],
                [
                    1.4000e-03,
                    1.1700e-02,
                    2.6100e-02,
                    -0.0000e00,
                    -2.6100e-02,
                    -1.1700e-02,
                    -1.4000e-03,
                ],
                [
                    3.2000e-03,
                    2.6100e-02,
                    5.8500e-02,
                    -0.0000e00,
                    -5.8500e-02,
                    -2.6100e-02,
                    -3.2000e-03,
                ],
                [
                    -0.0000e00,
                    -0.0000e00,
                    -0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -3.2000e-03,
                    -2.6100e-02,
                    -5.8500e-02,
                    0.0000e00,
                    5.8500e-02,
                    2.6100e-02,
                    3.2000e-03,
                ],
                [
                    -1.4000e-03,
                    -1.1700e-02,
                    -2.6100e-02,
                    0.0000e00,
                    2.6100e-02,
                    1.1700e-02,
                    1.4000e-03,
                ],
                [
                    -2.0000e-04,
                    -1.4000e-03,
                    -3.2000e-03,
                    0.0000e00,
                    3.2000e-03,
                    1.4000e-03,
                    2.0000e-04,
                ],
            ]
        )
        g_yy = torch.tensor(
            [
                [
                    1.5713e-04,
                    1.9142e-03,
                    8.5790e-03,
                    1.4144e-02,
                    8.5790e-03,
                    1.9142e-03,
                    1.5713e-04,
                ],
                [
                    7.1784e-04,
                    8.7451e-03,
                    3.9193e-02,
                    6.4618e-02,
                    3.9193e-02,
                    8.7451e-03,
                    7.1784e-04,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -1.7681e-03,
                    -2.1539e-02,
                    -9.6532e-02,
                    -1.5915e-01,
                    -9.6532e-02,
                    -2.1539e-02,
                    -1.7681e-03,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    7.1784e-04,
                    8.7451e-03,
                    3.9193e-02,
                    6.4618e-02,
                    3.9193e-02,
                    8.7451e-03,
                    7.1784e-04,
                ],
                [
                    1.5713e-04,
                    1.9142e-03,
                    8.5790e-03,
                    1.4144e-02,
                    8.5790e-03,
                    1.9142e-03,
                    1.5713e-04,
                ],
            ]
        )
        response_weight = torch.stack([g_xx, g_xy, g_yy]).unsqueeze(1)  # [3, 1, 7, 7]
        self.response_conv = nn.Conv2d(3, 3, 7, padding=3, groups=3, bias=False)
        self.response_conv.weight = nn.Parameter(response_weight, requires_grad=False)

        # 10-pixel binary border mask for response [1, H, W]
        resp_mask = torch.zeros(1, self.H, self.W)
        b = self.RESP_BORDER
        resp_mask[:, b:-b, b:-b] = 1.0
        self.register_buffer("resp_mask", resp_mask)

    def forward(self, image_clone: torch.Tensor):
        # -- Step 1: Normalize to [0, 1] --
        x = image_clone / 255.0

        # -- Step 2: Hessian via grouped conv --
        x_3c = torch.cat((x, x, x), dim=1)
        hessian = self.response_conv(x_3c)  # [1, 3, H, W]
        dxx = hessian[:, 0:1, :, :]
        dxy = hessian[:, 1:2, :, :]
        dyy = hessian[:, 2:3, :, :]

        # -- Step 3: No-sqrt vesselness (trace²-4*det) --
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy

        trace_sq = trace * trace
        discriminant = F.relu(trace_sq - 4.0 * det)

        a = trace + discriminant
        b = trace - discriminant

        response = F.relu((torch.abs(a) + torch.abs(b)) * (a - b))

        # -- Step 4: Apply 10-pixel border mask --
        response = response * self.resp_mask

        # -- Step 5: Per-axis max projection --
        resp_2d = response.view(self.H, self.W)  # [H, W]
        x_max = resp_2d.max(dim=0).values  # [W]
        y_max = resp_2d.max(dim=1).values  # [H]

        return response, x_max, y_max


def export_onnx(output_path: str = "Norm_Frangi_NoSqrt_480.onnx"):
    model = NormFrangiNoSqrtMaskedMax()
    model.eval()

    dummy_input = torch.randn(1, 1, 480, 480)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["image_clone"],
        output_names=["response", "x_max", "y_max"],
        opset_version=18,
        do_constant_folding=True,
        optimize=True,
    )
    print(f"Exported to {output_path}")


def compare_models(original_path: str, new_path: str):
    """Run both ONNX models on the same input and compare outputs."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        return

    # Create reproducible test input
    np.random.seed(42)
    test_input = np.random.rand(1, 1, 480, 480).astype(np.float32) * 255

    sess_orig = ort.InferenceSession(original_path)
    sess_new = ort.InferenceSession(new_path)

    out_orig = sess_orig.run(None, {"image_clone": test_input})
    out_new = sess_new.run(None, {"image_clone": test_input})

    names = ["response", "x_max", "y_max"]
    for i, name in enumerate(names):
        o = out_orig[i]
        n = out_new[i]
        max_diff = np.max(np.abs(o - n))
        mean_diff = np.mean(np.abs(o - n))
        cos_sim = np.dot(o.flatten(), n.flatten()) / (
            np.linalg.norm(o.flatten()) * np.linalg.norm(n.flatten()) + 1e-8
        )
        print(f"[{name}] shape_orig={o.shape} shape_new={n.shape}")
        print(
            f"  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  cos_sim={cos_sim:.6f}"
        )
        print(f"  orig range=[{o.min():.4f}, {o.max():.4f}]")
        print(f"  new  range=[{n.min():.4f}, {n.max():.4f}]")
        print()


if __name__ == "__main__":
    import sys

    nosqrt_path = "Norm_Frangi_NoSqrt_480.onnx"
    original_path = (
        "/home/downeyflyfan/Research_Projects/Track/CPP-Track/"
        "hspeedtrack_x86/onnx2trt/Norm_Grad_Response_Masked_Max_480.onnx"
    )

    export_onnx(nosqrt_path)
    print("\n=== Original (det²) vs NormFrangiNoSqrt (det, no sqrt) ===")
    compare_models(original_path, nosqrt_path)
