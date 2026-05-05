"""
Export the Hessian-DoH (Determinant-of-Hessian) blob detector to ONNX.

Pipeline:
  1. Normalize input by dividing by 255
  2. Hessian (Dxx, Dxy, Dyy) via grouped conv directly on normalized image
  3. DoH (Determinant of Hessian): response = ReLU(Dxx*Dyy - Dxy^2)
     - Bright/dark blob centers have Dxx, Dyy large same-sign and Dxy ~= 0
       so det(H) is large positive (perfect blob detector — what SURF uses).
     - Edges and saddles have det(H) ~= 0 or negative -> suppressed.
     - Replaces the previous Frangi-style "no-sqrt vesselness" formula
       which suppressed isotropic blobs because of the (lambda_max-lambda_min)
       term — wrong for tracking a single bright drone target.
  4. Apply 10-pixel binary border mask
  5. Outputs: response [1,1,H,W], x_max [W] (max along rows), y_max [H] (max along cols)

Hardware operations: conv, mul, sub, ReLU only (no sqrt/abs/exp).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormHessianDoHMaskedMax(nn.Module):
    """
    Determinant-of-Hessian blob detector for tracking a single bright drone.
    Fixed input size 256x256.

    Pipeline:
      1. image / 255.0
      2. Hessian (Dxx, Dxy, Dyy) via single 3-channel grouped conv
      3. DoH: response = ReLU(Dxx*Dyy - Dxy^2)
      4. Apply 10-pixel binary border mask
      5. Per-axis max projection
    """

    def __init__(self, height=256, width=256, border_mask_size=10):
        super().__init__()
        self.height = height
        self.width = width
        self.border_mask_size = border_mask_size

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
        resp_mask = torch.zeros(1, self.height, self.width)
        b = self.border_mask_size
        resp_mask[:, b:-b, b:-b] = 1.0
        self.register_buffer("resp_mask", resp_mask)

    def forward(self, image_clone: torch.Tensor):
        # -- Step 1: Normalize to [0, 1] --
        x = image_clone / 255.0

        # -- Step 2: Hessian via 3-channel grouped conv --
        x_3c = torch.cat((x, x, x), dim=1)
        hessian = self.response_conv(x_3c)  # [1, 3, H, W]
        dxx = hessian[:, 0:1, :, :]
        dxy = hessian[:, 1:2, :, :]
        dyy = hessian[:, 2:3, :, :]

        # -- Step 3: Determinant of Hessian (blob detector) --
        # det(H) = Dxx*Dyy - Dxy^2
        # Bright blob center: Dxx, Dyy < 0, Dxy ~= 0  ->  det > 0  (kept)
        # Dark blob center : Dxx, Dyy > 0, Dxy ~= 0   ->  det > 0  (kept)
        # Edge / saddle    : sign mixed or one ~= 0    ->  det <= 0 (clipped)
        response = F.relu(dxx * dyy - dxy * dxy)

        # -- Step 4: Apply 10-pixel border mask --
        response = response * self.resp_mask

        # -- Step 5: Per-axis max projection --
        resp_2d = response.view(self.height, self.width)  # [H, W]
        x_max = resp_2d.max(dim=0).values  # [W]
        y_max = resp_2d.max(dim=1).values  # [H]

        return response, x_max, y_max


def export_onnx(output_path: str = "Norm_Hessian_DoH_256.onnx"):
    model = NormHessianDoHMaskedMax()
    model.eval()

    dummy_input = torch.randn(1, 1, 256, 256)

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


if __name__ == "__main__":
    export_onnx("Norm_Hessian_DoH_256.onnx")
