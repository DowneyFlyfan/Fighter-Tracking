"""Export ConvNeXtV2-Atto pretrained on ImageNet to ONNX for use as a
generic shape feature extractor. Output = pre-classifier global-pooled
feature vector (320-d for Atto).

Input: 128x128x3 uint8 normalized to ImageNet mean/std internally by
the wrapper module. Drone patches in IR are gray-scale; the wrapper
takes 1-channel input and replicates to 3 channels.
"""

import torch
import torch.nn as nn
import timm

INPUT_SIZE = 128


class GrayConvNeXtAttoEmbedder(nn.Module):
    """Wraps ConvNeXtV2-Atto for grayscale -> 320-d embedding.

    - Input  : (1, 1, H, W) float32 in [0, 255]
    - Output : (1, 320)     float32, L2-normalized
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_atto.fcmae_ft_in1k",
            pretrained=True,
            num_classes=0,        # drop the classifier head
            global_pool="avg",    # 320-d vector
        )
        # ImageNet normalisation (input is 0-255 grayscale, divide by 255 first)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        # x: (B, 1, H, W), uint8-range float
        x = x / 255.0
        x = x.expand(-1, 3, -1, -1)              # gray -> 3-channel replicate
        x = (x - self.mean) / self.std            # ImageNet normalise
        feat = self.backbone(x)                   # (B, 320)
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)
        return feat


def main():
    model = GrayConvNeXtAttoEmbedder().eval()
    # Quick param count.
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.2f}M")

    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    out_path = "engine_model/ConvNeXtV2_Atto_Embedder.onnx"
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["patch"], output_names=["embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={"patch": {0: "B"}, "embedding": {0: "B"}},
    )
    print(f"saved: {out_path}")
    # Quick sanity: emit one forward.
    with torch.no_grad():
        e = model(dummy)
    print(f"embedding shape: {e.shape}, |e|={e.norm(dim=1).item():.3f}")


if __name__ == "__main__":
    main()
