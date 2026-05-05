#!/usr/bin/env python3
"""
Box the drone in the first frame of every test*/infrared.mp4 using
Grounding DINO Tiny (IDEA-Research) zero-shot detection.

Strategy:
- Load Grounding DINO Tiny once, reuse for all videos
- Try multiple text prompts (drone-shaped, then bright-spot fallback)
- For each video: take frame 0, run detection, pick best-scoring box
- Save bboxes to <out_json> as { "<video_name>": [x, y, w, h], ... }
- Also write per-video annotated PNG to <viz_dir>/

Usage:
    python tools/detect_first_frame.py \
        --videos-root Datasets/Anti-UAV-RGBT/test \
        --out-json /tmp/first_frame_boxes.json \
        --viz-dir /tmp/first_frame_viz
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# Text prompts to try, in order of specificity. Grounding DINO accepts
# period-separated phrases as alternative classes within a single forward.
PROMPT = "drone. small aircraft. bright dot. small bright object."

# Confidence thresholds (tuned for small-IR-target regime)
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20


def load_model(device: str):
    """Load Grounding DINO Tiny processor + model on the requested device."""
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )
    model.eval()
    return processor, model


def first_frame(video_path: Path):
    """Read frame 0 as a PIL Image (RGB). Returns None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@torch.no_grad()
def detect_drone(image: Image.Image, processor, model, device: str):
    """
    Returns the highest-scoring detection as (x, y, w, h, score, label),
    or None if nothing passed the threshold.
    """
    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    # post-processing: convert outputs to (xyxy, score, label_text) lists,
    # filtered by threshold
    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    if len(results["scores"]) == 0:
        return None

    # Filter: reject boxes that span more than half the image in either
    # dimension. Drones are small targets; a near-full-frame box is the
    # detector grabbing the whole sky and means real detection failed.
    img_w, img_h = image.size
    max_w = img_w * 0.5
    max_h = img_h * 0.5
    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    keep = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw <= max_w and bh <= max_h:
            keep.append(i)
    if not keep:
        return None

    # Pick the kept box with the highest score.
    best_idx = max(keep, key=lambda i: float(scores[i]))
    box = boxes[best_idx].tolist()  # [x1, y1, x2, y2]
    score = float(scores[best_idx])
    label = results.get("labels", results.get("text_labels", ["?"]))[best_idx]
    if hasattr(label, "item"):
        label = str(label)
    x1, y1, x2, y2 = box
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2 - x1)),
        int(round(y2 - y1)),
        score,
        str(label),
    )


def annotate(image: Image.Image, det, gt_box=None) -> "cv2.Mat":
    """Draw detection (green) and optional GT (red) on the image."""
    bgr = cv2.cvtColor(__import__("numpy").array(image), cv2.COLOR_RGB2BGR)
    if det is not None:
        x, y, w, h, score, label = det
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"{label} {score:.2f}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    if gt_box is not None:
        gx, gy, gw, gh = gt_box
        cv2.rectangle(bgr, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 1)
    return bgr


def load_gt(video_dir: Path):
    """Read frame-0 GT box from infrared.json if present, else None."""
    json_path = video_dir / "infrared.json"
    if not json_path.exists():
        return None
    with json_path.open() as f:
        data = json.load(f)
    rects = data.get("gt_rect") or []
    if not rects:
        return None
    box = rects[0]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return list(box)  # [x, y, w, h]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos-root", type=Path, required=True,
                   help="root containing test*/infrared.mp4 subdirs")
    p.add_argument("--out-json", type=Path, required=True,
                   help="path for the JSON dict {video_name: [x,y,w,h,score]}")
    p.add_argument("--viz-dir", type=Path,
                   help="optional dir for annotated PNGs")
    p.add_argument("--prefix", default="test",
                   help="only process subdirs starting with this prefix")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    print(f"loading Grounding DINO Tiny on {args.device}...")
    processor, model = load_model(args.device)

    if args.viz_dir:
        args.viz_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    subdirs = [d for d in sorted(args.videos_root.iterdir())
               if d.is_dir() and d.name.startswith(args.prefix)]
    print(f"found {len(subdirs)} videos under {args.videos_root}")

    for sub in subdirs:
        vp = sub / "infrared.mp4"
        if not vp.exists():
            print(f"  skip {sub.name}: no infrared.mp4")
            continue
        img = first_frame(vp)
        if img is None:
            print(f"  skip {sub.name}: cannot read frame 0")
            continue
        det = detect_drone(img, processor, model, args.device)
        if det is None:
            print(f"  {sub.name}: NO detection")
            out[sub.name] = None
        else:
            x, y, w, h, score, label = det
            print(f"  {sub.name}: [{x},{y},{w},{h}] score={score:.3f} '{label}'")
            out[sub.name] = {"box": [x, y, w, h], "score": score, "label": label}
        if args.viz_dir:
            gt = load_gt(sub)
            annotated = annotate(img, det, gt)
            cv2.imwrite(str(args.viz_dir / f"{sub.name}.png"), annotated)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")
    if args.viz_dir:
        print(f"wrote viz to {args.viz_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
