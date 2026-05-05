#!/usr/bin/env python3
"""
Batch HUD-mask Anti-UAV-RGBT infrared videos.

For each source video, write a copy to <out_dir> with the three HUD text
rectangles (top-left line 1, top-left line 2, top-right timestamp) hidden
by tiling the row immediately below each mask up through the mask's
height. This adapts to local background intensity so the patched region
blends in instead of leaving a hard black box.

Mask rectangle constants mirror utils/types.h (LABEL_MASK_*).

Usage examples:
    # Mask one file, write to /tmp/test1_masked.mp4
    python3 tools/mask_hud.py Datasets/Anti-UAV-RGBT/test/test1/infrared.mp4 \
        /tmp/test1_masked.mp4

    # Batch: mask every test*/infrared.mp4 under <root> and write masked
    # copies to <out_dir> mirroring the directory layout
    python3 tools/mask_hud.py --batch Datasets/Anti-UAV-RGBT/test \
        Datasets/Anti-UAV-RGBT/test_masked
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Mirror of utils/types.h LABEL_MASK_* constants
L1_X, L1_Y, L1_W, L1_H = 0, 0, 150, 38
L2_X, L2_Y, L2_W, L2_H = 0, 55, 220, 40
R_W, R_H = 260, 38


def fill_from_below(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Tile the row at (x, y+h) up through rows [y, y+h)."""
    src_row = img[y + h : y + h + 1, x : x + w]
    img[y : y + h, x : x + w] = np.repeat(src_row, h, axis=0)


def mask_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a BGR frame with the three HUD text rectangles blended away."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fill_from_below(gray, L1_X, L1_Y, L1_W, L1_H)
    fill_from_below(gray, L2_X, L2_Y, L2_W, L2_H)
    fill_from_below(gray, gray.shape[1] - R_W, 0, R_W, R_H)
    # Convert back to BGR so downstream tools that expect 3-channel input work
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def mask_video(in_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {in_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {out_path}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(mask_frame(frame))
        n += 1
    cap.release()
    writer.release()
    print(f"  wrote {n} frames -> {out_path}")


def batch(src_root: Path, dst_root: Path, name_filter: str = "test") -> None:
    """Mask every <src_root>/<name_filter>*/infrared.mp4 into <dst_root>/<name>/infrared.mp4."""
    for sub in sorted(src_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith(name_filter):
            continue
        in_path = sub / "infrared.mp4"
        if not in_path.exists():
            continue
        out_path = dst_root / sub.name / "infrared.mp4"
        print(f"masking {in_path}")
        mask_video(in_path, out_path)


def in_place_batch(root: Path, name_filter: str = "test") -> None:
    """Overwrite <root>/<name_filter>*/infrared.mp4 with masked versions.

    Uses a sibling .tmp.mp4 then atomic rename, so a crash mid-encode
    won't corrupt the source.
    """
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith(name_filter):
            continue
        in_path = sub / "infrared.mp4"
        if not in_path.exists():
            continue
        tmp_path = sub / "infrared.tmp.mp4"
        print(f"masking {in_path} -> in place")
        mask_video(in_path, tmp_path)
        tmp_path.replace(in_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", action="store_true",
                   help="treat positional args as src_root and dst_root")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite each <src>/test*/infrared.mp4 with its "
                        "masked version via atomic rename. Only the first "
                        "positional argument (src_root) is used; dst is "
                        "ignored.")
    p.add_argument("--prefix", default="test",
                   help="only process subdirs whose name starts with this "
                        "prefix (default: 'test')")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path, nargs="?")
    args = p.parse_args()
    if args.in_place:
        in_place_batch(args.src, args.prefix)
    elif args.batch:
        if args.dst is None:
            p.error("--batch requires both src and dst positional args")
        batch(args.src, args.dst, args.prefix)
    else:
        if args.dst is None:
            p.error("single-file mode requires both src and dst")
        mask_video(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
