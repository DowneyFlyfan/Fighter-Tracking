"""Run ./main on a video, overlay BOTH predicted box (green) and
ground-truth box (red) on each frame, write to /tmp/viz_<name>.mp4.

Also annotates frame index + center error so you can scrub to bad
frames quickly."""
import json, re, subprocess, sys
from pathlib import Path
import cv2

if len(sys.argv) < 2:
    print("usage: viz_compare.py <video_dir>")
    sys.exit(1)

vid_dir = Path(sys.argv[1])
mp4 = vid_dir / "infrared.mp4"
gt = json.load(open(vid_dir / "infrared.json"))["gt_rect"]
out_path = f"/tmp/viz_{vid_dir.name}.mp4"

# Parse predicted boxes from main stdout.
out = subprocess.run(["./main", str(mp4)], capture_output=True, text=True).stdout
boxes = re.findall(r"BOX:(-?\d+),(-?\d+) W,H:(\d+),(\d+)", out)
print(f"parsed {len(boxes)} pred boxes")

cap = cv2.VideoCapture(str(mp4))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, 30.0, (W, H))

f_idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if frame.ndim == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # GT: red
    if f_idx < len(gt):
        gx, gy, gw, gh = gt[f_idx]
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)
        gcx, gcy = gx + gw / 2, gy + gh / 2
    else:
        gcx, gcy = -1, -1
    # Pred (green) — main skips frame 0 (uses init), so pred[i] = frame i+1.
    pi = f_idx - 1
    err = -1
    if 0 <= pi < len(boxes):
        cx, cy, w, h = (int(v) for v in boxes[pi])
        x0 = cx - w // 2; y0 = cy - h // 2
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        if gcx >= 0:
            err = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
    txt = f"f={f_idx}  err={err:.1f}px" if err >= 0 else f"f={f_idx}"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    writer.write(frame)
    f_idx += 1

cap.release(); writer.release()
print(f"wrote {out_path}  ({f_idx} frames)")
