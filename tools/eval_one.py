#!/usr/bin/env python3
"""Run a tracker command on a video, parse BOX/W,H stdout, compare to GT json.
Reports w5 / w10 / w25 (% frames within N px center error), IoU mean,
center-error mean / median.

Default tracker is ./main (the C++ binary). Override with --cmd "...":
    eval_one.py <video_dir> --cmd "python SOTA_methods/run_socf.py"
The video mp4 path is appended as the final argument to <cmd>.
"""
import json, re, shlex, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]
cmd = ["./main"]
if "--cmd" in argv:
    i = argv.index("--cmd")
    cmd = shlex.split(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
if len(argv) < 1:
    print("usage: eval_one.py <video_dir> [--cmd \"<tracker cmd>\"]")
    sys.exit(1)

vid_dir = Path(argv[0])
mp4 = vid_dir / "infrared.mp4"
gt  = json.load(open(vid_dir / "infrared.json"))["gt_rect"]
exist = json.load(open(vid_dir / "infrared.json"))["exist"]

out = subprocess.run(cmd + [str(mp4)], capture_output=True, text=True).stdout
boxes = re.findall(r"BOX:(-?\d+),(-?\d+) W,H:(\d+),(\d+)", out)
print(f"parsed {len(boxes)} pred boxes (frame 1..N)")

# Frame 0 is init box, predictions start from frame 1 → index 1..N
errs, ious, w5 = [], [], 0
w10 = 0; w25 = 0; n = 0
for i, (pcx, pcy, pw, ph) in enumerate(boxes):
    f = i + 1
    if f >= len(gt) or not exist[f] or not gt[f]: continue
    gx, gy, gw, gh = gt[f]
    gcx, gcy = gx + gw/2, gy + gh/2
    pcx, pcy, pw, ph = int(pcx), int(pcy), int(pw), int(ph)
    err = ((pcx - gcx)**2 + (pcy - gcy)**2)**0.5
    errs.append(err)
    if err <= 5:  w5  += 1
    if err <= 10: w10 += 1
    if err <= 25: w25 += 1
    # IoU
    px0, py0 = pcx - pw/2, pcy - ph/2
    px1, py1 = pcx + pw/2, pcy + ph/2
    ix0, iy0 = max(px0, gx), max(py0, gy)
    ix1, iy1 = min(px1, gx + gw), min(py1, gy + gh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = pw*ph + gw*gh - inter
    ious.append(inter / union if union > 0 else 0.0)
    n += 1

if n == 0:
    print("no valid frames"); sys.exit(0)
errs.sort()
med = errs[len(errs)//2]
print(f"n={n}  w5={100*w5/n:.1f}%  w10={100*w10/n:.1f}%  w25={100*w25/n:.1f}%"
      f"  IoU={sum(ious)/n:.3f}  mean_err={sum(errs)/n:.1f}  med_err={med:.1f}")
