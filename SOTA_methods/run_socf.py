#!/usr/bin/env python3
"""SOCF runner CLI. Reads infrared.mp4, init box from tools/init_boxes.json,
prints per-frame:

  BOX:cx,cy W,H:w,h
  RUN time: X ms

matching main.cc output so tools/eval_one.py parses it identically.
"""
import json
import sys
import time
from pathlib import Path

import cv2

# Allow `python SOTA_methods/run_socf.py <video>` from project root.
sys.path.insert(0, str(Path(__file__).parent))
from socf.socf_tracker import SOCFTracker


def main():
    if len(sys.argv) < 2:
        print("usage: run_socf.py <video.mp4>", file=sys.stderr)
        sys.exit(1)
    video = sys.argv[1]
    project_root = Path(__file__).resolve().parents[1]
    init_boxes = json.load(open(project_root / "tools" / "init_boxes.json"))
    key = Path(video).parent.name
    if key not in init_boxes:
        print(f"init box for '{key}' missing", file=sys.stderr)
        sys.exit(1)
    init_box = init_boxes[key]["box"]
    print(f"init_box: {init_box}", flush=True)

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    if not ok:
        print("cannot read video", file=sys.stderr)
        sys.exit(1)
    gray = (frame if frame.ndim == 2
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    tracker = SOCFTracker()
    tracker.init(gray, init_box)
    # Frame 0: skip output (main.cc also doesn't emit BOX for frame 0).

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = (frame if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        t0 = time.perf_counter()
        x, y, w, h = tracker.update(gray)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        # Match main.cc format exactly:
        #   BOX:cx,cy W,H:w,h
        #   RUN time: X ms
        #   <blank>
        print(f"BOX:{cx},{cy} W,H:{int(w)},{int(h)}", flush=True)
        print(f"RUN time: {dt_ms:.6f} ms\n", flush=True)

    cap.release()


if __name__ == "__main__":
    main()
