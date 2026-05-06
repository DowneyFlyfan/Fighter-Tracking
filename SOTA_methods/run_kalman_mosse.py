#!/usr/bin/env python3
"""KalmanMOSSE runner CLI. Same I/O as run_socf.py / main.cc."""
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from socf.kalman_mosse import KalmanMOSSE


def main():
    if len(sys.argv) < 2:
        print("usage: run_kalman_mosse.py <video.mp4>", file=sys.stderr)
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

    tracker = KalmanMOSSE()
    tracker.init(gray, init_box)

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
        print(f"BOX:{cx},{cy} W,H:{int(w)},{int(h)}", flush=True)
        print(f"RUN time: {dt_ms:.6f} ms\n", flush=True)

    cap.release()


if __name__ == "__main__":
    main()
