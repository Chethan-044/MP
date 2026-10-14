"""
vidiotoframe.py  --  video -> lossless PNG frames (+ video_metadata.json with fps).
Standalone helper (app.py already extracts frames itself).

    python vidiotoframe.py --video input.mp4 [--output frames]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default=os.path.join(BASE, "frames"))
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {a.video}", flush=True)
        return 2
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    os.makedirs(a.output, exist_ok=True)
    for n in os.listdir(a.output):
        if n.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(a.output, n))

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        good, buf = cv2.imencode(".png", frame)
        if not good:
            print(f"ERROR: could not encode frame {count}", flush=True)
            return 1
        buf.tofile(os.path.join(a.output, f"frame_{count:06d}.png"))
        count += 1
        print(f"[{count}/{max(total, count)}] frame_{count - 1:06d}.png", flush=True)
    cap.release()

    with open(os.path.join(BASE, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"source_video": os.path.basename(a.video), "fps": fps, "frame_count": count}, f, indent=4)
    print(f"Done: {count} frames.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())