"""
frametovideo.py  --  PNG frames -> video.

Called by app.py:   python frametovideo.py --input encrypted_frames --output <file>
                    python frametovideo.py --input decrypted_frames --output <file>

Videos are for VIEWING only.  Decryption always reads the lossless PNGs in encrypted_frames/
directly, never a video, so a normal codec is fine:  .mp4/.mov/.mkv -> mp4v,  .avi -> FFV1/MJPG.
Pass --lossless to force a verified bit-exact FFV1 video.
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def read_fps():
    try:
        with open(os.path.join(BASE, "video_metadata.json"), encoding="utf-8") as f:
            fps = float(json.load(f).get("fps", 25.0))
        return fps if fps > 0 else 25.0
    except Exception:
        return 25.0


def numbered_frames(folder):
    files = sorted(n for n in os.listdir(folder) if n.lower().endswith(".png"))
    if not files:
        raise RuntimeError(f"No PNG frames in {folder}")
    nums = []
    for n in files:
        m = re.findall(r"\d+", os.path.splitext(n)[0])
        nums.append(int(m[-1]) if m else -1)
    if -1 in nums or nums != list(range(nums[0], nums[0] + len(nums))):
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))[:10]
        raise RuntimeError(f"Frame sequence has gaps (missing e.g. {missing}). "
                           "Some frames failed earlier - fix them before building the video.")
    return [os.path.join(folder, n) for n in files]


def read_png(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lossless", action="store_true")
    a = ap.parse_args(argv)

    folder = a.input if os.path.isabs(a.input) else os.path.join(BASE, a.input)
    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}", flush=True)
        return 2
    lossless = a.lossless
    ext = os.path.splitext(a.output)[1].lower()

    try:
        files = numbered_frames(folder)
        first = read_png(files[0])
        h, w = first.shape[:2]
        fps = read_fps()
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)

        if lossless:
            codecs = ["FFV1"]
        elif ext == ".avi":
            codecs = ["FFV1", "MJPG"]
        else:
            codecs = ["mp4v"]

        writer = None
        for c in codecs:
            writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*c), fps, (w, h))
            if writer.isOpened():
                codec = c
                break
            writer.release()
            writer = None
        if writer is None:
            print(f"ERROR: OpenCV cannot write {'/'.join(codecs)} to '{ext}'. "
                  "Try saving as .avi.", flush=True)
            return 1

        total = len(files)
        for n, f in enumerate(files, 1):
            img = first if n == 1 else read_png(f)
            if img.shape[:2] != (h, w):
                writer.release()
                os.remove(a.output)
                raise RuntimeError(f"{os.path.basename(f)} is {img.shape[1]}x{img.shape[0]}, expected {w}x{h}.")
            writer.write(img)
            print(f"[{n}/{total}] {os.path.basename(f)}", flush=True)
        writer.release()

        if lossless:                                   # prove the video is bit-exact
            cap = cv2.VideoCapture(a.output)
            picks = sorted({0, total // 2, total - 1})
            for idx in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, got = cap.read()
                if not ok or not np.array_equal(got, read_png(files[idx])):
                    cap.release()
                    os.remove(a.output)
                    print(f"ERROR: video read-back of frame {idx} is not bit-exact ({codec} in '{ext}'). "
                          "Encrypted video would be undecryptable. Try saving as .avi.", flush=True)
                    return 1
            cap.release()
            print(f"Verified lossless ({codec}).", flush=True)

        print(f"Done: {a.output}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())