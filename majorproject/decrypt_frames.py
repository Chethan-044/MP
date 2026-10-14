"""
decrypt_frames.py  --  restore ROIs using ONLY the metadata embedded in each frame (no YOLO).

Called by app.py:   python decrypt_frames.py --seed 0.37 --factor 5 --input <dir>
Reads  <dir>/*.png        Writes  decrypted_frames/frame_<id>.png  (lossless)

Steps per frame:  extract metadata -> verify (CRC, frame size, seed/factor, ROI bounds)
                  -> XOR -> inverse column permutation -> inverse row permutation
                  -> place ROI back at its original coordinates -> verify ROI CRC.
A frame that fails ANY check is reported and NOT written (an encrypted frame is never
saved as a decrypted one).  Exit code is non-zero if any frame failed.
"""
import argparse
import os
import sys
import zlib

import numpy as np

from decode_steganography import extract_metadata
from frameencryption import CryptoError, check_key, clear_pngs, decrypt_roi, key_check
from steganography import MetadataNotFound, StegoError, load_png, save_png, validate_boxes

BASE = os.path.dirname(os.path.abspath(__file__))
ENCRYPTED = os.path.join(BASE, "encrypted_frames")
DECRYPTED = os.path.join(BASE, "decrypted_frames")


class KeyMismatchError(CryptoError):
    pass


def decrypt_frame(frame, seed, division):
    """Returns (restored_frame, meta). Decrypts ROIs based on the provided seed and division."""
    seed, division = check_key(seed, division)
    frame = np.ascontiguousarray(frame)

    swapped = False                                  # tolerate a BGR<->RGB mismatch between save and load
    try:
        meta = extract_metadata(frame)
    except MetadataNotFound:
        flipped = np.ascontiguousarray(frame[..., ::-1])
        meta = extract_metadata(flipped)             # raises MetadataNotFound if truly absent
        frame, swapped = flipped, True

    fh, fw = frame.shape[:2]
    if (meta["frame_w"], meta["frame_h"]) != (fw, fh):
        raise CryptoError(f"Frame is {fw}x{fh} but metadata says {meta['frame_w']}x{meta['frame_h']}.")

    validate_boxes(meta["rois"], fw, fh)

    out = frame.copy()
    for i in reversed(range(meta["roi_count"])):     # reverse of encryption order
        x, y, w, h = meta["rois"][i][:4]
        dec = decrypt_roi(out[y:y + h, x:x + w].copy(), seed, division, meta["frame_id"], i)
        out[y:y + h, x:x + w] = dec
    if swapped:
        out = np.ascontiguousarray(out[..., ::-1])
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--factor", required=True)
    ap.add_argument("--input", default=ENCRYPTED)
    ap.add_argument("--output", default=DECRYPTED)
    a = ap.parse_args(argv)

    try:
        seed, factor = check_key(a.seed, a.factor)
    except CryptoError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    if not os.path.isdir(a.input):
        print(f"ERROR: input folder not found: {a.input}", flush=True)
        return 2
    files = sorted(os.path.join(a.input, n) for n in os.listdir(a.input) if n.lower().endswith(".png"))
    if not files:
        print(f"ERROR: no PNG frames in {a.input}", flush=True)
        return 2

    clear_pngs(a.output)
    failed, seen, total = [], set(), len(files)
    for n, path in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            dec, meta = decrypt_frame(load_png(path), seed, factor)
            fid = meta["frame_id"]
            if fid in seen:
                raise CryptoError(f"Duplicate frame ID {fid}.")
            save_png(os.path.join(a.output, f"frame_{fid:06d}.png"), dec)
            seen.add(fid)
            print(f"[{n}/{total}] {name} -> frame {fid}  ROIs={meta['roi_count']}", flush=True)
        except (StegoError, CryptoError, OSError) as e:
            failed.append((name, str(e)))
            print(f"[{n}/{total}] FAILED {name}: {e}", flush=True)

    if failed:
        print(f"ERROR: {len(failed)} of {total} frames FAILED and were NOT written. "
              f"First: {failed[0][0]}: {failed[0][1]}", flush=True)
        return 1
    print(f"Done: {total} frames decrypted.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())