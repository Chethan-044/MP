"""
decode_steganography.py  --  LSB EXTRACTION (used during decryption).

Reads frame ID, ROI count and ROI boxes from the encrypted frame itself.
YOLO is NOT needed.  Layout / constants come from steganography.py.
"""
import struct
import sys

import numpy as np

from steganography import (COUNT_OFFSET, HDR, MAGIC, ROI_REC, STRIDE, MetadataNotFound,
                           StegoError, parse_metadata)


def _read(frame, start_px, nbytes):
    flat = frame.reshape(-1)
    nbits = nbytes * 8
    return np.packbits(flat[3 * start_px:3 * start_px + nbits] & 1).tobytes()


def extract_metadata(frame):
    """Return the validated metadata dict.
    Raises MetadataNotFound if nothing is embedded, StegoError if it is corrupt."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise StegoError("Frame must be uint8 (H, W, 3).")
    frame = np.ascontiguousarray(frame)
    flat = frame.reshape(-1)
    total = frame.shape[0] * frame.shape[1]
    head_px = -(-HDR.size * 8 // 3)
    if total < head_px + 1:
        raise StegoError("Frame too small to hold metadata.")

    starts = np.arange(0, total - head_px + 1, STRIDE, dtype=np.int64)
    magic_bits = np.unpackbits(np.frombuffer(MAGIC, dtype=np.uint8))
    cand = np.ones(starts.size, dtype=bool)
    for b in range(32):                                    # vectorised magic scan
        cand &= (flat[3 * starts + b] & 1) == magic_bits[b]

    last_err = None
    for s in starts[cand]:
        s = int(s)
        try:
            n = struct.unpack_from(">H", _read(frame, s, HDR.size), COUNT_OFFSET)[0]
            size = HDR.size + n * ROI_REC.size + 4
            if 3 * s + size * 8 > flat.size:
                continue
            return parse_metadata(_read(frame, s, size))
        except StegoError as e:                            # maybe a false magic -> keep scanning
            last_err = e
    if last_err is not None:
        raise last_err
    raise MetadataNotFound(
        "No embedded metadata found: not an encrypted frame, or it was re-encoded lossily "
        "(JPEG / MP4). Encrypted video must be lossless.")


def decode_frame(frame_or_path):
    """Convenience helper: returns metadata dict with boxes formatted as (x1, y1, x2, y2)."""
    if isinstance(frame_or_path, (str, bytes)):
        from steganography import load_png
        frame = load_png(frame_or_path)
    else:
        frame = frame_or_path
    meta = extract_metadata(frame)
    return {
        "frame_id": meta["frame_id"],
        "box_count": meta["roi_count"],
        "boxes": [
            (x, y, x + w, y + h) for x, y, w, h, _ in meta["rois"]
        ],
        "rois": meta["rois"],
        "key_check": meta["key_check"],
        "frame_w": meta["frame_w"],
        "frame_h": meta["frame_h"],
    }


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    stego_dir = os.path.join(base, "encrypted_frames")
    files = sorted(f for f in os.listdir(stego_dir) if f.lower().endswith(".png")) if os.path.isdir(stego_dir) else []
    if not files:
        print("No PNG frames found in encrypted_frames/.")
        sys.exit(0)
    print(f"Found {len(files)} stego frames.\n")
    success, fail = 0, 0
    for name in files:
        path = os.path.join(stego_dir, name)
        try:
            res = decode_frame(path)
            print(f"{name} -> Frame ID: {res['frame_id']}, Box Count: {res['box_count']}, Boxes: {res['boxes']}")
            success += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            fail += 1
    print(f"\nDone: {success} succeeded, {fail} failed.")