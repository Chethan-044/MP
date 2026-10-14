"""
steganography.py  --  LSB EMBEDDING (used during encryption).

Hides  frame ID | ROI count | ROI bounding boxes  inside the encrypted PNG frame itself.
No JSON / sidecar files.  The extractor lives in decode_steganography.py.

How it stays safe
  * Metadata is written AFTER the ROIs are encrypted, and ONLY into pixels that are
    outside every ROI box -> encrypted ROI pixels are never modified.
  * Carrier = a contiguous raster-order run of pixels (1 LSB per colour channel) starting
    at pixel k * STRIDE.  The encoder uses the first run that touches no ROI pixel; the
    decoder finds it by scanning start positions for the magic bytes + a valid CRC32.
  * Frames MUST be stored losslessly (PNG / FFV1).  JPEG / lossy video destroys LSBs.

Payload (big-endian)
  header : "TVP1" | version u8 | frame_id u32 | frame_w u16 | frame_h u16 | roi_count u16 | key_check 4B
  per ROI: x u16 | y u16 | w u16 | h u16 | plaintext_crc32 u32
  trailer: crc32 u32 (over everything above)
"""
import struct
import zlib

import numpy as np

MAGIC = b"TVP1"
VERSION = 2
STRIDE = 64
HDR = struct.Struct(">4sBIHHH4s")
ROI_REC = struct.Struct(">HHHHI")
CRC = struct.Struct(">I")
COUNT_OFFSET = 13


class StegoError(ValueError):
    pass


class MetadataNotFound(StegoError):
    pass


def pack_metadata(frame_id, frame_w, frame_h, records, key_check):
    """records: list of (x, y, w, h, plaintext_crc32)."""
    if not 0 <= frame_id < 2 ** 32:
        raise StegoError("frame_id must fit in 32 bits.")
    if frame_w > 0xFFFF or frame_h > 0xFFFF:
        raise StegoError("Frame dimensions exceed 65535 px.")
    if len(records) > 0xFFFF:
        raise StegoError("Too many ROIs (max 65535).")
    if len(key_check) != 4:
        raise StegoError("key_check must be 4 bytes.")
    body = HDR.pack(MAGIC, VERSION, frame_id, frame_w, frame_h, len(records), key_check)
    body += b"".join(ROI_REC.pack(*r) for r in records)
    return body + CRC.pack(zlib.crc32(body))


def parse_metadata(raw):
    if len(raw) < HDR.size + CRC.size:
        raise StegoError("Metadata too short.")
    body, (crc,) = raw[:-4], CRC.unpack(raw[-4:])
    if zlib.crc32(body) != crc:
        raise StegoError("Metadata CRC mismatch (frame corrupted, re-encoded lossily, or tampered).")
    magic, ver, fid, fw, fh, n, kc = HDR.unpack_from(body, 0)
    if magic != MAGIC or ver != VERSION or len(body) != HDR.size + n * ROI_REC.size:
        raise StegoError("Invalid metadata structure (or frame made by an older version).")
    rois = [ROI_REC.unpack_from(body, HDR.size + i * ROI_REC.size) for i in range(n)]
    return {"frame_id": fid, "frame_w": fw, "frame_h": fh,
            "roi_count": n, "key_check": kc, "rois": rois}


def validate_boxes(boxes, frame_w, frame_h):
    for i, b in enumerate(boxes):
        x, y, w, h = (int(v) for v in b[:4])
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > frame_w or y + h > frame_h:
            raise StegoError(f"ROI {i} ({x},{y},{w},{h}) is empty or outside the {frame_w}x{frame_h} frame.")


def embed_metadata(frame, payload, boxes):
    """Return a copy of `frame` with payload hidden in LSBs outside all ROI boxes."""
    h, w = frame.shape[:2]
    frame = np.ascontiguousarray(frame)
    total = h * w
    nbits = len(payload) * 8
    n_px = -(-nbits // 3)
    if n_px > total:
        raise StegoError(f"Payload ({nbits} bits) exceeds frame capacity ({total * 3} bits).")

    roi = np.zeros((h, w), dtype=bool)
    for x, y, bw, bh in boxes:
        roi[y:y + bh, x:x + bw] = True
    prefix = np.concatenate(([0], np.cumsum(roi.reshape(-1), dtype=np.int64)))

    starts = np.arange(0, total - n_px + 1, STRIDE, dtype=np.int64)
    free = starts[(prefix[starts + n_px] - prefix[starts]) == 0]    # runs with zero ROI pixels
    if free.size == 0:
        raise StegoError(
            f"No ROI-free run of {n_px} pixels exists (ROIs cover almost the whole frame or there are "
            "too many). Refusing to overwrite encrypted ROI pixels.")

    s = int(free[0])
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    out = frame.copy()
    flat = out.reshape(-1)
    flat[3 * s:3 * s + nbits] = (flat[3 * s:3 * s + nbits] & 0xFE) | bits
    return out


# ------------------------------------------------------------------ lossless PNG helpers
def load_png(path):
    import cv2
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise StegoError(f"Cannot read image: {path}")
    return img


def save_png(path, frame):
    """Lossless PNG + read-back verification (works with non-ASCII Windows paths)."""
    import cv2
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() != ".png":
        raise StegoError("Frames carrying LSB metadata must be saved as .png")
    ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise StegoError("PNG encoding failed.")
    p.write_bytes(buf.tobytes())
    if not np.array_equal(load_png(p), frame):
        p.unlink(missing_ok=True)
        raise StegoError("PNG read-back differs from the frame written.")