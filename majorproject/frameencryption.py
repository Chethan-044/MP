"""
frameencryption.py  --  YOLO detect (faces + plates) -> chaotic ROI encryption -> LSB metadata.

Called by app.py:   python frameencryption.py --seed 0.37 --factor 5
Reads  frames/*.png      Writes  encrypted_frames/*.png   (lossless, metadata embedded)
Progress lines look like:  [12/300] frame_000011.png  ROIs=2   (app.py parses [cur/total])

Per ROI:  row permutation -> column permutation -> XOR with chaotic mask
(chaotic equations / permutation / mask are the ones from your original encryption code).
This module is also imported by decrypt_frames.py for the shared crypto functions;
ultralytics is imported lazily, so decryption never needs YOLO.
"""
import argparse
import glob
import hashlib
import math
import os
import re
import sys
import zlib

import cv2
import numpy as np

import steganography as stego

BASE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(BASE, "frames")
ENCRYPTED = os.path.join(BASE, "encrypted_frames")

# ---- YOLO weights: default to best.pt for face and license-plate-finetune-v1s.pt for plate
FACE_WEIGHTS = "best.pt"
PLATE_WEIGHTS = "license-plate-finetune-v1s.pt"
CONF = 0.25
IMGSZ = 960
PAD = 0.05            # grow each detected box by 5% so edges are covered


class CryptoError(ValueError):
    pass


# ============================================================ chaotic crypto core
def check_key(seed, division):
    try:
        seed, division = float(seed), float(division)
    except (TypeError, ValueError):
        raise CryptoError("Seed and factor must be numbers.")
    if not math.isfinite(seed) or seed <= 0:
        raise CryptoError("Seed must be a positive finite number.")
    if not math.isfinite(division) or division <= 0:
        raise CryptoError("Factor must be a positive finite number.")
    return seed, division


def key_check(seed, division):
    """4-byte fingerprint stored in the metadata -> wrong seed/factor is detected up-front."""
    return hashlib.sha256(f"KC|{seed!r}|{division!r}".encode()).digest()[:4]


def seed_to_int(seed):
    return int(abs(seed) * 1_000_000) % (2 ** 32)


def _order(seed, n):
    return np.random.default_rng(seed_to_int(seed)).permutation(n)


def chaotic_sequence(h, w, seed):
    n = h * w
    r = 0.9
    x = (abs(seed) * 0.123456789 % 0.98) + 0.01
    y = (abs(seed) * 0.987654321 % 0.98) + 0.01
    values = np.empty(n, dtype=np.float64)
    for i in range(n + 100):
        xn = r * (3 * y + 1) * x * (1 - x)
        yn = r * (3 * xn + 1) * y * (1 - y)
        x, y = xn, yn
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CryptoError("Chaotic equations produced non-finite values. Try another seed.")
        if i >= 100:
            values[i - 100] = (x + y) / 2
    return values


def _xor_mask(h, w, seed):
    binary = (chaotic_sequence(h, w, seed) >= 0.5).astype(np.uint8)
    return (binary * 255).reshape(h, w, 1)


def roi_seeds(seed, division, frame_id, idx, w, h):
    """row = s, column = s/division, chaotic = s/division^2, where s = seed + a deterministic
    tweak from (frame_id, ROI index, ROI width/height).  Decryption regenerates the same."""
    material = f"ROI|{seed!r}|{division!r}|{frame_id}|{idx}|{w}|{h}".encode()
    s = seed + (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10 ** 9) / 1e6
    return s, s / division, s / (division ** 2)


def encrypt_roi(roi, seed, division, frame_id, idx):
    h, w = roi.shape[:2]
    rs, cs, xs = roi_seeds(seed, division, frame_id, idx, w, h)
    out = np.take(roi, _order(rs, h), axis=0)                  # 1. row permutation
    out = np.take(out, _order(cs, w), axis=1)                  # 2. column permutation
    return np.bitwise_xor(out, _xor_mask(h, w, xs))            # 3. XOR chaotic mask


def decrypt_roi(roi, seed, division, frame_id, idx):
    h, w = roi.shape[:2]
    rs, cs, xs = roi_seeds(seed, division, frame_id, idx, w, h)
    out = np.bitwise_xor(roi, _xor_mask(h, w, xs))             # 1. XOR
    out = np.take(out, np.argsort(_order(cs, w)), axis=1)      # 2. inverse column permutation
    return np.take(out, np.argsort(_order(rs, h)), axis=0)     # 3. inverse row permutation


def generate_chaos(height, width, seed, factor, frame_id):
    """Backward compatibility helper."""
    rs, cs, xs = roi_seeds(seed, factor, frame_id, 0, width, height)
    return _order(rs, height), _order(cs, width), _xor_mask(height, width, xs)


# ============================================================ ROI detection
def merge_overlapping(boxes):
    """Union overlapping (x, y, w, h) boxes so ROIs are disjoint."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed, out = False, []
        while boxes:
            x, y, w, h = boxes.pop()
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]
                if x < bx + bw and bx < x + w and y < by + bh and by < y + h:
                    nx, ny = min(x, bx), min(y, by)
                    w, h = max(x + w, bx + bw) - nx, max(y + h, by + bh) - ny
                    x, y = nx, ny
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append([x, y, w, h])
        boxes = out
    return sorted((tuple(b) for b in boxes), key=lambda b: (b[1], b[0]))


def resolve_model_path(name_or_path, candidates):
    if name_or_path:
        for candidate in [name_or_path, os.path.join(BASE, "models", name_or_path), os.path.join(BASE, name_or_path)]:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    for c in candidates:
        for candidate in [os.path.join(BASE, "models", c), os.path.join(BASE, c)]:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return ""


def find_weights():
    pts = sorted(set(glob.glob(os.path.join(BASE, "*.pt")) + glob.glob(os.path.join(BASE, "models", "*.pt"))))
    face = resolve_model_path(FACE_WEIGHTS, ["best.pt", "yolov8n-face.pt", "yolov8-face.pt"])
    if not face:
        face = next((p for p in pts if "face" in os.path.basename(p).lower() or "best" in os.path.basename(p).lower()), "")

    plate = resolve_model_path(PLATE_WEIGHTS, ["license-plate-finetune-v1s.pt", "license_plate_detector.pt", "license-plate-detector.pt"])
    if not plate:
        plate = next((p for p in pts if re.search(r"plate|licen|anpr|lpr", os.path.basename(p).lower())), "")

    return face, plate, pts


def load_models():
    from ultralytics import YOLO                               # lazy: decryption never imports this
    face, plate, pts = find_weights()
    models = []
    if face:
        models.append(("face", YOLO(face)))
        print(f"Loaded face model: {os.path.basename(face)}", flush=True)
    else:
        print("WARNING: no face model found -> faces will NOT be encrypted. Set FACE_WEIGHTS.", flush=True)

    if plate:
        models.append(("plate", YOLO(plate)))
        print(f"Loaded plate model: {os.path.basename(plate)}", flush=True)
    else:
        print("WARNING: no licence-plate model found -> plates will NOT be encrypted. "
              "Set PLATE_WEIGHTS in frameencryption.py. .pt files seen: "
              + (", ".join(os.path.basename(p) for p in pts) or "none"), flush=True)

    if not models:
        raise CryptoError("No YOLO weights found (put .pt files in the project folder or models/).")
    return models


def detect_rois(models, frame):
    fh, fw = frame.shape[:2]
    boxes = []
    for _, model in models:
        res = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        if res.boxes is None:
            continue
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            boxes.append((x1 - px, y1 - py, x2 + px, y2 + py))
    return merge_overlapping(normalize_boxes(boxes, fw, fh, "xyxy"))


def normalize_boxes(boxes, fw, fh, box_format="xywh"):
    """-> integer (x, y, w, h) clamped to the frame; empty boxes dropped."""
    out = []
    for b in boxes:
        a, b2, c, d = (float(v) for v in b[:4])
        x2, y2 = (c, d) if box_format == "xyxy" else (a + c, b2 + d)
        x1, y1 = max(0, math.floor(a)), max(0, math.floor(b2))
        x2, y2 = min(fw, math.ceil(x2)), min(fh, math.ceil(y2))
        if x2 > x1 and y2 > y1:
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return out


# ============================================================ frame encryption
def encrypt_frame(frame, frame_id, boxes, seed, division):
    """boxes: [(x, y, w, h)].  Returns (stego_frame, boxes_used)."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise CryptoError("Frame must be uint8 (H, W, 3).")
    seed, division = check_key(seed, division)
    frame = np.ascontiguousarray(frame)
    fh, fw = frame.shape[:2]
    boxes = [tuple(int(v) for v in b[:4]) for b in boxes]
    stego.validate_boxes(boxes, fw, fh)

    out = frame.copy()
    records = []
    for i, (x, y, w, h) in enumerate(boxes):
        roi = out[y:y + h, x:x + w].copy()
        crc = zlib.crc32(roi.tobytes())                        # plaintext CRC -> decrypt-side check
        out[y:y + h, x:x + w] = encrypt_roi(roi, seed, division, frame_id, i)
        records.append((x, y, w, h, crc))

    payload = stego.pack_metadata(frame_id, fw, fh, records, key_check(seed, division))
    result = stego.embed_metadata(out, payload, boxes)

    # self-check: metadata round-trips, ROI pixels untouched
    from decode_steganography import extract_metadata
    meta = extract_metadata(result)
    if meta["frame_id"] != frame_id or meta["rois"] != records:
        raise CryptoError("Metadata self-check failed after embedding.")
    for x, y, w, h in boxes:
        if not np.array_equal(result[y:y + h, x:x + w], out[y:y + h, x:x + w]):
            raise CryptoError("Embedding modified encrypted ROI pixels.")
    return result, boxes


# ============================================================ CLI (called by app.py)
def frame_id_from_name(path, fallback):
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(nums[-1]) if nums else fallback


def clear_pngs(folder):
    os.makedirs(folder, exist_ok=True)
    for n in os.listdir(folder):
        if n.lower().endswith(".png"):
            os.remove(os.path.join(folder, n))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--factor", required=True)
    ap.add_argument("--input", default=FRAMES)
    ap.add_argument("--output", default=ENCRYPTED)
    a = ap.parse_args(argv)

    try:
        seed, factor = check_key(a.seed, a.factor)
    except CryptoError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    files = sorted(os.path.join(a.input, n) for n in os.listdir(a.input)
                   if n.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        print(f"ERROR: no frames in {a.input}", flush=True)
        return 2

    ids = [frame_id_from_name(f, i) for i, f in enumerate(files)]
    if len(set(ids)) != len(ids):
        print("ERROR: frame numbers in file names are not unique.", flush=True)
        return 2

    try:
        models = load_models()
    except Exception as e:
        print(f"ERROR: cannot load YOLO models: {e}", flush=True)
        return 2

    clear_pngs(a.output)
    failed, total = [], len(files)
    for n, (path, fid) in enumerate(zip(files, ids), 1):
        name = os.path.basename(path)
        try:
            frame = stego.load_png(path)
            boxes = detect_rois(models, frame)
            enc, used = encrypt_frame(frame, fid, boxes, seed, factor)
            stego.save_png(os.path.join(a.output, f"frame_{fid:06d}.png"), enc)
            print(f"[{n}/{total}] {name}  ROIs={len(used)}", flush=True)
        except Exception as e:                                  # never silent
            failed.append((name, str(e)))
            print(f"[{n}/{total}] FAILED {name}: {e}", flush=True)

    if failed:
        print(f"ERROR: {len(failed)} of {total} frames FAILED and were NOT written. "
              f"First: {failed[0][0]}: {failed[0][1]}", flush=True)
        return 1
    print(f"Done: {total} frames encrypted.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())