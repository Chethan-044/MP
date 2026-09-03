"""
Flask Web UI — Video Privacy Pipeline
======================================
Controller layer connecting the browser UI to the existing
video-to-frame, YOLO detection + chaotic XOR encryption,
and frame-to-video processing code.

IMPORTANT: The chaotic XOR parameters, YOLO settings, and
VideoWriter codec are identical to the originals in
vidiotoframe.py, frameencryption, and frametovideo.py.
"""

import os
import cv2
import shutil
import threading
import subprocess
import numpy as np
from flask import (
    Flask, render_template, request,
    jsonify, send_file, send_from_directory
)
from ultralytics import YOLO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB


# ============================================================
# PATHS
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR  = os.path.join(BASE_DIR, "frames")
ENCRYPTED_DIR = os.path.join(BASE_DIR, "encrypted_frames")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for _d in [UPLOAD_DIR, FRAMES_DIR, ENCRYPTED_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)


# ============================================================
# SETTINGS  (same as frameencryption)
# ============================================================

FACE_CONF  = 0.20
PLATE_CONF = 0.20
DEVICE     = "cpu"


# ============================================================
# LOAD YOLO MODELS  (same as frameencryption)
# ============================================================

FACE_MODEL_PATH  = os.path.join(MODELS_DIR, "yolov8n-face.pt")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license-plate-finetune-v1s.pt")

print("[Pipeline] Loading models …")
face_model  = YOLO(FACE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
print("[Pipeline] Models loaded ✓")


# ============================================================
# SHARED STATE
# ============================================================

progress = {
    "stage":      "idle",
    "current":    0,
    "total":      0,
    "percentage": 0,
    "faces":      0,
    "plates":     0,
    "status":     "idle",
    "message":    "",
}

video_info = {
    "fps":      25,
    "filename": "",
    "path":     "",
}

processing_lock = threading.Lock()


# ============================================================
# CHAOTIC XOR  (verbatim from frameencryption)
# ============================================================

chaos_cache = {}


def chaos(shape):

    h, w = shape[:2]
    key = (h, w)

    if key in chaos_cache:
        return chaos_cache[key]

    n = h * w

    x = 0.37
    y = 0.71

    X = np.empty(n + 100)
    Y = np.empty(n + 100)

    for i in range(n + 100):

        x, y = (
            (3.99 * x * (1 - x) + 0.01 * y) % 1,
            (3.99 * y * (1 - y) + 0.01 * x) % 1
        )

        X[i] = x
        Y[i] = y

    C = (X[100:] + Y[100:]) / 2

    B = (C >= 0.5).astype(np.uint8)

    B = B.reshape(h, w)

    P = np.argsort(C)

    chaos_cache[key] = B, P

    return B, P


def encrypt_roi(roi):

    B, P = chaos(roi.shape)

    pixels = roi.reshape(-1, 3)

    # Pixel permutation
    shuffled = pixels[P].reshape(roi.shape)

    # XOR
    encrypted = np.bitwise_xor(
        shuffled,
        (B * 255)[:, :, None]
    )

    return encrypted


# ============================================================
# HELPERS
# ============================================================

def _clear(path):
    """Remove every file inside *path* (not sub-dirs)."""
    if os.path.exists(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                os.remove(fp)


def _reset_progress():
    progress.update({
        "stage": "idle", "current": 0, "total": 0,
        "percentage": 0, "faces": 0, "plates": 0,
        "status": "idle", "message": "",
    })


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ----- upload ------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Wipe previous run
    _clear(FRAMES_DIR)
    _clear(ENCRYPTED_DIR)
    _clear(OUTPUT_DIR)
    _reset_progress()

    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    cap = cv2.VideoCapture(path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_info["fps"]      = fps if fps > 0 else 25
    video_info["filename"] = filename
    video_info["path"]     = path

    return jsonify({
        "filename":     filename,
        "fps":          round(video_info["fps"], 2),
        "total_frames": total_frames,
        "width":        width,
        "height":       height,
    })


# ----- extract ------------------------------------------------

@app.route("/extract", methods=["POST"])
def extract():
    if not video_info["path"]:
        return jsonify({"error": "No video uploaded"}), 400

    if not processing_lock.acquire(blocking=False):
        return jsonify({"error": "A task is already running"}), 409

    def _task():
        try:
            progress["stage"]   = "extraction"
            progress["status"]  = "running"
            progress["current"] = 0
            progress["message"] = "Extracting frames …"

            _clear(FRAMES_DIR)

            # ── same logic as vidiotoframe.py ──────────────
            cap   = cv2.VideoCapture(video_info["path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress["total"] = total

            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    os.path.join(FRAMES_DIR, f"frame_{i:06d}.jpg"),
                    frame,
                )
                i += 1
                progress["current"]    = i
                progress["percentage"] = int(i / total * 100) if total else 0

            cap.release()
            # ── end vidiotoframe.py logic ──────────────────

            progress["total"]      = i
            progress["current"]    = i
            progress["percentage"] = 100
            progress["status"]     = "completed"
            progress["message"]    = f"Frame extraction completed. {i} frames extracted."

        except Exception as exc:
            progress["status"]  = "error"
            progress["message"] = f"Error: {exc}"
        finally:
            processing_lock.release()

    threading.Thread(target=_task, daemon=True).start()
    return jsonify({"status": "started"})


# ----- detect ------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():
    frame_files = sorted(
        f for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        return jsonify({"error": "No frames found. Extract first."}), 400

    if not processing_lock.acquire(blocking=False):
        return jsonify({"error": "A task is already running"}), 409

    def _task():
        try:
            progress["stage"]   = "detection"
            progress["status"]  = "running"
            progress["current"] = 0
            progress["faces"]   = 0
            progress["plates"]  = 0
            progress["message"] = "Detecting & encrypting …"

            _clear(ENCRYPTED_DIR)

            # ── same logic as frameencryption ──────────────
            files = sorted(
                f for f in os.listdir(FRAMES_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            progress["total"] = len(files)

            total_faces  = 0
            total_plates = 0

            for frame_no, filename in enumerate(files):

                path  = os.path.join(FRAMES_DIR, filename)
                frame = cv2.imread(path)
                if frame is None:
                    continue

                H, W = frame.shape[:2]

                # ---- face detection ----
                face_results = face_model.predict(
                    frame,
                    conf=FACE_CONF,
                    imgsz=640,
                    device=DEVICE,
                    verbose=False,
                )

                for result in face_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W, x2)
                        y2 = min(H, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = frame[y1:y2, x1:x2].copy()
                        encrypted = encrypt_roi(roi)
                        frame[y1:y2, x1:x2] = encrypted
                        total_faces += 1

                # ---- plate detection ----
                plate_results = plate_model.predict(
                    frame,
                    conf=PLATE_CONF,
                    imgsz=640,
                    device=DEVICE,
                    verbose=False,
                )

                for result in plate_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W, x2)
                        y2 = min(H, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = frame[y1:y2, x1:x2].copy()
                        encrypted = encrypt_roi(roi)
                        frame[y1:y2, x1:x2] = encrypted
                        total_plates += 1

                # ---- save encrypted frame ----
                cv2.imwrite(
                    os.path.join(ENCRYPTED_DIR, filename),
                    frame,
                )

                progress["current"]    = frame_no + 1
                progress["percentage"] = int((frame_no + 1) / len(files) * 100)
                progress["faces"]      = total_faces
                progress["plates"]     = total_plates
            # ── end frameencryption logic ──────────────────

            progress["status"]     = "completed"
            progress["percentage"] = 100
            progress["message"]    = (
                f"Detection & encryption completed. "
                f"Faces: {total_faces}, Plates: {total_plates}"
            )
        except Exception as exc:
            progress["status"]  = "error"
            progress["message"] = f"Error: {exc}"
        finally:
            processing_lock.release()

    threading.Thread(target=_task, daemon=True).start()
    return jsonify({"status": "started"})


# ----- combine -----------------------------------------------

@app.route("/combine", methods=["POST"])
def combine():
    enc = sorted(
        f for f in os.listdir(ENCRYPTED_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not enc:
        return jsonify({"error": "No encrypted frames found."}), 400

    if not processing_lock.acquire(blocking=False):
        return jsonify({"error": "A task is already running"}), 409

    def _task():
        try:
            progress["stage"]   = "combining"
            progress["status"]  = "running"
            progress["current"] = 0
            progress["message"] = "Combining frames …"

            raw_out   = os.path.join(OUTPUT_DIR, "encrypted_video_raw.mp4")
            final_out = os.path.join(OUTPUT_DIR, "encrypted_video.mp4")

            for f in [raw_out, final_out]:
                if os.path.exists(f):
                    os.remove(f)

            # ── same logic as frametovideo.py ──────────────
            folder = ENCRYPTED_DIR

            files = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            if not files:
                progress["status"]  = "error"
                progress["message"] = "No frames found."
                return

            progress["total"] = len(files)

            first = cv2.imread(os.path.join(folder, files[0]))
            height, width = first.shape[:2]

            fps = video_info.get("fps", 25)

            # Try browser-compatible H.264 first, fall back to mp4v
            codecs_to_try = [
                ("avc1", final_out),   # H.264 — plays in all browsers
                ("mp4v", raw_out),     # MPEG-4 — needs ffmpeg re-encode
            ]

            writer = None
            used_raw = False
            for fourcc_str, out_path in codecs_to_try:
                w = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*fourcc_str),
                    fps,
                    (width, height),
                )
                if w.isOpened():
                    writer = w
                    used_raw = (out_path == raw_out)
                    break
                w.release()

            if writer is None:
                progress["status"]  = "error"
                progress["message"] = "No suitable video codec found."
                return

            for i, f in enumerate(files):
                frame = cv2.imread(os.path.join(folder, f))
                if frame is not None:
                    writer.write(frame)
                progress["current"]    = i + 1
                progress["percentage"] = int((i + 1) / len(files) * 100)

            writer.release()
            # ── end frametovideo.py logic ──────────────────

            # If we used mp4v fallback, try ffmpeg re-encode
            if used_raw:
                progress["message"] = "Encoding for browser playback …"
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-i", raw_out,
                            "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            final_out,
                        ],
                        check=True,
                        capture_output=True,
                        timeout=600,
                    )
                    os.remove(raw_out)
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    if os.path.exists(final_out):
                        os.remove(final_out)
                    os.rename(raw_out, final_out)

            progress["status"]     = "completed"
            progress["percentage"] = 100
            progress["message"]    = "Video reconstruction completed."

        except Exception as exc:
            progress["status"]  = "error"
            progress["message"] = f"Error: {exc}"
        finally:
            processing_lock.release()

    threading.Thread(target=_task, daemon=True).start()
    return jsonify({"status": "started"})


# ----- progress / video / download ---------------------------

@app.route("/progress")
def get_progress():
    return jsonify(progress)


@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4")


@app.route("/download")
def download():
    path = os.path.join(OUTPUT_DIR, "encrypted_video.mp4")
    if not os.path.exists(path):
        return jsonify({"error": "No video found"}), 404
    return send_file(path, as_attachment=True, download_name="encrypted_video.mp4")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
