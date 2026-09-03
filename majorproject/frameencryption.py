import os
import cv2
import numpy as np
import time

from ultralytics import YOLO

from steganography import encode_frame


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRAMES_DIR = os.path.join(
    BASE_DIR,
    "frames"
)

ENCRYPTED_DIR = os.path.join(
    BASE_DIR,
    "encrypted_frames"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FACE_MODEL = os.path.join(
    MODELS_DIR,
    "yolov8n-face.pt"
)

PLATE_MODEL = os.path.join(
    MODELS_DIR,
    "license-plate-finetune-v1s.pt"
)

os.makedirs(
    ENCRYPTED_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FACE_CONF = 0.20
PLATE_CONF = 0.20

DEVICE = "cpu"


# ============================================================
# CHAOTIC XOR
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
            (
                3.99 * x * (1 - x)
                + 0.01 * y
            ) % 1,

            (
                3.99 * y * (1 - y)
                + 0.01 * x
            ) % 1
        )

        X[i] = x
        Y[i] = y

    C = (X[100:] + Y[100:]) / 2

    B = (
        C >= 0.5
    ).astype(np.uint8)

    B = B.reshape(
        h,
        w
    )

    P = np.argsort(C)

    chaos_cache[key] = B, P

    return B, P


def encrypt_roi(roi):

    B, P = chaos(roi.shape)

    pixels = roi.reshape(
        -1,
        3
    )

    shuffled = pixels[P].reshape(
        roi.shape
    )

    encrypted = np.bitwise_xor(
        shuffled,
        (B * 255)[:, :, None]
    )

    return encrypted


# ============================================================
# LOAD YOLO MODELS
# ============================================================

print(
    "[Pipeline] Loading models..."
)

face_model = YOLO(
    FACE_MODEL
)

plate_model = YOLO(
    PLATE_MODEL
)

print(
    "[Pipeline] Models loaded"
)


# ============================================================
# GET FRAMES
# ============================================================

frame_files = sorted(
    f
    for f in os.listdir(
        FRAMES_DIR
    )
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
)

total_frames = len(
    frame_files
)

print(
    f"[Pipeline] Frames found: "
    f"{total_frames}"
)


# ============================================================
# STATISTICS
# ============================================================

total_faces = 0
total_plates = 0
total_rois = 0

start_time = time.perf_counter()


# ============================================================
# PROCESS FRAMES
# ============================================================

for frame_no, filename in enumerate(
    frame_files
):

    path = os.path.join(
        FRAMES_DIR,
        filename
    )

    frame = cv2.imread(
        path
    )

    if frame is None:
        print(
            f"[WARNING] Could not read "
            f"{filename}"
        )

        continue

    H, W = frame.shape[:2]

    # ========================================================
    # STORE ALL YOLO BOUNDING BOXES
    # ========================================================

    all_boxes = []


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_results = face_model.predict(
        frame,
        conf=FACE_CONF,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    face_count = 0


    # ========================================================
    # ENCRYPT FACE ROI
    # ========================================================

    for result in face_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates inside frame
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                W,
                x2
            )

            y2 = min(
                H,
                y2
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue


            # ------------------------------------------------
            # STORE COORDINATES
            # ------------------------------------------------

            all_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # ENCRYPT ROI
            # ------------------------------------------------

            roi = frame[
                y1:y2,
                x1:x2
            ].copy()

            encrypted = encrypt_roi(
                roi
            )

            frame[
                y1:y2,
                x1:x2
            ] = encrypted


            face_count += 1

            total_faces += 1
            total_rois += 1


    # ========================================================
    # PLATE DETECTION
    # ========================================================

    plate_results = plate_model.predict(
        frame,
        conf=PLATE_CONF,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    plate_count = 0


    # ========================================================
    # ENCRYPT PLATE ROI
    # ========================================================

    for result in plate_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates inside frame
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                W,
                x2
            )

            y2 = min(
                H,
                y2
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue


            # ------------------------------------------------
            # STORE COORDINATES
            # ------------------------------------------------

            all_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # ENCRYPT ROI
            # ------------------------------------------------

            roi = frame[
                y1:y2,
                x1:x2
            ].copy()

            encrypted = encrypt_roi(
                roi
            )

            frame[
                y1:y2,
                x1:x2
            ] = encrypted


            plate_count += 1

            total_plates += 1
            total_rois += 1


    # ========================================================
    # LSB STEGANOGRAPHY
    # ========================================================

    try:

        stego_frame = encode_frame(
            frame=frame,
            frame_id=frame_no,
            boxes=all_boxes
        )

    except ValueError as e:

        print(
            f"[ERROR] Steganography failed "
            f"for {filename}: {e}"
        )

        continue


    # ========================================================
    # SAVE STEGO FRAME
    # ========================================================

    output_path = os.path.join(
        ENCRYPTED_DIR,
        f"frame_{frame_no:06d}.png"
    )

    cv2.imwrite(
        output_path,
        stego_frame
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        f"[{frame_no + 1}/{total_frames}] "
        f"{filename} | "
        f"Faces: {face_count} | "
        f"Plates: {plate_count} | "
        f"Total Boxes: {len(all_boxes)}"
    )


# ============================================================
# TIME ANALYSIS
# ============================================================

total_time = (
    time.perf_counter()
    - start_time
)

processing_fps = (
    total_frames / total_time
    if total_time > 0
    else 0
)

avg_time = (
    total_time / total_frames
    if total_frames > 0
    else 0
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "              TIME ANALYSIS"
)

print(
    "=" * 60
)

print(
    f"{'Metric':<35}"
    f"{'Value':>15}"
)

print(
    "-" * 60
)

print(
    f"{'Frames processed':<35}"
    f"{total_frames:>15}"
)

print(
    f"{'Total faces encrypted':<35}"
    f"{total_faces:>15}"
)

print(
    f"{'Total plates encrypted':<35}"
    f"{total_plates:>15}"
)

print(
    f"{'Total ROIs encrypted':<35}"
    f"{total_rois:>15}"
)

print(
    f"{'Total processing time (sec)':<35}"
    f"{total_time:>15.2f}"
)

print(
    f"{'Average time / frame (sec)':<35}"
    f"{avg_time:>15.3f}"
)

print(
    f"{'Processing FPS':<35}"
    f"{processing_fps:>15.2f}"
)

print(
    "=" * 60
)

print(
    f"Output folder: "
    f"{ENCRYPTED_DIR}"
)