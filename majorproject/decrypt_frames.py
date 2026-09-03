import os
import cv2
import numpy as np

from steganography import decode_frame


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STEGO_DIR = os.path.join(
    BASE_DIR,
    "encrypted_frames"
)

DECRYPTED_DIR = os.path.join(
    BASE_DIR,
    "decrypted_frames"
)

os.makedirs(
    DECRYPTED_DIR,
    exist_ok=True
)


# ============================================================
# CHAOS CACHE
# ============================================================

chaos_cache = {}


# ============================================================
# CHAOTIC SEQUENCE
# ============================================================

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


# ============================================================
# DECRYPT ROI
# ============================================================

def decrypt_roi(encrypted_roi):

    """
    Reverse the encryption performed by:

        shuffled = pixels[P]

        encrypted = shuffled XOR chaotic_mask

    Decryption:

        1. XOR encrypted data with chaotic mask
        2. Reverse pixel permutation
    """

    B, P = chaos(
        encrypted_roi.shape
    )

    # --------------------------------------------------------
    # STEP 1
    # Reverse XOR
    # --------------------------------------------------------

    shuffled = np.bitwise_xor(
        encrypted_roi,
        (B * 255)[:, :, None]
    )

    # --------------------------------------------------------
    # STEP 2
    # Reverse permutation
    # --------------------------------------------------------

    shuffled_pixels = shuffled.reshape(
        -1,
        3
    )

    original_pixels = np.empty_like(
        shuffled_pixels
    )

    original_pixels[P] = shuffled_pixels

    decrypted = original_pixels.reshape(
        encrypted_roi.shape
    )

    return decrypted


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def decrypt_frame(frame):

    # --------------------------------------------------------
    # GET COORDINATES FROM STEGANOGRAPHY
    # --------------------------------------------------------

    metadata = decode_frame(
        frame
    )

    frame_id = metadata["frame_id"]

    box_count = metadata["box_count"]

    boxes = metadata["boxes"]

    print(
        f"Frame {frame_id} | "
        f"Boxes: {box_count}"
    )

    # --------------------------------------------------------
    # COPY FRAME
    # --------------------------------------------------------

    decrypted_frame = frame.copy()

    H, W = frame.shape[:2]

    # --------------------------------------------------------
    # PROCESS EVERY ROI
    # --------------------------------------------------------

    for box_number, box in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2 = box

        # ----------------------------------------------------
        # SAFETY CLAMP
        # ----------------------------------------------------

        x1 = max(
            0,
            min(W, x1)
        )

        y1 = max(
            0,
            min(H, y1)
        )

        x2 = max(
            0,
            min(W, x2)
        )

        y2 = max(
            0,
            min(H, y2)
        )

        if x2 <= x1 or y2 <= y1:

            print(
                f"  Box {box_number}: "
                f"Invalid coordinates"
            )

            continue

        # ----------------------------------------------------
        # EXTRACT ENCRYPTED ROI
        # ----------------------------------------------------

        encrypted_roi = frame[
            y1:y2,
            x1:x2
        ].copy()

        # ----------------------------------------------------
        # DECRYPT ROI
        # ----------------------------------------------------

        decrypted_roi = decrypt_roi(
            encrypted_roi
        )

        # ----------------------------------------------------
        # PUT DECRYPTED ROI BACK
        # ----------------------------------------------------

        decrypted_frame[
            y1:y2,
            x1:x2
        ] = decrypted_roi

        print(
            f"  Box {box_number}: "
            f"({x1},{y1}) → "
            f"({x2},{y2}) decrypted"
        )

    return (
        decrypted_frame,
        metadata
    )


# ============================================================
# GET INPUT FILES
# ============================================================

frame_files = sorted(
    f
    for f in os.listdir(
        STEGO_DIR
    )
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
)


if not frame_files:

    print(
        "No encrypted/stego frames found."
    )

    exit()


print(
    f"Found {len(frame_files)} frames."
)

print()


# ============================================================
# PROCESS ALL FRAMES
# ============================================================

successful = 0
failed = 0

for filename in frame_files:

    input_path = os.path.join(
        STEGO_DIR,
        filename
    )

    frame = cv2.imread(
        input_path
    )

    if frame is None:

        print(
            f"[ERROR] Cannot read "
            f"{filename}"
        )

        failed += 1

        continue

    try:

        decrypted_frame, metadata = decrypt_frame(
            frame
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_name = (
            f"frame_{metadata['frame_id']:06d}.png"
        )

        output_path = os.path.join(
            DECRYPTED_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            decrypted_frame
        )

        successful += 1

        print(
            f"  Saved: {output_name}"
        )

        print()


    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )

        failed += 1


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "=" * 60
)

print(
    "             DECRYPTION COMPLETE"
)

print(
    "=" * 60
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed    : {failed}"
)

print(
    f"Output    : {DECRYPTED_DIR}"
)

print(
    "=" * 60
)