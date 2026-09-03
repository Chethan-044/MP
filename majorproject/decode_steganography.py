import os
import cv2
import numpy as np
import struct

from steganography import (
    MAGIC,
    VERSION,
    HEADER_FORMAT,
    HEADER_SIZE,
    COORD_BYTES
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STEGO_DIR = os.path.join(
    BASE_DIR,
    "encrypted_frames"
)


# ============================================================
# EXTRACT BITS
# ============================================================

def extract_bits(frame, number_of_bits):

    flat = frame.reshape(-1)

    if number_of_bits > len(flat):

        raise ValueError(
            "Requested more bits than frame capacity."
        )

    bits = flat[:number_of_bits] & 1

    return bits.astype(np.uint8)


# ============================================================
# BITS → BYTES
# ============================================================

def bits_to_bytes(bits):

    if len(bits) % 8 != 0:

        raise ValueError(
            "Number of bits must be divisible by 8."
        )

    byte_array = np.packbits(bits)

    return byte_array.tobytes()


# ============================================================
# READ HEADER
# ============================================================

def read_header(frame):

    header_bits = extract_bits(
        frame,
        HEADER_SIZE * 8
    )

    header_bytes = bits_to_bytes(
        header_bits
    )

    (
        magic,
        version,
        frame_id,
        box_count
    ) = struct.unpack(
        HEADER_FORMAT,
        header_bytes
    )

    if magic != MAGIC:

        raise ValueError(
            "Invalid MAGIC. "
            "This frame does not contain valid YOLO steganography data."
        )

    if version != VERSION:

        raise ValueError(
            f"Unsupported payload version: {version}"
        )

    return frame_id, box_count


# ============================================================
# DECODE FRAME
# ============================================================

def decode_frame(frame):

    # --------------------------------------------------------
    # READ HEADER
    # --------------------------------------------------------

    frame_id, box_count = read_header(
        frame
    )


    # --------------------------------------------------------
    # CALCULATE PAYLOAD SIZE
    # --------------------------------------------------------

    # Every box contains:
    #
    # x1
    # y1
    # x2
    # y2
    #
    # Each coordinate = 2 bytes

    coordinates_per_box = 4

    coordinate_bytes = (
        box_count
        * coordinates_per_box
        * COORD_BYTES
    )

    total_payload_bytes = (
        HEADER_SIZE
        + coordinate_bytes
    )


    # --------------------------------------------------------
    # EXTRACT COMPLETE PAYLOAD
    # --------------------------------------------------------

    total_bits = (
        total_payload_bytes
        * 8
    )

    bits = extract_bits(
        frame,
        total_bits
    )

    data = bits_to_bytes(
        bits
    )


    # --------------------------------------------------------
    # PARSE HEADER
    # --------------------------------------------------------

    (
        magic,
        version,
        extracted_frame_id,
        extracted_box_count
    ) = struct.unpack(
        HEADER_FORMAT,
        data[:HEADER_SIZE]
    )


    # --------------------------------------------------------
    # READ COORDINATES
    # --------------------------------------------------------

    boxes = []

    offset = HEADER_SIZE

    for _ in range(
        extracted_box_count
    ):

        x1 = struct.unpack(
            "<H",
            data[
                offset:
                offset + 2
            ]
        )[0]

        offset += 2


        y1 = struct.unpack(
            "<H",
            data[
                offset:
                offset + 2
            ]
        )[0]

        offset += 2


        x2 = struct.unpack(
            "<H",
            data[
                offset:
                offset + 2
            ]
        )[0]

        offset += 2


        y2 = struct.unpack(
            "<H",
            data[
                offset:
                offset + 2
            ]
        )[0]

        offset += 2


        boxes.append(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


    return {
        "frame_id": extracted_frame_id,
        "box_count": extracted_box_count,
        "boxes": boxes
    }


# ============================================================
# PROCESS ALL FRAMES
# ============================================================

files = sorted(
    f
    for f in os.listdir(
        STEGO_DIR
    )
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)


if not files:

    print(
        "No PNG frames found."
    )

    exit()


print(
    f"Found {len(files)} stego frames."
)

print()


# ============================================================
# DECODE
# ============================================================

successful = 0
failed = 0

for filename in files:

    path = os.path.join(
        STEGO_DIR,
        filename
    )

    frame = cv2.imread(
        path
    )

    if frame is None:

        print(
            f"[ERROR] Cannot read {filename}"
        )

        failed += 1

        continue


    try:

        result = decode_frame(
            frame
        )

        print(
            f"{filename}"
        )

        print(
            f"  Frame ID : "
            f"{result['frame_id']}"
        )

        print(
            f"  Box Count: "
            f"{result['box_count']}"
        )


        for i, box in enumerate(
            result["boxes"],
            start=1
        ):

            x1, y1, x2, y2 = box

            print(
                f"  Box {i}: "
                f"Top-Left=({x1},{y1}) "
                f"Bottom-Right=({x2},{y2})"
            )


        print()

        successful += 1


    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print(
    "=" * 60
)

print(
    "DECODING COMPLETE"
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
    "=" * 60
)