import cv2
import numpy as np
import struct


# ============================================================
# SETTINGS
# ============================================================

MAGIC = b"YOLOSTEG"
VERSION = 1

# Each coordinate uses 2 bytes
COORD_BYTES = 2

# Header:
#
# MAGIC       = 8 bytes
# VERSION     = 1 byte
# FRAME_ID    = 4 bytes
# BOX_COUNT   = 2 bytes
#
HEADER_FORMAT = "<8sBIH"

HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)


# ============================================================
# CREATE PAYLOAD
# ============================================================

def create_payload(frame_id, boxes):

    box_count = len(boxes)

    if box_count > 65535:
        raise ValueError(
            "Too many bounding boxes."
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    payload = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        frame_id,
        box_count
    )

    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    for x1, y1, x2, y2 in boxes:

        coordinates = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        for value in coordinates:

            if value < 0 or value > 65535:

                raise ValueError(
                    f"Coordinate {value} "
                    f"is outside 16-bit range."
                )

            payload += struct.pack(
                "<H",
                value
            )

    return payload


# ============================================================
# BYTES → BITS
# ============================================================

def bytes_to_bits(data):

    return np.unpackbits(
        np.frombuffer(
            data,
            dtype=np.uint8
        )
    )


# ============================================================
# BITS → BYTES
# ============================================================

def bits_to_bytes(bits):

    if len(bits) % 8 != 0:

        raise ValueError(
            "Number of bits must be divisible by 8."
        )

    return np.packbits(
        bits
    ).tobytes()


# ============================================================
# CAPACITY
# ============================================================

def get_capacity_bits(frame):

    h, w = frame.shape[:2]

    return h * w * 3


# ============================================================
# LSB ENCODING
# ============================================================

def embed_lsb(frame, data):

    bits = bytes_to_bits(
        data
    )

    capacity = get_capacity_bits(
        frame
    )

    if len(bits) > capacity:

        raise ValueError(
            f"Payload too large. "
            f"Required: {len(bits)} bits, "
            f"Available: {capacity} bits."
        )

    stego = frame.copy()

    flat = stego.reshape(-1)

    # Clear LSB
    flat[:len(bits)] = (
        flat[:len(bits)] & 0xFE
    )

    # Insert secret bits
    flat[:len(bits)] |= bits

    return stego


# ============================================================
# ENCODE FRAME
# ============================================================

def encode_frame(
    frame,
    frame_id,
    boxes
):

    payload = create_payload(
        frame_id,
        boxes
    )

    stego_frame = embed_lsb(
        frame,
        payload
    )

    return stego_frame


# ============================================================
# EXTRACT LSB BITS
# ============================================================

def extract_bits(
    frame,
    number_of_bits
):

    flat = frame.reshape(-1)

    if number_of_bits > len(flat):

        raise ValueError(
            "Requested more bits "
            "than frame capacity."
        )

    bits = (
        flat[:number_of_bits] & 1
    )

    return bits.astype(
        np.uint8
    )


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

    # --------------------------------------------------------
    # CHECK MAGIC
    # --------------------------------------------------------

    if magic != MAGIC:

        raise ValueError(
            "Invalid MAGIC. "
            "This frame does not contain "
            "valid YOLO steganography data."
        )

    # --------------------------------------------------------
    # CHECK VERSION
    # --------------------------------------------------------

    if version != VERSION:

        raise ValueError(
            f"Unsupported payload version: "
            f"{version}"
        )

    return (
        frame_id,
        box_count
    )


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

    total_bits = (
        total_payload_bytes
        * 8
    )

    # --------------------------------------------------------
    # EXTRACT PAYLOAD
    # --------------------------------------------------------

    bits = extract_bits(
        frame,
        total_bits
    )

    data = bits_to_bytes(
        bits
    )

    # --------------------------------------------------------
    # READ HEADER AGAIN
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