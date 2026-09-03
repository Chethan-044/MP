import cv2
import os

folder = "encrypted_frames"
output = "encrypted_video.mp4"

files = sorted(
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if not files:
    print("No frames found.")
    exit()

first = cv2.imread(os.path.join(folder, files[0]))
height, width = first.shape[:2]

fps = 25  # change to your original video's FPS

writer = cv2.VideoWriter(
    output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for f in files:
    frame = cv2.imread(os.path.join(folder, f))

    if frame is not None:
        writer.write(frame)

writer.release()

print(f"Created: {output}")