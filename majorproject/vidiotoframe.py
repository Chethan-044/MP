import cv2, os
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()

video = filedialog.askopenfilename(
    title="Select Video",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

cap = cv2.VideoCapture(video)
os.makedirs("frames", exist_ok=True)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"frames/frame_{i:06d}.jpg", frame)
    i += 1

cap.release()
root.destroy()

print(f"Saved {i} frames")