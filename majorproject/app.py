import os
import sys
import re
import json
import cv2
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BASE = os.path.dirname(os.path.abspath(__file__))

FRAMES = os.path.join(BASE, "frames")
ENCRYPTED = os.path.join(BASE, "encrypted_frames")
DECRYPTED = os.path.join(BASE, "decrypted_frames")

for folder in (FRAMES, ENCRYPTED, DECRYPTED):
    os.makedirs(folder, exist_ok=True)


class VisionGuard(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("VisionGuard - Video Privacy")
        self.geometry("1100x740")
        self.minsize(900, 620)
        self.configure(bg="#F4F7FB")

        self.video = ""
        self.cap = None
        self.playing = False
        self.preview_job = None
        self.is_busy = False
        self.saved_encrypted_video_path = ""
        self.saved_restored_video_path = ""
        self.enc_preview_visible = False
        self.dec_preview_visible = False
        self.process_poll_job = None

        self.configure_styles()
        self.build_layout()
        self.show_page("Input & Frames")

        self.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------
    # THEME
    # --------------------------------------------------------

    def configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background="#F4F7FB"
        )

        style.configure(
            "TLabel",
            background="#F4F7FB",
            foreground="#172B4D",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 21, "bold")
        )

        style.configure(
            "Muted.TLabel",
            foreground="#64748B",
            font=("Segoe UI", 9)
        )

        style.configure(
            "TButton",
            padding=9,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Primary.TButton",
            background="#2457A7",
            foreground="white",
            padding=10,
            font=("Segoe UI", 10, "bold")
        )

        style.map(
            "Primary.TButton",
            background=[("active", "#1D478A")]
        )

        style.configure(
            "TProgressbar",
            troughcolor="#E2E8F0",
            background="#2457A7",
            thickness=12
        )

    # --------------------------------------------------------
    # MAIN LAYOUT
    # --------------------------------------------------------

    def build_layout(self):
        self.sidebar = tk.Frame(
            self,
            bg="white",
            width=220
        )

        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="◉ VisionGuard",
            bg="white",
            fg="#172B4D",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w", padx=18, pady=(25, 30))

        tk.Label(
            self.sidebar,
            text="VIDEO PRIVACY WORKSPACE",
            bg="white",
            fg="#94A3B8",
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=18, pady=(0, 20))

        self.nav_buttons = {}

        for name in (
            "Input & Frames",
            "Detect & Encrypt",
            "Decrypt"
        ):
            button = tk.Button(
                self.sidebar,
                text="   " + name,
                anchor="w",
                relief="flat",
                bd=0,
                padx=12,
                pady=13,
                bg="white",
                fg="#475569",
                activebackground="#E8F0FE",
                command=lambda page=name: self.show_page(page)
            )

            button.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[name] = button

        tk.Frame(
            self.sidebar,
            bg="#E2E8F0",
            height=1
        ).pack(fill="x", padx=18, pady=20)

        tk.Label(
            self.sidebar,
            text="PIPELINE STATUS",
            bg="white",
            fg="#94A3B8",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=18)

        self.pipeline_status = tk.StringVar(
            value="Waiting for video"
        )

        tk.Label(
            self.sidebar,
            textvariable=self.pipeline_status,
            bg="white",
            fg="#64748B",
            wraplength=175,
            justify="left"
        ).pack(anchor="w", padx=18, pady=12)

        self.main = tk.Frame(
            self,
            bg="#F4F7FB"
        )

        self.main.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.topbar = tk.Frame(
            self.main,
            bg="white",
            height=60
        )

        self.topbar.pack(fill="x")
        self.topbar.pack_propagate(False)

        self.page_title = tk.Label(
            self.topbar,
            text="Input & Frames",
            bg="white",
            fg="#172B4D",
            font=("Segoe UI", 14, "bold")
        )

        self.page_title.pack(
            side="left",
            padx=25
        )

        tk.Label(
            self.topbar,
            text="YOLO | Chaotic XOR | LSB Steganography",
            bg="white",
            fg="#64748B",
            font=("Segoe UI", 9)
        ).pack(side="right", padx=20)

        # Scrollable content area
        self.scroll_wrapper = tk.Frame(
            self.main,
            bg="#F4F7FB"
        )

        self.scroll_wrapper.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=22
        )

        self.content_canvas = tk.Canvas(
            self.scroll_wrapper,
            bg="#F4F7FB",
            highlightthickness=0,
            bd=0
        )

        self.content_scrollbar = ttk.Scrollbar(
            self.scroll_wrapper,
            orient="vertical",
            command=self.content_canvas.yview
        )

        self.content = tk.Frame(
            self.content_canvas,
            bg="#F4F7FB"
        )

        self._content_window = self.content_canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw"
        )

        self.content.bind(
            "<Configure>",
            lambda e: self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all")
            )
        )

        self.content_canvas.bind(
            "<Configure>",
            lambda e: self.content_canvas.itemconfig(
                self._content_window,
                width=e.width
            )
        )

        self.content_scrollbar.pack(
            side="right",
            fill="y"
        )

        self.content_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.content_canvas.bind_all(
            "<MouseWheel>",
            self._on_mousewheel
        )

    def show_page(self, page):
        if self.is_busy:
            messagebox.showwarning(
                "Operation running",
                "Wait until the current operation finishes."
            )
            return

        self.stop_preview()
        self.stop_process_poll()
        self.enc_preview_visible = False
        self.dec_preview_visible = False

        for widget in self.content.winfo_children():
            widget.destroy()

        self.content_canvas.yview_moveto(0)

        self.page_title.configure(text=page)

        for name, button in self.nav_buttons.items():
            button.configure(
                bg="#E8F0FE" if name == page else "white",
                fg="#2457A7" if name == page else "#475569"
            )

        if page == "Input & Frames":
            self.input_page()

        elif page == "Detect & Encrypt":
            self.encrypt_page()

        else:
            self.decrypt_page()

    def heading(self, title, description):
        ttk.Label(
            self.content,
            text=title,
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            self.content,
            text=description,
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(5, 18))

    # --------------------------------------------------------
    # PAGE 1: INPUT VIDEO AND EXTRACT FRAMES
    # --------------------------------------------------------

    def input_page(self):
        self.heading(
            "Input video",
            "Select your video, preview it, and extract frames."
        )

        self.preview = tk.Label(
            self.content,
            text="Select a video to preview",
            bg="#111827",
            fg="white",
            font=("Segoe UI", 13)
        )

        self.preview.pack(
            fill="both",
            expand=True,
            pady=(0, 15)
        )

        buttons = ttk.Frame(self.content)
        buttons.pack(fill="x")

        ttk.Button(
            buttons,
            text="Select Video",
            style="Primary.TButton",
            command=self.select_video
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            buttons,
            text="▶ Play",
            command=self.start_preview
        ).pack(side="left", padx=4)

        ttk.Button(
            buttons,
            text="Ⅱ Pause",
            command=self.stop_preview
        ).pack(side="left", padx=4)

        ttk.Button(
            buttons,
            text="Extract Frames",
            style="Primary.TButton",
            command=self.extract_frames
        ).pack(side="left", padx=4)

        self.extract_progress = ttk.Progressbar(
            self.content,
            maximum=100,
            mode="determinate"
        )

        self.extract_progress.pack(
            fill="x",
            pady=(18, 5)
        )

        self.extract_status = tk.StringVar(
            value="Waiting for video"
        )

        ttk.Label(
            self.content,
            textvariable=self.extract_status
        ).pack(anchor="w")

        ttk.Button(
            self.content,
            text="Continue to Detect & Encrypt",
            command=lambda: self.show_page("Detect & Encrypt")
        ).pack(anchor="e", pady=12)

    def clear_previous_outputs(self):
        """Remove all frames, encrypted frames, decrypted frames
        and generated videos from a previous run."""

        for folder in (FRAMES, ENCRYPTED, DECRYPTED):
            if not os.path.isdir(folder):
                continue

            for name in os.listdir(folder):
                fpath = os.path.join(folder, name)

                if os.path.isfile(fpath) and name.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    os.remove(fpath)

        # Remove generated video files and metadata from previous runs.
        for fname in (
            "encrypted_video.mp4",
            "encrypted_video.avi",
            "decrypted_video.mp4",
            "decrypted_video.avi",
            "encryption_metadata.json",
            "video_metadata.json"
        ):
            video_path = os.path.join(BASE, fname)

            if os.path.isfile(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )

        if not path:
            return

        self.stop_preview()

        if self.cap:
            self.cap.release()

        # Clean up all outputs from the previous video.
        self.clear_previous_outputs()

        self.video = path
        self.cap = cv2.VideoCapture(path)

        self.pipeline_status.set(
            "Selected: " + os.path.basename(path)
        )

        self.start_preview()

    def start_preview(self):
        if not self.video:
            messagebox.showwarning(
                "Select video",
                "Select a video first."
            )
            return

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video)

        self.playing = True
        self.render_preview()

    def stop_preview(self):
        self.playing = False

        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except tk.TclError:
                pass

            self.preview_job = None

    def render_preview(self):
        if not self.playing or self.cap is None:
            return

        success, frame = self.cap.read()

        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()

        if success:
            try:
                from PIL import Image, ImageTk

                height, width = frame.shape[:2]

                scale = min(
                    750 / width,
                    410 / height
                )

                frame = cv2.resize(
                    frame,
                    (
                        max(1, int(width * scale)),
                        max(1, int(height * scale))
                    )
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = ImageTk.PhotoImage(
                    Image.fromarray(rgb)
                )

                self.preview.configure(
                    image=image,
                    text=""
                )

                self.preview.image = image

            except ImportError:
                self.preview.configure(
                    text="Install Pillow to preview video."
                )

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 40

        self.preview_job = self.after(
            max(15, delay),
            self.render_preview
        )

    def extract_frames(self):
        if not self.video:
            messagebox.showwarning(
                "Select video",
                "Select a video first."
            )
            return

        self.stop_preview()
        self.set_busy(True)

        self.extract_progress["value"] = 0
        self.extract_status.set("Extracting frames...")

        def worker():
            cap = cv2.VideoCapture(self.video)

            total = max(
                1,
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25.0

            # Remove old frames from the previous video.
            for name in os.listdir(FRAMES):
                path = os.path.join(FRAMES, name)

                if os.path.isfile(path) and name.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    os.remove(path)

            count = 0

            try:
                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    output = os.path.join(
                        FRAMES,
                        f"frame_{count:06d}.png"
                    )

                    if not cv2.imwrite(output, frame):
                        raise IOError(
                            f"Could not save frame {count}"
                        )

                    count += 1

                    percentage = min(
                        100,
                        count * 100 / total
                    )

                    self.after(
                        0,
                        lambda p=percentage, c=count, t=total:
                        self.update_extract_progress(p, c, t)
                    )

            except Exception as exc:
                self.after(
                    0,
                    lambda msg=str(exc): self.operation_error(msg)
                )
                return

            finally:
                cap.release()

            # Preserve the original frame rate.
            metadata = {
                "source_video": os.path.basename(self.video),
                "fps": fps,
                "frame_count": count
            }

            with open(
                os.path.join(BASE, "video_metadata.json"),
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=4)

            self.after(
                0,
                lambda: self.extraction_finished(count)
            )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def update_extract_progress(self, percentage, current, total):
        self.extract_progress["value"] = percentage

        self.extract_status.set(
            f"Extracting frames: {current}/{total} "
            f"({percentage:.0f}%)"
        )

    def extraction_finished(self, count):
        self.set_busy(False)

        self.extract_progress["value"] = 100
        self.extract_status.set(
            f"Extraction complete: {count} frames"
        )

        self.pipeline_status.set(
            f"Frames extracted: {count}"
        )

        messagebox.showinfo(
            "Complete",
            f"Extracted {count} frames successfully."
        )

    # --------------------------------------------------------
    # PAGE 2: DETECT AND ENCRYPT
    # --------------------------------------------------------

    def encrypt_page(self):
        self.heading(
            "Detect & Encrypt",
            "YOLO detection, row/column permutation, XOR and steganography."
        )

        form = ttk.Frame(self.content)
        form.pack(fill="x", pady=5)

        ttk.Label(
            form,
            text="Seed"
        ).grid(row=0, column=0, sticky="w", pady=8)

        self.enc_seed = tk.StringVar(value="0.37")

        ttk.Entry(
            form,
            textvariable=self.enc_seed,
            width=24
        ).grid(row=0, column=1, padx=15, pady=8, sticky="w")

        ttk.Label(
            form,
            text="Mod Factor"
        ).grid(row=1, column=0, sticky="w", pady=8)

        self.enc_factor = tk.StringVar(value="5")

        ttk.Entry(
            form,
            textvariable=self.enc_factor,
            width=24
        ).grid(row=1, column=1, padx=15, pady=8, sticky="w")

        self.enc_progress = ttk.Progressbar(
            self.content,
            maximum=100,
            mode="determinate"
        )

        self.enc_progress.pack(
            fill="x",
            pady=(20, 6)
        )

        self.enc_status = tk.StringVar(
            value="Ready to encrypt"
        )

        ttk.Label(
            self.content,
            textvariable=self.enc_status
        ).pack(anchor="w", pady=4)

        ttk.Button(
            self.content,
            text="Detect & Encrypt Frames",
            style="Primary.TButton",
            command=self.encrypt
        ).pack(anchor="w", pady=12)

        ttk.Button(
            self.content,
            text="Create Encrypted Video",
            command=self.create_encrypted_video
        ).pack(anchor="w", pady=4)

        ttk.Button(
            self.content,
            text="▶ Play Encrypted Video",
            command=lambda: self.play_file(
                self.saved_encrypted_video_path or "encrypted_video.mp4"
            )
        ).pack(anchor="w", pady=4)

        # ---- Live process preview ----
        self.enc_show_btn = ttk.Button(
            self.content,
            text="▼ Show Process",
            command=self.toggle_enc_preview
        )
        self.enc_show_btn.pack(anchor="w", pady=(12, 4))

        self.enc_preview_container = tk.Frame(
            self.content,
            bg="#111827",
            height=400
        )

        self.enc_frame_info = tk.StringVar(
            value="Waiting for process to start..."
        )

        tk.Label(
            self.enc_preview_container,
            textvariable=self.enc_frame_info,
            bg="#111827",
            fg="#E2E8F0",
            font=("Segoe UI", 9, "bold"),
            pady=6
        ).pack(side="top", fill="x")

        self.enc_preview_img = tk.Label(
            self.enc_preview_container,
            text="Frame preview will appear here\nwhen encryption starts",
            bg="#111827",
            fg="#64748B",
            font=("Segoe UI", 11)
        )
        self.enc_preview_img.pack(
            expand=True,
            fill="both",
            padx=4,
            pady=(0, 4)
        )

    def create_encrypted_video(self):
        if self.is_busy:
            return

        if not os.path.isdir(ENCRYPTED) or not any(
            f.lower().endswith(".png") for f in os.listdir(ENCRYPTED)
        ):
            messagebox.showwarning(
                "No encrypted frames",
                "Encrypt frames before creating video."
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Encrypted Video",
            defaultextension=".mp4",
            initialfile="encrypted_video.mp4",
            filetypes=[
                ("MP4 Video (*.mp4)", "*.mp4"),
                ("Lossless AVI (*.avi)", "*.avi"),
                ("All Video Files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )

        if not output_path:
            return

        self.saved_encrypted_video_path = output_path
        self.build_video(
            "encrypted_frames",
            output_path,
            "video_encrypt"
        )

    def encrypt(self):
        if self.is_busy:
            return

        if not self.video:
            messagebox.showwarning(
                "No video",
                "Select a video and extract frames first."
            )
            return

        if not any(
            f.lower().endswith((".png", ".jpg", ".jpeg"))
            for f in os.listdir(FRAMES)
        ):
            messagebox.showwarning(
                "No frames",
                "Extract frames before encryption."
            )
            return

        try:
            seed = float(self.enc_seed.get())
            factor = float(self.enc_factor.get())

            if factor <= 0:
                raise ValueError

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Enter a numeric seed and a positive mod factor."
            )
            return

        self.enc_progress["value"] = 0
        self.enc_status.set("Starting encryption...")

        self.start_process_poll(
            ENCRYPTED,
            getattr(self, "enc_preview_img", None),
            getattr(self, "enc_frame_info", None),
        )

        self.run_script(
            "frameencryption.py",
            ["--seed", str(seed), "--factor", str(factor)],
            stage="encrypt",
            success_message="Encryption completed successfully."
        )

    # --------------------------------------------------------
    # PAGE 3: DECRYPT  (reads the lossless PNGs in encrypted_frames)
    # --------------------------------------------------------

    def decrypt_page(self):
        self.heading(
            "Decrypt",
            "Enter the seed and factor to restore the frames in encrypted_frames."
        )

        ready = 0

        if os.path.isdir(ENCRYPTED):
            ready = sum(
                1 for f in os.listdir(ENCRYPTED)
                if f.lower().endswith(".png")
            )

        ttk.Label(
            self.content,
            text=f"Encrypted frames ready: {ready}",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        # ---- Seed and factor ----
        form = ttk.Frame(self.content)
        form.pack(anchor="w", pady=5)

        ttk.Label(
            form,
            text="Seed"
        ).grid(row=0, column=0, sticky="w", pady=8)

        self.dec_seed = tk.StringVar(value="")

        ttk.Entry(
            form,
            textvariable=self.dec_seed,
            width=25
        ).grid(row=0, column=1, padx=15, pady=8)

        ttk.Label(
            form,
            text="Mod Factor"
        ).grid(row=1, column=0, sticky="w", pady=8)

        self.dec_factor = tk.StringVar(value="")

        ttk.Entry(
            form,
            textvariable=self.dec_factor,
            width=25
        ).grid(row=1, column=1, padx=15, pady=8)

        # ---- Decryption progress ----
        self.dec_progress = ttk.Progressbar(
            self.content,
            maximum=100,
            mode="determinate"
        )

        self.dec_progress.pack(
            fill="x",
            pady=(20, 6)
        )

        self.dec_status = tk.StringVar(
            value="Ready to decrypt"
        )

        ttk.Label(
            self.content,
            textvariable=self.dec_status
        ).pack(anchor="w", pady=4)

        # ---- Action buttons ----
        ttk.Button(
            self.content,
            text="Decrypt Frames",
            style="Primary.TButton",
            command=self.decrypt
        ).pack(anchor="w", pady=12)

        ttk.Button(
            self.content,
            text="Create Restored Video",
            command=self.create_restored_video
        ).pack(anchor="w", pady=4)

        ttk.Button(
            self.content,
            text="▶ Play Restored Video",
            command=lambda: self.play_file(
                self.saved_restored_video_path or "decrypted_video.mp4"
            )
        ).pack(anchor="w", pady=4)

        # ---- Live process preview ----
        self.dec_show_btn = ttk.Button(
            self.content,
            text="▼ Show Process",
            command=self.toggle_dec_preview
        )
        self.dec_show_btn.pack(anchor="w", pady=(12, 4))

        self.dec_preview_container = tk.Frame(
            self.content,
            bg="#111827",
            height=400
        )

        self.dec_frame_info = tk.StringVar(
            value="Waiting for process to start..."
        )

        tk.Label(
            self.dec_preview_container,
            textvariable=self.dec_frame_info,
            bg="#111827",
            fg="#E2E8F0",
            font=("Segoe UI", 9, "bold"),
            pady=6
        ).pack(side="top", fill="x")

        self.dec_preview_img = tk.Label(
            self.dec_preview_container,
            text="Frame preview will appear here\nwhen decryption starts",
            bg="#111827",
            fg="#64748B",
            font=("Segoe UI", 11)
        )
        self.dec_preview_img.pack(
            expand=True,
            fill="both",
            padx=4,
            pady=(0, 4)
        )

    def create_restored_video(self):
        if self.is_busy:
            return

        if not os.path.isdir(DECRYPTED) or not any(
            f.lower().endswith(".png") for f in os.listdir(DECRYPTED)
        ):
            messagebox.showwarning(
                "No restored frames",
                "Decrypt frames before creating video."
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Restored Video",
            defaultextension=".mp4",
            initialfile="decrypted_video.mp4",
            filetypes=[
                ("MP4 Video (*.mp4)", "*.mp4"),
                ("Lossless AVI (*.avi)", "*.avi"),
                ("All Video Files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )

        if not output_path:
            return

        self.saved_restored_video_path = output_path
        self.build_video(
            "decrypted_frames",
            output_path,
            "video_decrypt"
        )

    def decrypt(self):
        if self.is_busy:
            return

        try:
            seed = float(self.dec_seed.get())
            factor = float(self.dec_factor.get())

            if factor <= 0:
                raise ValueError

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Enter a numeric seed and a positive mod factor."
            )
            return

        input_dir = ENCRYPTED

        if not os.path.isdir(input_dir) or not any(
            f.lower().endswith(".png")
            for f in os.listdir(input_dir)
        ):
            messagebox.showwarning(
                "No encrypted frames",
                "Run Detect & Encrypt first."
            )
            return

        self.dec_progress["value"] = 0
        self.dec_status.set("Starting decryption...")

        self.start_process_poll(
            DECRYPTED,
            getattr(self, "dec_preview_img", None),
            getattr(self, "dec_frame_info", None),
        )

        self.run_script(
            "decrypt_frames.py",
            [
                "--seed", str(seed),
                "--factor", str(factor),
                "--input", input_dir,
            ],
            stage="decrypt",
            success_message="Decryption completed successfully."
        )

    # --------------------------------------------------------
    # RUN BACKEND SCRIPTS AND PARSE PROGRESS
    # --------------------------------------------------------

    def run_script(
        self,
        filename,
        arguments,
        stage,
        success_message
    ):
        if self.is_busy:
            return

        script = os.path.join(BASE, filename)

        if not os.path.isfile(script):
            messagebox.showerror(
                "Missing script",
                f"Could not find:\n{script}"
            )
            return

        self.set_busy(True)

        def worker():
            process = None

            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", script] + arguments,
                    cwd=BASE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1
                )

                output_lines = []

                for line in process.stdout:
                    line = line.rstrip()

                    if not line:
                        continue

                    output_lines.append(line)
                    output_lines = output_lines[-15:]

                    # Backend must print: [current/total]
                    match = re.search(
                        r"\[(\d+)/(\d+)\]",
                        line
                    )

                    if match:
                        current = int(match.group(1))
                        total = max(1, int(match.group(2)))

                        percentage = min(
                            100,
                            current * 100 / total
                        )

                        self.after(
                            0,
                            lambda p=percentage, c=current, t=total:
                            self.update_backend_progress(
                                stage, p, c, t
                            )
                        )

                process.stdout.close()
                return_code = process.wait()

                if return_code != 0:
                    details = "\n".join(output_lines)

                    self.after(
                        0,
                        lambda msg=details: self.operation_error(
                            f"{filename} failed.\n\n{msg}"
                        )
                    )
                    return

                self.after(
                    0,
                    lambda: self.operation_finished(
                        stage,
                        success_message
                    )
                )

            except Exception as exc:
                if process and process.poll() is None:
                    process.kill()

                self.after(
                    0,
                    lambda msg=str(exc): self.operation_error(msg)
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def update_backend_progress(
        self,
        stage,
        percentage,
        current,
        total
    ):
        if stage == "encrypt":
            self.enc_progress["value"] = percentage
            self.enc_status.set(
                f"Encrypting frames: {current}/{total} "
                f"({percentage:.0f}%)"
            )

        elif stage == "decrypt":
            self.dec_progress["value"] = percentage
            self.dec_status.set(
                f"Decrypting frames: {current}/{total} "
                f"({percentage:.0f}%)"
            )

    def operation_finished(self, stage, message):
        self.set_busy(False)
        self.stop_process_poll()

        if stage == "encrypt":
            self.enc_progress["value"] = 100
            self.enc_status.set("Encryption complete.")

        elif stage == "decrypt":
            self.dec_progress["value"] = 100
            self.dec_status.set("Decryption complete.")

        elif stage == "video_encrypt":
            self.enc_status.set("Encrypted video created.")

        elif stage == "video_decrypt":
            self.dec_status.set("Restored video created.")

        self.pipeline_status.set(message)

        # Short popup: never show the full backend log.
        messagebox.showinfo("Complete", message)

    def operation_error(self, message):
        self.set_busy(False)
        self.stop_process_poll()

        if hasattr(self, "enc_status"):
            self.enc_status.set("Operation failed.")

        if hasattr(self, "dec_status"):
            self.dec_status.set("Operation failed.")

        messagebox.showerror("Operation failed", message)

    def set_busy(self, value):
        self.is_busy = bool(value)

    # --------------------------------------------------------
    # CREATE ENCRYPTED / RESTORED VIDEO
    # --------------------------------------------------------

    def build_video(self, folder, output, stage):
        script = os.path.join(BASE, "frametovideo.py")

        if not os.path.isfile(script):
            messagebox.showerror(
                "Missing script",
                "frametovideo.py was not found."
            )
            return

        display_name = os.path.basename(output)
        self.run_script(
            "frametovideo.py",
            ["--input", folder, "--output", output],
            stage=stage,
            success_message=f"Created {display_name} successfully."
        )

    # --------------------------------------------------------
    # PLAY VIDEO
    # --------------------------------------------------------

    def play_file(self, filename):
        path = filename if os.path.isabs(filename) else os.path.join(BASE, filename)

        if not os.path.isfile(path):
            messagebox.showwarning(
                "Video not found",
                f"Create the video first:\n{path}"
            )
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(path)

            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])

            else:
                subprocess.Popen(["xdg-open", path])

        except Exception as exc:
            messagebox.showerror(
                "Playback error",
                str(exc)
            )

    # --------------------------------------------------------
    # LIVE PROCESS PREVIEW
    # --------------------------------------------------------

    def toggle_enc_preview(self):
        """Toggle the encryption process preview panel."""

        self.enc_preview_visible = not self.enc_preview_visible

        if self.enc_preview_visible:
            self.enc_show_btn.configure(text="▲ Hide Process")
            self.enc_preview_container.pack(
                fill="both",
                expand=True,
                pady=(8, 0)
            )
        else:
            self.enc_show_btn.configure(text="▼ Show Process")
            self.enc_preview_container.pack_forget()

    def toggle_dec_preview(self):
        """Toggle the decryption process preview panel."""

        self.dec_preview_visible = not self.dec_preview_visible

        if self.dec_preview_visible:
            self.dec_show_btn.configure(text="▲ Hide Process")
            self.dec_preview_container.pack(
                fill="both",
                expand=True,
                pady=(8, 0)
            )
        else:
            self.dec_show_btn.configure(text="▼ Show Process")
            self.dec_preview_container.pack_forget()

    def start_process_poll(self, directory, label, info_var):
        """Begin polling a directory for newly written frames."""

        self.stop_process_poll()

        self._poll_dir = directory
        self._poll_label = label
        self._poll_info = info_var
        self._poll_last_file = ""

        self.poll_process_frame()

    def stop_process_poll(self):
        """Stop the frame-polling loop."""

        if self.process_poll_job is not None:
            try:
                self.after_cancel(self.process_poll_job)
            except tk.TclError:
                pass

            self.process_poll_job = None

    def poll_process_frame(self):
        """Load the latest frame from the watched directory."""

        if not hasattr(self, "_poll_dir") or not self._poll_dir:
            return

        try:
            directory = self._poll_dir

            if os.path.isdir(directory):
                files = sorted(
                    f for f in os.listdir(directory)
                    if f.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    )
                )

                if files and files[-1] != self._poll_last_file:
                    latest = files[-1]
                    self._poll_last_file = latest

                    path = os.path.join(directory, latest)
                    frame = cv2.imread(
                        path,
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None and self._poll_label:
                        self.display_frame_in_label(
                            frame,
                            self._poll_label
                        )

                    if self._poll_info:
                        self._poll_info.set(
                            f"Processing: {latest}  "
                            f"({len(files)} frames done)"
                        )

        except Exception:
            pass

        self.process_poll_job = self.after(
            400,
            self.poll_process_frame
        )

    def display_frame_in_label(self, frame, label):
        """Scale a BGR frame and display it in a tk.Label."""

        try:
            from PIL import Image, ImageTk

            height, width = frame.shape[:2]

            scale = min(
                680 / width,
                380 / height
            )

            resized = cv2.resize(
                frame,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale))
                )
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            image = ImageTk.PhotoImage(
                Image.fromarray(rgb)
            )

            label.configure(image=image, text="")
            label.image = image

        except ImportError:
            label.configure(
                text="Install Pillow to preview frames."
            )

    # --------------------------------------------------------
    # CLOSE APPLICATION
    # --------------------------------------------------------

    def close(self):
        self.stop_preview()
        self.stop_process_poll()

        if self.cap:
            self.cap.release()

        self.destroy()

    def _on_mousewheel(self, event):
        """Scroll the content canvas with the mouse wheel."""

        self.content_canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )


if __name__ == "__main__":
    VisionGuard().mainloop()