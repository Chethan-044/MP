/* ===================================================================
   VIDEO PRIVACY SYSTEM — SCRIPT
   Sequential pipeline controller with progress polling
   =================================================================== */

(function () {
    "use strict";

    /* ─── DOM refs ──────────────────────────────────────────────── */
    const $ = (id) => document.getElementById(id);

    // Step 1
    const uploadZone     = $("uploadZone");
    const videoInput     = $("videoInput");
    const selectVideoBtn = $("selectVideoBtn");
    const fileInfo       = $("fileInfo");
    const fileName       = $("fileName");
    const totalFrames    = $("totalFrames");
    const videoFps       = $("videoFps");
    const videoRes       = $("videoRes");
    const extractBtn     = $("extractBtn");
    const extractProgress = $("extractProgress");
    const extractFill    = $("extractFill");
    const extractPercent = $("extractPercent");
    const extractDetail  = $("extractDetail");
    const extractStatus  = $("extractStatus");
    const extractStatusText = $("extractStatusText");

    // Step 2
    const detectBtn      = $("detectBtn");
    const detectProgress = $("detectProgress");
    const detectFill     = $("detectFill");
    const detectPercent  = $("detectPercent");
    const detectDetail   = $("detectDetail");
    const detectionStats = $("detectionStats");
    const facesCount     = $("facesCount");
    const platesCount    = $("platesCount");
    const detectStatus   = $("detectStatus");
    const detectStatusText = $("detectStatusText");

    // Step 3
    const combineBtn     = $("combineBtn");
    const combineProgress = $("combineProgress");
    const combineFill    = $("combineFill");
    const combinePercent = $("combinePercent");
    const combineDetail  = $("combineDetail");
    const combineStatus  = $("combineStatus");
    const combineStatusText = $("combineStatusText");

    // Step 4
    const videoContainer = $("videoContainer");
    const videoPlayer    = $("videoPlayer");
    const videoSource    = $("videoSource");


    // Header
    const pipelineBadge  = $("pipelineBadge");
    const pipelineStatus = $("pipelineStatus");

    let pollTimer = null;

    // Change Video
    const changeVideoBtn = $("changeVideoBtn");

    /* ─── Reset pipeline to initial state ───────────────────────── */
    function resetPipeline() {
        // Stop any active polling
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }

        // Step 1: show upload zone, hide file info / progress / status
        uploadZone.style.display       = "flex";
        fileInfo.style.display         = "none";
        extractProgress.style.display  = "none";
        extractStatus.style.display    = "none";
        extractFill.style.width        = "0";
        extractFill.classList.remove("active");
        extractBtn.disabled            = true;
        selectVideoBtn.disabled        = false;
        selectVideoBtn.innerHTML       = "Select Video";
        videoInput.value               = "";

        // Step 2
        detectProgress.style.display   = "none";
        detectionStats.style.display   = "none";
        detectStatus.style.display     = "none";
        detectFill.style.width         = "0";
        detectFill.classList.remove("active");
        detectBtn.disabled             = true;
        facesCount.textContent         = "0";
        platesCount.textContent        = "0";

        // Step 3
        combineProgress.style.display  = "none";
        combineStatus.style.display    = "none";
        combineFill.style.width        = "0";
        combineFill.classList.remove("active");
        combineBtn.disabled            = true;

        // Step 4
        videoContainer.style.display   = "none";
        videoPlayer.pause();
        videoSource.src                = "";

        // Lock steps 2-4, reset step 1
        ["step2", "step3", "step4"].forEach(function (id) {
            var el = $(id);
            el.classList.remove("unlocked", "completed");
            el.classList.add("locked");
        });
        var s1 = $("step1");
        s1.classList.remove("completed", "locked");
        s1.classList.add("unlocked");

        setBadge("Pipeline Ready", false);
    }

    if (changeVideoBtn) changeVideoBtn.addEventListener("click", resetPipeline);

    /* ─── Header badge helper ───────────────────────────────────── */
    function setBadge(text, processing) {
        pipelineStatus.textContent = text;
        pipelineBadge.classList.toggle("processing", !!processing);
    }

    /* ─── Toast ─────────────────────────────────────────────────── */
    function showToast(message, type) {
        type = type || "info";
        const container = $("toastContainer");
        const toast = document.createElement("div");
        toast.className = "toast toast-" + type;

        const icons = { success: "✓", error: "✕", info: "ℹ" };
        toast.innerHTML =
            '<span class="toast-icon">' + (icons[type] || "ℹ") + "</span>" +
            '<span class="toast-message">' + message + "</span>";

        container.appendChild(toast);
        requestAnimationFrame(function () { toast.classList.add("show"); });

        setTimeout(function () {
            toast.classList.remove("show");
            toast.classList.add("hide");
            setTimeout(function () { toast.remove(); }, 400);
        }, 4500);
    }

    /* ─── Unlock a step card ────────────────────────────────────── */
    function unlockStep(n) {
        var card = $("step" + n);
        if (card) {
            card.classList.remove("locked");
            card.classList.add("unlocked");
        }
    }

    function completeStep(n) {
        var card = $("step" + n);
        if (card) {
            card.classList.remove("unlocked");
            card.classList.add("completed");
        }
    }

    /* ─── Progress polling ──────────────────────────────────────── */
    function startPolling(stage, opts) {
        if (pollTimer) clearInterval(pollTimer);
        setBadge("Processing …", true);

        pollTimer = setInterval(async function () {
            try {
                var res  = await fetch("/progress");
                var prog = await res.json();

                if (prog.stage !== stage) return;

                opts.fill.style.width = prog.percentage + "%";
                opts.percent.textContent = prog.percentage + " %";
                opts.detail.textContent = prog.current + " / " + prog.total + " frames";

                if (prog.status === "running") {
                    opts.fill.classList.add("active");
                    if (opts.onProgress) opts.onProgress(prog);
                }

                if (prog.status === "completed") {
                    clearInterval(pollTimer);
                    pollTimer = null;
                    opts.fill.classList.remove("active");
                    opts.fill.style.width = "100%";
                    setBadge("Pipeline Ready", false);
                    if (opts.onComplete) opts.onComplete(prog);
                }

                if (prog.status === "error") {
                    clearInterval(pollTimer);
                    pollTimer = null;
                    opts.fill.classList.remove("active");
                    setBadge("Error", false);
                    showToast(prog.message || "An error occurred", "error");
                    if (opts.onError) opts.onError(prog);
                }
            } catch (_) { /* ignore transient fetch errors */ }
        }, 500);
    }

    /* ═══════════════════════════════════════════════════════════════
       STEP 1 — Video Selection & Frame Extraction
       ═══════════════════════════════════════════════════════════════ */

    selectVideoBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        videoInput.click();
    });

    uploadZone.addEventListener("click", function (e) {
        if (e.target.closest("button") || e.target.closest("input")) return;
        videoInput.click();
    });

    uploadZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });
    uploadZone.addEventListener("dragleave", function () {
        uploadZone.classList.remove("drag-over");
    });
    uploadZone.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
    });

    videoInput.addEventListener("change", function () {
        if (videoInput.files.length) uploadVideo(videoInput.files[0]);
    });

    async function uploadVideo(file) {
        var formData = new FormData();
        formData.append("video", file);

        selectVideoBtn.disabled = true;
        selectVideoBtn.innerHTML = "Uploading …";

        try {
            var res  = await fetch("/upload", { method: "POST", body: formData });
            var data = await res.json();

            if (!res.ok) {
                showToast(data.error || "Upload failed", "error");
                selectVideoBtn.disabled = false;
                selectVideoBtn.innerHTML = "Select Video";
                return;
            }

            fileName.textContent    = data.filename;
            totalFrames.textContent = data.total_frames.toLocaleString();
            videoFps.textContent    = data.fps;
            videoRes.textContent    = data.width + " × " + data.height;

            uploadZone.style.display = "none";
            fileInfo.style.display   = "block";
            extractBtn.disabled      = false;

            showToast("Video uploaded — " + data.filename, "success");
        } catch (err) {
            showToast("Upload failed: " + err.message, "error");
            selectVideoBtn.disabled = false;
            selectVideoBtn.innerHTML = "Select Video";
        }
    }

    /* ── Extract frames ─────────────────────────────────────────── */
    extractBtn.addEventListener("click", async function () {
        extractBtn.disabled = true;
        extractProgress.style.display = "flex";
        extractStatus.style.display   = "none";

        try {
            var res  = await fetch("/extract", { method: "POST" });
            var data = await res.json();

            if (!res.ok) {
                showToast(data.error || "Could not start", "error");
                extractBtn.disabled = false;
                extractProgress.style.display = "none";
                return;
            }

            startPolling("extraction", {
                fill:    extractFill,
                percent: extractPercent,
                detail:  extractDetail,
                onComplete: function (prog) {
                    extractProgress.style.display = "none";
                    extractStatus.style.display   = "flex";
                    extractStatusText.textContent  = prog.message;
                    completeStep(1);
                    unlockStep(2);
                    detectBtn.disabled = false;
                    showToast("Frame extraction completed!", "success");
                },
                onError: function () {
                    extractBtn.disabled = false;
                    extractProgress.style.display = "none";
                }
            });
        } catch (err) {
            showToast("Error: " + err.message, "error");
            extractBtn.disabled = false;
            extractProgress.style.display = "none";
        }
    });

    /* ═══════════════════════════════════════════════════════════════
       STEP 2 — Detection & Encryption
       ═══════════════════════════════════════════════════════════════ */
    detectBtn.addEventListener("click", async function () {
        detectBtn.disabled = true;
        detectProgress.style.display = "flex";
        detectionStats.style.display = "grid";
        detectStatus.style.display   = "none";

        try {
            var res  = await fetch("/detect", { method: "POST" });
            var data = await res.json();

            if (!res.ok) {
                showToast(data.error || "Could not start", "error");
                detectBtn.disabled = false;
                detectProgress.style.display = "none";
                detectionStats.style.display = "none";
                return;
            }

            startPolling("detection", {
                fill:    detectFill,
                percent: detectPercent,
                detail:  detectDetail,
                onProgress: function (prog) {
                    facesCount.textContent  = prog.faces;
                    platesCount.textContent = prog.plates;
                },
                onComplete: function (prog) {
                    facesCount.textContent  = prog.faces;
                    platesCount.textContent = prog.plates;
                    detectProgress.style.display = "none";
                    detectStatus.style.display   = "flex";
                    detectStatusText.textContent  = prog.message;
                    completeStep(2);
                    unlockStep(3);
                    combineBtn.disabled = false;
                    showToast("Detection & encryption completed!", "success");
                },
                onError: function () {
                    detectBtn.disabled = false;
                    detectProgress.style.display = "none";
                }
            });
        } catch (err) {
            showToast("Error: " + err.message, "error");
            detectBtn.disabled = false;
            detectProgress.style.display = "none";
        }
    });

    /* ═══════════════════════════════════════════════════════════════
       STEP 3 — Video Reconstruction
       ═══════════════════════════════════════════════════════════════ */
    combineBtn.addEventListener("click", async function () {
        combineBtn.disabled = true;
        combineProgress.style.display = "flex";
        combineStatus.style.display   = "none";

        try {
            var res  = await fetch("/combine", { method: "POST" });
            var data = await res.json();

            if (!res.ok) {
                showToast(data.error || "Could not start", "error");
                combineBtn.disabled = false;
                combineProgress.style.display = "none";
                return;
            }

            startPolling("combining", {
                fill:    combineFill,
                percent: combinePercent,
                detail:  combineDetail,
                onComplete: function (prog) {
                    combineProgress.style.display = "none";
                    combineStatus.style.display   = "flex";
                    combineStatusText.textContent  = prog.message;
                    completeStep(3);
                    unlockStep(4);
                    showVideoOutput(prog.video_filename);
                    showToast("Video reconstruction completed!", "success");
                },
                onError: function () {
                    combineBtn.disabled = false;
                    combineProgress.style.display = "none";
                }
            });
        } catch (err) {
            showToast("Error: " + err.message, "error");
            combineBtn.disabled = false;
            combineProgress.style.display = "none";
        }
    });

    /* ═══════════════════════════════════════════════════════════════
       STEP 4 — Video Output
       ═══════════════════════════════════════════════════════════════ */
    function showVideoOutput(filename) {
        filename = filename || "encrypted_video.mp4";
        videoSource.src = "/video/" + filename + "?" + Date.now();
        if (filename.endsWith(".webm")) {
            videoSource.type = "video/webm";
        } else {
            videoSource.type = "video/mp4";
        }
        videoPlayer.load();
        videoContainer.style.display = "block";
    }


})();
