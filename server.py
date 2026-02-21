"""
server.py — Flask backend for Frame Analyzer
Run with: python server.py
Requires: pip install flask opencv-python numpy
Also requires: ffmpeg (sudo apt install ffmpeg  OR  brew install ffmpeg)
"""

import os
import sys
import subprocess
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory, Response

try:
    import cv2
except Exception:
    print("Error: pip install opencv-python"); sys.exit(1)
try:
    import numpy as np
except Exception:
    print("Error: pip install numpy"); sys.exit(1)

app = Flask(__name__, static_folder=".")

current_output_path = None
RATIO_THRESHOLD = 1.8


def find_ffmpeg():
    # 1. Check system ffmpeg
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        try:
            r = subprocess.run([candidate, "-version"], capture_output=True)
            if r.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue

    # 2. Fall back to imageio-ffmpeg (pip install imageio[ffmpeg])
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except ImportError:
        pass

    return None


FFMPEG = find_ffmpeg()
if not FFMPEG:
    print("=" * 60)
    print("ERROR: ffmpeg not found!")
    print("Fix (no admin needed):  pip install imageio[ffmpeg]")
    print("Then restart server.py")
    print("=" * 60)
    sys.exit(1)

print(f"ffmpeg found: {FFMPEG}")


def compute_mad_threshold(values, k=3.0):
    arr    = np.array(values)
    median = np.median(arr)
    mad    = np.median(np.abs(arr - median))
    return float(median + k * 1.4826 * mad)


def analyze(input_path: str, output_path: str) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    expected_interval = 1.0 / fps

    # Pass 1: collect motion scores
    motion_scores, timestamps = [], []
    prev_gray = None

    while True:
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ret, frame = cap.read()
        if not ret:
            break
        timestamps.append(ts)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_scores.append(float(np.mean(diff)))
        else:
            motion_scores.append(None)

        prev_gray = gray

    cap.release()

    valid_motion = [s for s in motion_scores if s is not None]
    if not valid_motion:
        raise RuntimeError("No frames extracted from video.")

    motion_arr  = np.array(valid_motion)
    motion_mean = float(np.mean(motion_arr))
    motion_std  = float(np.std(motion_arr))

    motion_drop_threshold  = compute_mad_threshold(valid_motion, k=3.0)
    motion_merge_threshold = max(0.1, motion_mean - 1.5 * motion_std)

    # Pass 2: classify
    labels = []
    for i, (timestamp, motion) in enumerate(zip(timestamps, motion_scores)):
        if motion is None:
            labels.append("NORMAL"); continue

        prev_motion    = motion_scores[i - 1] if i > 1 else None
        relative_spike = (
            prev_motion is not None
            and prev_motion > 0
            and motion > prev_motion * RATIO_THRESHOLD
        )
        is_drop = (
            (i > 0 and (timestamp - timestamps[i - 1]) > expected_interval * 1.5)
            or motion > motion_drop_threshold
            or relative_spike
        )
        is_merge = (not is_drop) and (motion < motion_merge_threshold)

        if is_drop:
            labels.append("DROP")
        elif is_merge:
            labels.append("MERGE")
        else:
            labels.append("NORMAL")

    # Pass 3: render — pipe raw BGR frames directly into ffmpeg
    total_frames = len(labels)
    BAR_H      = 8
    BAR_MARGIN = 40
    bar_width  = width // 2
    bar_x0     = width - BAR_MARGIN - bar_width
    bar_y      = BAR_MARGIN

    COLOR = {
        "NORMAL": (0, 200,   0),
        "MERGE":  (0, 220, 220),
        "DROP":   (0,   0, 220),
    }

    xs       = np.linspace(bar_x0, bar_x0 + bar_width, total_frames + 1, dtype=int)
    full_bar = np.zeros((BAR_H, bar_width, 3), dtype=np.uint8)
    for j, lbl in enumerate(labels):
        x1 = xs[j]     - bar_x0
        x2 = xs[j + 1] - bar_x0
        full_bar[:, x1:x2] = COLOR[lbl]

    # ffmpeg: read raw BGR from stdin → browser-safe H.264 MP4
    ffmpeg_cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]

    # Use a temp file for stderr to avoid pipe buffer deadlock
    import tempfile as _tempfile
    stderr_tmp = _tempfile.TemporaryFile()

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=stderr_tmp,
    )

    cap = cv2.VideoCapture(input_path)

    try:
        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            reveal_x = xs[fi + 1] - bar_x0
            frame[bar_y: bar_y + BAR_H, bar_x0: bar_x0 + reveal_x] = full_bar[:, :reveal_x]
            cv2.rectangle(frame, (bar_x0, bar_y),
                          (bar_x0 + bar_width, bar_y + BAR_H), (255, 255, 255), 1)

            lbl = labels[fi]
            col = COLOR[lbl]
            cv2.putText(frame, f"Frame: {fi}  |  {lbl}", (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

            for k, (l, c) in enumerate(COLOR.items()):
                cv2.rectangle(frame, (20, height - 30 - k * 22),
                              (38, height - 12 - k * 22), c, -1)
                cv2.putText(frame, l, (44, height - 14 - k * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()

    ffmpeg_proc.wait()
    stderr_tmp.seek(0)
    stderr_out = stderr_tmp.read().decode(errors="replace")
    stderr_tmp.close()

    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {ffmpeg_proc.returncode}):\n"
            + stderr_out[-2000:]
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("ffmpeg produced no output file.")

    return {
        "labels": labels,
        "counts": {
            "NORMAL": labels.count("NORMAL"),
            "DROP":   labels.count("DROP"),
            "MERGE":  labels.count("MERGE"),
            "TOTAL":  len(labels),
        },
        "thresholds": {
            "motion_drop":  round(motion_drop_threshold, 2),
            "motion_merge": round(motion_merge_threshold, 2),
            "blur":         "N/A",
        },
        "fps":    fps,
        "width":  width,
        "height": height,
    }


@app.route("/")
def index():
    return send_from_directory(".", "video_analyzer_ui.html")


@app.route("/video")
def serve_video():
    global current_output_path
    if not current_output_path or not os.path.exists(current_output_path):
        return "No video available", 404

    file_size    = os.path.getsize(current_output_path)
    range_header = request.headers.get("Range")

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start  = int(byte_range[0])
        end    = int(byte_range[1]) if byte_range[1] else file_size - 1
        end    = min(end, file_size - 1)
        length = end - start + 1

        def stream_chunk():
            with open(current_output_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return Response(stream_chunk(), status=206, headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(length),
            "Content-Type":   "video/mp4",
            "Cache-Control":  "no-cache",
        })

    def stream_full():
        with open(current_output_path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                yield data

    return Response(stream_full(), status=200, headers={
        "Accept-Ranges":  "bytes",
        "Content-Length": str(file_size),
        "Content-Type":   "video/mp4",
        "Cache-Control":  "no-cache",
    })


@app.route("/analyze", methods=["POST"])
def analyze_route():
    global current_output_path

    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400
    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if current_output_path and os.path.exists(current_output_path):
        try:
            os.remove(current_output_path)
        except Exception:
            pass
    current_output_path = None

    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        input_path = tmp_in.name
        video_file.save(input_path)

    tmp_out = tempfile.NamedTemporaryFile(suffix="_analyzed.mp4", delete=False)
    output_path = tmp_out.name
    tmp_out.close()

    try:
        result = analyze(input_path, output_path)
        current_output_path = output_path
        result["video_url"] = "/video"
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(input_path)
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 48)
    print("  Frame Analyzer — Flask Server")
    print("  http://127.0.0.1:5000")
    print("=" * 48)
    app.run(debug=True, host="0.0.0.0", port=5000)