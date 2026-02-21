# Pragyan_khel

# Frame Analyzer

A web-based video frame analysis tool that detects **dropped frames**, **merged frames**, and **normal frames** in any video file. Built with OpenCV, Flask, and ffmpeg.

![Frame Analyzer](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

---

## Features

- 🎬 Upload any video (MP4, AVI, MOV, MKV)
- 🔍 Detects **DROP**, **MERGE**, and **NORMAL** frames using motion analysis
- 📊 Visual report with frame distribution timeline and bar charts
- 🎥 Annotated output video playable directly in the browser
- ⚡ Uses MAD (Median Absolute Deviation) thresholding for robust detection

---

## Requirements

- Python 3.8+
- ffmpeg (for browser-compatible video output)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/frame-analyzer.git
cd frame-analyzer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

**Windows (no admin needed):**
```bash
pip install imageio[ffmpeg]
```
> `server.py` will automatically detect and use imageio's bundled ffmpeg.

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

---

## Usage

### Start the server

```bash
python server.py
```

Then open your browser at **http://127.0.0.1:5000**

### How it works

1. Drop or select a video file in the interface
2. Click **Run Analysis**
3. The server processes the video in three passes:
   - **Pass 1** — Extracts motion scores between consecutive frames
   - **Pass 2** — Classifies each frame as NORMAL / DROP / MERGE using MAD thresholding
   - **Pass 3** — Renders an annotated output video via ffmpeg (H.264, browser-compatible)
4. Results are displayed with an annotated video player and detailed report

---

## Frame Classification

| Label | Color | Meaning |
|-------|-------|---------|
| `NORMAL` | 🟢 Green | Frame looks as expected |
| `DROP` | 🔴 Red | Dropped frame — large motion spike or timing gap |
| `MERGE` | 🟡 Yellow | Merged/duplicate frame — very low motion |

---

## Project Structure

```
frame-analyzer/
├── server.py               # Flask backend + analysis logic
├── video_analyzer_ui.html  # Frontend UI (served by Flask)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the UI |
| `POST` | `/analyze` | Accepts video upload, returns JSON results |
| `GET` | `/video` | Streams the analyzed output video (supports HTTP range requests) |

---

## Troubleshooting

**Video not playing in browser?**
Install ffmpeg or run `pip install imageio[ffmpeg]` and restart the server.

**`pip install` failing?**
Make sure you're using Python 3.8+ and try `pip3` instead of `pip`.

**Port already in use?**
Change the port at the bottom of `server.py`:
```python
app.run(debug=True, host="0.0.0.0", port=5001)
```

---

## License

MIT
