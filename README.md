# Media Sorter AI

An AI-powered tool that automatically sorts media files (videos and images) into named creator folders using OCR watermark detection, face recognition, and optional web search verification.

---

## Features

* **OCR watermark detection** — reads embedded creator handles from video frames using PaddleOCR (primary) or EasyOCR (fallback)

* **Face recognition** — clusters unknown faces across files and groups them into consistent folders

* **Web search verification** — validates OCR-detected handles via web search to reduce false positives (optional, can be disabled)

* **SQLite result cache** — avoids redundant OCR/web lookups on re-runs; supports resume after interruption

* **Watermark prefix stripping** — cleans third-party repost watermarks embedded in files before using a label as a folder name

* **Streamlit web UI** — browser-based interface to start/stop sorting, monitor progress, and review logs in real time

* **GPU acceleration** — uses NVIDIA GPU via CUDA for faster OCR and face detection (CPU fallback available)

---

## Requirements

* Docker + Docker Compose

* NVIDIA GPU with CUDA support (recommended; CPU works but is significantly slower)

* NVIDIA Container Toolkit installed on the host

---

## Installation

### 1\. Create the config directory and copy files

```bash
sudo mkdir -p /opt/media-sorter/logs
sudo cp config.json /opt/media-sorter/config.json
sudo cp videoocr.py /opt/media-sorter/videoocr.py
sudo cp web_app.py /opt/media-sorter/web_app.py
```

### 2\. Edit `docker-compose.yaml`

Adjust the volume mount to point to your media directory:

```yaml
volumes:
  - /your/media/path:/your/media/path   # ← change this
  - /opt/media-sorter/config.json:/app/config.json
  - /opt/media-sorter/web_app.py:/app/web_app.py
  - /opt/media-sorter/videoocr.py:/app/videoocr.py
  - /opt/media-sorter/logs:/app/logs
  - paddle_cache:/root/.paddleocr
```

### 3\. Build and start

```bash
docker compose up -d --build
```

The web UI is then available at **[http://localhost:8501](http://localhost:8501)**

---

## Updating the scripts (without rebuild)

Since `videoocr.py` and `web_app.py` are mounted directly into the container, you can update them without rebuilding the image:

```bash
sudo cp videoocr.py /opt/media-sorter/videoocr.py
docker compose restart
```

---

## Configuration (`config.json`)

| Key | Default | Description |
| --- | --- | --- |
| `MAX_SECONDS_OCR` | `3` | Max time per frame for OCR analysis |
| `MAX_SECONDS_FACE` | `3` | Max time per frame for face detection |
| `FACE_RECOGNITION_THRESHOLD` | `0.4` | Face match sensitivity (lower = stricter) |
| `MIN_VIDEO_FACE_FRAMES` | `3` | Minimum frames with the same face to confirm identity |
| `FACE_UPSAMPLE` | `1` | Upsampling passes for small-face detection |
| `MAX_FRAME_WIDTH` | `960` | Frame width cap before OCR/face processing |
| `FACE_MODEL` | `"cnn"` | `"cnn"` (GPU, accurate) or `"hog"` (CPU, fast) |
| `WEB_BLACKLIST` | see below | Words that disqualify a web search result as a valid creator name |
| `STUDIOS` | see below | Known studio names — sorted into separate studio folders |
| `SUPPORTED_EXTENSIONS` | `mp4, mov, webm, mkv, avi, m4v, jpg, jpeg, png` | File types to process |

### Example `config.json`

```json
{
  "MAX_SECONDS_OCR": 3,
  "MAX_SECONDS_FACE": 3,
  "FACE_RECOGNITION_THRESHOLD": 0.4,
  "MIN_VIDEO_FACE_FRAMES": 3,
  "FACE_UPSAMPLE": 1,
  "MAX_FRAME_WIDTH": 960,
  "FACE_MODEL": "cnn",
  "WEB_BLACKLIST": [
    "subscribe", "follow", "vip", "preview",
    "model", "girl", "video", "media", "hd", "hq"
  ],
  "STUDIOS": [
    "Studio One", "Studio Two"
  ],
  "SUPPORTED_EXTENSIONS": [
    "*.mp4", "*.mov", "*.webm", "*.mkv", "*.avi",
    "*.m4v", "*.jpg", "*.jpeg", "*.png"
  ]
}
```

---

## How it works

```
Input folder
    │
    ├── OCR scan (PaddleOCR / EasyOCR)
    │       reads watermarks from video frames → extracts creator handle
    │
    ├── Watermark cleaning
    │       strips platform and repost prefixes from OCR text
    │       rejects placeholders (u11862532), junk labels, spaces in handles
    │
    ├── Web search verification (optional)
    │       confirms handle is a real creator page
    │       caches result in SQLite for future runs
    │
    ├── Face recognition (fallback if OCR yields nothing)
    │       detects and clusters faces across files
    │       groups into "Unknown_Face_1", "Unknown_Face_2" … folders
    │
    └── Output
            /target/CreatorHandle/2024-07/filename.mp4
            /target/_Studios/StudioName/filename.mp4
            /target/_Unknown/filename.mp4
```

---

## Web UI

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

* **Base directory** — the root folder containing your media files

* **Target folder** — where sorted files are moved to

* **OCR engine** — PaddleOCR (faster, GPU) or EasyOCR

* **Web search** — enable/disable web search verification

* **Start / Stop** — run the sorter and watch live progress in the terminal output

---

## Utilities

### `cleanup_folders.py`

Removes empty folders left behind after sorting:

```bash
python3 cleanup_folders.py /your/target/folder
```

---

## Folder structure

```
media-sorter-ai/
├── videoocr.py          # Core processor (OCR, face recognition, sorting logic)
├── web_app.py           # Streamlit web UI
├── config.json          # Configuration (edit this)
├── docker-compose.yaml  # Docker service definition
├── Dockerfile           # Container image definition
├── requirements.txt     # Python dependencies
└── cleanup_folders.py   # Utility: remove empty output folders
```

---

## License

MIT — use freely, modify as needed. No warranty.