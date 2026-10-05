import streamlit as st
# REMOVE: import streamlit.components.v1 as components
import subprocess
import os
import sys
import shutil
import json
import time
from html import escape
from typing import Optional

import cv2
import base64
import numpy as np
import psutil
from PIL import Image

st.set_page_config(page_title="Media Sorter AI", page_icon="🎥", layout="wide")

st.markdown('''
<style>
    /* 10% global downscale (zoom) for webkit-based browsers */
    html {
        zoom: 0.9;
    }

    header[data-testid="stHeader"] { display: none !important; }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stMain"] {
        overflow: hidden !important;
    }

    /* FIX: the content area right of the sidebar becomes the positioning
       reference (containing block) for position:fixed elements: left: 0 now
       means "left edge of the content area". The sidebar can never cover the
       status bar any more, and the html zoom no longer matters. Both
       containments together are belt and suspenders across browsers. */
    [data-testid="stMain"], section.stMain {
        contain: layout paint;
        transform: translateZ(0);
    }

    /* Hide all global scrollbars completely */
    ::-webkit-scrollbar {
        width: 0px !important;
        height: 0px !important;
        background: transparent !important;
    }

    .stMainBlockContainer {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    .metrics-container {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
        font-size: 13px;
        font-family: monospace;
        color: #c9d1d9;
        background: #161b22;
        padding: 8px 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 5px;
    }

    /* Log window: fills the remaining screen height, with its own scrollbars */
    .term-container {
        height: calc(100vh - 350px) !important;
        min-height: 400px !important;
        max-height: none !important;
        overflow-y: auto !important;
        background-color: #0e1117;
        padding: 12px;
        border-radius: 5px;
        border: 1px solid #262730;
        font-family: monospace;
        font-size: 13px;
        color: #c9d1d9;
        line-height: 1.5;
    }

    /* Fixed status bar pinned to the bottom of the viewport */
        .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        height: 38px;
        box-sizing: border-box;
        background: #161b22;
        border-top: 1px solid #30363d;
        padding: 0 15px;
        line-height: 38px;
        font-family: monospace;
        font-size: 13px;
        color: #c9d1d9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .status-bar .sb-ok { color: #3fb950; font-weight: bold; }
    .status-bar .sb-warn { color: #f85149; font-weight: bold; }

    /* Progress bar docked directly above the fixed status bar */
        [data-testid="stProgress"] {
        position: fixed;
        bottom: 38px;
        left: 0;
        right: 0;
        z-index: 999;
        margin: 0 !important;
        padding: 4px 15px 6px 15px;
        background: #161b22;
        border-top: 1px solid #30363d;
    }
    
    /* Own scrollbars ONLY for the term-container */
    .term-container::-webkit-scrollbar {
        width: 10px !important;
        background: #0e1117 !important;
    }
    .term-container::-webkit-scrollbar-thumb {
        background: #30363d !important;
        border-radius: 5px !important;
    }
    .term-container::-webkit-scrollbar-thumb:hover {
        background: #58a6ff !important;
    }
</style>
''', unsafe_allow_html=True)

# ====
# PATHS (relative to this script so the app works from any working directory)
# ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, ".ai_sorter.pid")
# Option A: log written to /app/logs/ so the host volume mount makes it accessible
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, ".ai_sorter.log")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
SORTER_SCRIPT = os.path.join(BASE_DIR, "videoocr.py")

DEFAULT_CONFIG = {
    "MAX_SECONDS_OCR": 5,
    "MAX_SECONDS_FACE": 5,
    "FACE_RECOGNITION_THRESHOLD": 0.40,
    "MAX_FRAME_WIDTH": 640,
    "FACE_UPSAMPLE": 0,
    "MIN_VIDEO_FACE_FRAMES": 3,
    "FACE_MODEL": "cnn",
    "WEB_BLACKLIST": [],
    "SUPPORTED_EXTENSIONS": ["*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi", "*.jpg", "*.jpeg", "*.png"],
    "STUDIOS": []
}

SESSION_DEFAULTS = {
    "log_lines": [],
    "log_offset": 0,
    "pending_line": "",
    "current_file": "",
    "current_faces": [],
    "current_ocr": "",
    "current_ghost_faces": [],
    "file_count": 0,
    "total": 1,
    "live_second": 0,
    "live_phase": "",
    "saw_finished": False,
    "show_final": False,
    "final_banner": "",
    "preview_frame_b64": ""
}

def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def check_process_running() -> Optional[int]:
    """Returns the PID of the sorter process if it is really running.
    FIX: verifies the command line to protect against PID reuse by unrelated processes."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        _safe_remove(PID_FILE)
        return None
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline()).lower()
        if "videoocr" in cmdline:
            return pid
        # PID was recycled by an unrelated process -> stale PID file
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        pass  # process no longer exists -> stale PID file
    _safe_remove(PID_FILE)
    return None

def kill_process():
    """FIX: terminates the sorter AND all of its child processes (OCR/CUDA workers)."""
    pid = check_process_running()
    if pid:
        try:
            proc = psutil.Process(pid)
            targets = proc.children(recursive=True) + [proc]
            for p in targets:
                try:
                    p.terminate()
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=5)
            for p in alive:
                try:
                    p.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    _safe_remove(PID_FILE)
    # FIX: the log file is intentionally NOT deleted here, so the user can always
    # review the last run even after stopping or completing the process.

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                merged = dict(DEFAULT_CONFIG)
                merged.update(loaded)
                return merged
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    """FIX: merges into the existing config and writes atomically via a temp file."""
    full_config = load_config()
    full_config.update(cfg)
    tmp_path = CONFIG_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(full_config, f, indent=2)
        os.replace(tmp_path, CONFIG_FILE)
    except Exception as e:
        st.error(f"Could not save config: {e}")

def reset_run_state():
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = list(value) if isinstance(value, list) else value

def restore_sorted_files(sorted_dir: str, input_dir: str):
    """FIX: restores files using the manifest written during sorting instead of
    flattening the directory tree, which lost subfolders and overwrote files."""
    manifest_path = os.path.join(sorted_dir, "manifest.json")
    restored, errors = 0, 0

    if os.path.exists(manifest_path):
        manifest = {}
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f).get("moves", {})
        except Exception:
            manifest = {}

        for target_path, original_path in manifest.items():
            if not os.path.exists(target_path):
                continue  # already restored or deleted
            dest = original_path
            if os.path.exists(dest):
                base, ext = os.path.splitext(original_path)
                dest = f"{base}_{int(time.time())}{ext}"  # never overwrite existing files
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(target_path, dest)
                restored += 1
            except Exception as e:
                errors += 1
                st.error(f"Error restoring {target_path}: {e}")
    else:
        # Legacy fallback for folders sorted without a manifest
        for root, dirs, files in os.walk(sorted_dir):
            for file in files:
                if file == "manifest.json":
                    continue
                src = os.path.join(root, file)
                dest = os.path.join(input_dir, file)
                if os.path.exists(dest):
                    base, ext = os.path.splitext(file)
                    dest = os.path.join(input_dir, f"{base}_{int(time.time())}{ext}")
                try:
                    shutil.move(src, dest)
                    restored += 1
                except Exception as e:
                    errors += 1
                    st.error(f"Error restoring {file}: {e}")

    shutil.rmtree(sorted_dir, ignore_errors=True)
    if errors:
        st.warning(f"Undo finished with {errors} error(s). {restored} file(s) restored.")
    else:
        st.success(f"Undo successful! {restored} file(s) restored.")

# Initialize session state (only for keys that do not exist yet)
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = list(value) if isinstance(value, list) else value

active_pid = check_process_running()
is_running = active_pid is not None
app_config = load_config()

# ====
# SIDEBAR: SETTINGS & CONTROLS
# ====
with st.sidebar:
    st.header("⚙️ Settings")
    base_dir = st.text_input("📁 Base Directory:", value="/home/downloads/", disabled=is_running)
    input_dir = None

    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
        subdirs.sort()
        subdirs.insert(0, ".")
        selected_folder = st.selectbox("📂 Target Folder:", options=subdirs, disabled=is_running)
        input_dir = os.path.abspath(os.path.join(base_dir, selected_folder))
    else:
        st.warning("⚠️ Base directory does not exist.")

    st.markdown("---")

    with st.expander("🎛️ AI Parameters", expanded=False):
        max_ocr = st.number_input("MAX_SECONDS_OCR", min_value=1, max_value=30, value=int(app_config.get("MAX_SECONDS_OCR", 5)), disabled=is_running)
        max_face = st.number_input("MAX_SECONDS_FACE", min_value=1, max_value=30, value=int(app_config.get("MAX_SECONDS_FACE", 5)), disabled=is_running)
        face_thresh = st.slider("FACE_RECOGNITION_THRESHOLD", min_value=0.1, max_value=1.0, value=float(app_config.get("FACE_RECOGNITION_THRESHOLD", 0.40)), step=0.01, disabled=is_running)
        max_width = st.number_input("MAX_FRAME_WIDTH (VRAM)", min_value=320, max_value=1920, value=int(app_config.get("MAX_FRAME_WIDTH", 640)), step=160, disabled=is_running)
        face_upsample = st.number_input("FACE_UPSAMPLE (VRAM)", min_value=0, max_value=2, value=int(app_config.get("FACE_UPSAMPLE", 0)), disabled=is_running)

    st.markdown("---")
    disable_web = st.checkbox("🌐 Disable Web Search", value=True, disabled=is_running)
    ocr_engine = st.selectbox("📝 OCR Engine:", options=["paddleocr", "easyocr"], index=0, disabled=is_running)
    auto_scroll = st.checkbox("⬇️ Log Auto-Scrolling", value=True)

    st.markdown("---")
    start_process = st.button("🚀 Start Sorting", type="primary", disabled=is_running, use_container_width=True)
    stop_process = st.button("🛑 Stop Process", disabled=not is_running, use_container_width=True)

    if input_dir and not is_running:
        sorted_dir = os.path.join(input_dir, "sorted_videos")
        if os.path.exists(sorted_dir):
            st.warning("⚠️ 'sorted_videos' folder exists.")
            if st.button("↩️ Undo (Restore Files)", use_container_width=True):
                restore_sorted_files(sorted_dir, input_dir)
                st.rerun()

# ====
# MAIN PAGE: HEADER & METRICS
# ====
col_title, col_metrics = st.columns([1, 2])
with col_title:
    st.title("🎥 Media Sorter AI")
    st.markdown("Watch the AI sort your media in real-time.")

with col_metrics:
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    cuda_status = "Not Available"
    gpu_mem_used = "N/A"

    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8", timeout=2
        ).strip().split('\n')[0].split(',')

        if len(smi_output) >= 3:
            gpu_name = smi_output[0].strip()
            gpu_used = float(smi_output[1].strip()) / 1024
            gpu_total = float(smi_output[2].strip()) / 1024
            cuda_status = f"Active ({gpu_name})"
            gpu_mem_used = f"{gpu_used:.1f}/{gpu_total:.1f} GB"
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                cuda_status = f"Active ({torch.cuda.get_device_name(0)})"
                gpu_mem_used = "Active"
        except Exception:
            pass

    st.markdown(f"""
        <div class="metrics-container">
            <div>🔌 CUDA: <b>{escape(cuda_status)}</b></div>
            <div>🖥️ CPU: <b>{cpu_percent}%</b></div>
            <div>💾 RAM: <b>{ram_used_gb:.1f}/{ram_total_gb:.1f} GB</b></div>
            <div>🎮 VRAM: <b>{escape(str(gpu_mem_used))}</b></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ====
# MAIN PAGE: CONTENT TABS
# ====
tab_prev, tab_term = st.tabs(["👁️ Live Scanner", "🖥️ System Log"])

with tab_prev:
    col_img, col_info, col_ghost = st.columns([5, 3, 2])
    with col_img:
        preview_image_container = st.empty()
    with col_info:
        preview_info_container = st.empty()
    with col_ghost:
        preview_ghost_container = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

with tab_term:
    terminal_container = st.empty()

# Status line lives OUTSIDE the tabs as the LAST element of the page; it is
# rendered as a fixed bar pinned to the bottom of the viewport (both tabs).
progress_bar = st.empty()
status_text_container = st.empty()

# ====
# LOG PROCESSING (incremental reading, HTML-escaped output)
# ====
def _fmt_hms(sec: float) -> str:
    """Format seconds as H:MM:SS for the status bar."""
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def append_log(html_line: str):
    st.session_state.log_lines.append(html_line)
    if len(st.session_state.log_lines) > 50:
        st.session_state.log_lines.pop(0)

def handle_log_line(line: str):
    line = line.strip()
    if not line:
        return
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        # FIX: non-JSON lines are shown escaped (handles partial/corrupted lines)
        append_log(escape(line) + "<br>")
        return

    event = data.get("event")

    if event == "progress":
        st.session_state.file_count = data.get("file_count", 0)
        st.session_state.total = max(1, data.get("total_files", 1))
        st.session_state.current_file = data.get("current_file", "")
        st.session_state.current_faces = []
        # --- ETA sampling: reset on (re)start, keep the last 30 samples ---
        now = time.time()
        count_now = data.get("file_count", 0)
        if ("run_start_ts" not in st.session_state
                or "eta_last_count" not in st.session_state
                or count_now < st.session_state.eta_last_count):
            st.session_state.run_start_ts = now
            st.session_state.eta_samples = []
        st.session_state.eta_last_count = count_now
        if "eta_samples" not in st.session_state:
            st.session_state.eta_samples = []
        st.session_state.eta_samples.append((now, count_now))
        if len(st.session_state.eta_samples) > 30:
            st.session_state.eta_samples.pop(0)
        st.session_state.current_ocr = ""
        st.session_state.current_ghost_faces = []
        st.session_state.live_second = 0
        st.session_state.live_phase = "Opening File..."
        st.session_state.preview_frame_b64 = ""
        append_log(f"<span style='color:#58a6ff;'>➔ ANALYZING [{st.session_state.file_count}/{st.session_state.total}]: "
                   f"{escape(os.path.basename(st.session_state.current_file))}</span><br>")

    elif event == "preview_update":
        st.session_state.live_second = data.get("second", 0)
        st.session_state.live_phase = data.get("phase", "Scanning...")
        if "frame_b64" in data:
            st.session_state.preview_frame_b64 = str(data.get("frame_b64", ""))

    elif event == "ocr_text":
        st.session_state.current_ocr = str(data.get("text", ""))
        sec = data.get("second", 0)
        append_log(f"Second {sec}: Extracting Text ➔<br>"
                   f"<div style='padding-left: 20px; color:#a5d6ff;'>{escape(st.session_state.current_ocr)}</div>")

    elif event == "face_detected":
        name = str(data.get("name", ""))
        if name and name not in st.session_state.current_faces:
            st.session_state.current_faces.append(name)
        sec = data.get("second", 0)
        append_log(f"Second {sec}: Face Detected ➔<br>"
                   f"<div style='padding-left: 20px; color:#3fb950;'>{escape(name)} RECOGNIZED</div>")

    elif event == "ghost_face":
        name = str(data.get("name", ""))
        count = data.get("count", 0)
        ghost_msg = f"⚠️ {name} ({count} frames)"
        if ghost_msg not in st.session_state.current_ghost_faces:
            st.session_state.current_ghost_faces.append(ghost_msg)
        sec = data.get("second", 0)
        append_log(f"Second {sec}: Ignored Face ➔<br>"
                   f"<div style='padding-left: 20px; color:#d29922;'>{escape(name)} (Ghost)</div>")

    elif event == "warning":
        append_log(f"<span style='color:#f85149;'>⚠️ WARNING: {escape(str(data.get('message', '')))}</span><br>")

    elif event == "error":
        append_log(f"<span style='color:#f85149;'>❌ ERROR: {escape(str(data.get('message', '')))}</span><br>")

    elif event == "finished":
        st.session_state.saw_finished = True

    else:
        append_log(escape(str(data.get("message", ""))) + "<br>")

def read_new_log_lines():
    """FIX: reads only NEW bytes since the last poll instead of re-reading the whole
    file, and buffers incomplete trailing lines to avoid JSON decode races."""
    if not os.path.exists(LOG_FILE):
        return
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(st.session_state.log_offset)
            chunk = f.read()
            st.session_state.log_offset = f.tell()
    except OSError:
        return
    if not chunk:
        return

    raw = st.session_state.pending_line + chunk.decode("utf-8", errors="replace")
    lines = raw.split("\n")
    st.session_state.pending_line = lines.pop()  # last element may be incomplete
    for line in lines:
        handle_log_line(line)

def render_dashboard(final_banner: Optional[str] = None):

    # Terminal log
    terminal_container.markdown(
        f"<div class='term-container' id='terminal'>{''.join(st.session_state.log_lines)}</div>",
        unsafe_allow_html=True
    )

    # JavaScript injection for auto-scrolling based on the checkbox
    if auto_scroll:
        st.html(
            """
            <script>
                const doc = window.parent.document;
                const terms = doc.querySelectorAll('.term-container');
                if (terms.length > 0) {
                    const term = terms[terms.length - 1];
                    term.scrollTop = term.scrollHeight;
                }
            </script>
            """,
            unsafe_allow_javascript=True
        )

    # Progress bar + status line
    total = max(1, st.session_state.total)
    progress_value = min(1.0, st.session_state.file_count / total)
    if final_banner and final_banner.startswith("✅"):
        progress_value = 1.0
    progress_bar.progress(progress_value)

    # NOTE: leading newline = multi-line HTML block (same pattern as the
    # metrics-container block), which Streamlit renders reliably.
    if final_banner:
        cls = "sb-ok" if final_banner.startswith("✅") else "sb-warn"
        status_inner = f"<span class='{cls}'>{escape(final_banner)}</span>"
    else:
        status_inner = (f"<b>Status:</b> Analyzing {st.session_state.file_count} / {total} | "
                        f"<b>File:</b> {escape(os.path.basename(st.session_state.current_file))}")
        if st.session_state.live_phase:
            status_inner += (f" | <b>Phase:</b> {escape(st.session_state.live_phase)} "
                             f"@ {st.session_state.live_second}s")
        sb_done = st.session_state.file_count
        sb_pct = sb_done / max(1, total) * 100.0
        sb_elapsed = time.time() - st.session_state.run_start_ts if "run_start_ts" in st.session_state else 0.0
        sb_samples = st.session_state.eta_samples if "eta_samples" in st.session_state else []
        sb_eta = None
        if len(sb_samples) >= 2:
            sb_t0, sb_c0 = sb_samples[0]
            sb_t1, sb_c1 = sb_samples[-1]
            sb_rate = (sb_c1 - sb_c0) / max(sb_t1 - sb_t0, 0.001)
            if sb_rate > 0 and sb_done < total:
                sb_eta = (total - sb_done) / sb_rate
        if sb_eta is None and 0 < sb_done < total:
            sb_eta = sb_elapsed / sb_done * (total - sb_done)
        status_inner += f" | <b>Progress:</b> {sb_pct:.1f}% | <b>Elapsed:</b> {_fmt_hms(sb_elapsed)}"
        if sb_eta is not None:
            status_inner += f" | <b>ETA:</b> ~{_fmt_hms(sb_eta)}"
    status_text_container.markdown(
        f"""
<div class='status-bar'>{status_inner}</div>
""",
        unsafe_allow_html=True,
    )

    # Live preview
    current_file = st.session_state.current_file
    if current_file and os.path.exists(current_file):
        try:
            target_height = 350
            frame_b64 = st.session_state.get("preview_frame_b64", "")
            if frame_b64:
                # The frame comes from the worker itself - the video is NOT decoded
                # here any more, which frees CPU for the analysis process.
                _buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
                _dec = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
                if _dec is not None:
                    _h, _w = _dec.shape[:2]
                    _sc = target_height / float(_h)
                    _dec = cv2.resize(_dec, (int(_w * _sc), target_height))
                    preview_image_container.image(
                        cv2.cvtColor(_dec, cv2.COLOR_BGR2RGB),
                        caption=f"Time: {st.session_state.live_second}s")
            elif current_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = Image.open(current_file)
                w_percent = (target_height / float(img.size[1]))
                h_size = int((float(img.size[0]) * float(w_percent)))
                img = img.resize((h_size, target_height), Image.Resampling.LANCZOS)
                preview_image_container.image(img, caption=os.path.basename(current_file))

            elif current_file.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')):
                cap = cv2.VideoCapture(current_file)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_MSEC, int(st.session_state.live_second * 1000))
                    ret, frame = cap.read()
                    if ret:
                        h, w = frame.shape[:2]
                        scale = target_height / h
                        resized_frame = cv2.resize(frame, (int(w * scale), target_height))
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        preview_image_container.image(rgb_frame, caption=f"Time: {st.session_state.live_second}s")
                cap.release()
        except Exception:
            preview_image_container.warning("Preview not available.")

    # Info panels (all dynamic values escaped to prevent HTML/JS injection)
    info_md = "### 📋 Valid Data\n"
    if st.session_state.current_ocr:
        info_md += f"**📝 OCR:**\n> {escape(st.session_state.current_ocr)}\n\n"
    if st.session_state.current_faces:
        info_md += "**👤 Faces:**\n" + "\n".join([f"- {escape(f)}" for f in st.session_state.current_faces]) + "\n\n"
    preview_info_container.markdown(info_md)

    ghost_md = "### 👻 Ignored\n"
    if st.session_state.current_ghost_faces:
        ghost_md += "\n".join([f"- {escape(g)}" for g in st.session_state.current_ghost_faces])
    preview_ghost_container.markdown(ghost_md)


# ====
# PROCESS EXECUTION
# ====
if stop_process:
    kill_process()
    st.session_state.show_final = True
    st.session_state.final_banner = "🛑 Process stopped by user."
    st.rerun()

if start_process and input_dir:
    if not os.path.exists(SORTER_SCRIPT):
        st.sidebar.error(f"Sorter script not found: {SORTER_SCRIPT}")
    else:
        save_config({
            "MAX_SECONDS_OCR": int(max_ocr),
            "MAX_SECONDS_FACE": int(max_face),
            "FACE_RECOGNITION_THRESHOLD": float(face_thresh),
            "MAX_FRAME_WIDTH": int(max_width),
            "FACE_UPSAMPLE": int(face_upsample)
        })

        reset_run_state()
        _safe_remove(LOG_FILE)

        # FIX: sys.executable uses the same Python environment as Streamlit,
        # cwd=BASE_DIR makes relative paths predictable,
        # start_new_session=True puts the sorter into its own process group.
        cmd = [sys.executable, SORTER_SCRIPT, input_dir, f"--ocr-engine={ocr_engine}"]
        if disable_web:
            cmd.append("--disable-web-search")

        try:
            with open(LOG_FILE, "w") as log_f:
                p = subprocess.Popen(
                    cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True,
                    cwd=BASE_DIR, start_new_session=True
                )
            with open(PID_FILE, "w") as pid_f:
                pid_f.write(str(p.pid))
        except Exception as e:
            st.sidebar.error(f"Failed to start process: {e}")

        st.rerun()
elif start_process and not input_dir:
    st.sidebar.error("Please select a valid base directory and target folder first.")

# ====
# LIVE MONITORING / FINAL VIEW
# ====
if is_running:
    read_new_log_lines()

    # FIX: verify the process still exists - detects crashes and unexpected exits
    still_running = check_process_running() is not None

    if not still_running:
        st.session_state.show_final = True
        st.session_state.final_banner = (
            "✅ Process completed!" if st.session_state.saw_finished
            else "⚠️ Process ended unexpectedly. Check the log for details."
        )

    render_dashboard(final_banner=st.session_state.final_banner if st.session_state.show_final else None)

    if still_running:
        time.sleep(0.5)
        st.rerun()
    # If the process has ended: stop auto-refreshing and keep the final view.

elif st.session_state.show_final and st.session_state.log_lines:
    # Keep the last run visible after completion/stop/crash
    render_dashboard(final_banner=st.session_state.final_banner)
