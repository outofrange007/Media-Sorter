import os
# Aggressively limit PaddlePaddle memory usage to prevent collisions with PyTorch on 4GB GPUs
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"
os.environ["FLAGS_memory_fraction_of_eager_deletion"] = "1.0"

# Optimize PyTorch memory allocation for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable PaddleOCR connectivity checks to speed up startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Suppress annoying FFmpeg C-level warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-1"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import base64
import easyocr
import face_recognition
import sys
import difflib
import numpy as np
import shutil
import re
import argparse
import time
import torch
import gc
import json
import sqlite3
from typing import List, Set, Dict, Any, Tuple, Optional
from ddgs import DDGS

# ====
# HELPER FOR SECURE JSON COMMUNICATION TO STREAMLIT
# ====
def emit_json(event_type: str, message: str, **kwargs):
    payload = {"event": event_type, "message": message}
    payload.update(kwargs)
    print(json.dumps(payload), flush=True)

# ====
# LOAD EXTERNAL CONFIGURATION (falls back to safe defaults if missing or broken)
# ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "MAX_SECONDS_OCR": 5,
    "MAX_SECONDS_FACE": 5,
    "FACE_RECOGNITION_THRESHOLD": 0.40,
    "MIN_VIDEO_FACE_FRAMES": 3,
    "FACE_UPSAMPLE": 0,
    "MAX_FRAME_WIDTH": 640,
    "FACE_MODEL": "cnn",
    "WEB_BLACKLIST": [],
    "SUPPORTED_EXTENSIONS": ["*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi", "*.jpg", "*.jpeg", "*.png"],
    "STUDIOS": []
}

config: Dict[str, Any] = dict(DEFAULT_CONFIG)
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config.update(json.load(f))
    except Exception as e:
        emit_json("warning", f"Could not parse {CONFIG_FILE} ({e}). Using default values.")
else:
    emit_json("warning", f"{CONFIG_FILE} not found. Using default values.")

MAX_SECONDS_OCR = config.get("MAX_SECONDS_OCR", 5)
MAX_SECONDS_FACE = config.get("MAX_SECONDS_FACE", 5)
FACE_RECOGNITION_THRESHOLD = config.get("FACE_RECOGNITION_THRESHOLD", 0.40)
MIN_VIDEO_FACE_FRAMES = config.get("MIN_VIDEO_FACE_FRAMES", 3)
FACE_UPSAMPLE = config.get("FACE_UPSAMPLE", 0)   # Default lowered for 4GB VRAM
MAX_FRAME_WIDTH = config.get("MAX_FRAME_WIDTH", 640)  # Default lowered for 4GB VRAM

FACE_MODEL = config.get("FACE_MODEL", "cnn")
if FACE_MODEL == "cnn" and not torch.cuda.is_available():
    # The dlib CNN face detector requires CUDA; fall back to HOG on CPU-only machines.
    emit_json("warning", "FACE_MODEL 'cnn' requires CUDA. Falling back to 'hog'.")
    FACE_MODEL = "hog"

WEB_BLACKLIST = set(config.get("WEB_BLACKLIST", []))
# FIX: never allow an empty extension list, otherwise no file would ever be processed
SUPPORTED_EXTENSIONS = tuple(
    config.get("SUPPORTED_EXTENSIONS") or DEFAULT_CONFIG["SUPPORTED_EXTENSIONS"]
)
STUDIOS = config.get("STUDIOS", [])
OCR_ENABLE_MKLDNN = bool(config.get("OCR_ENABLE_MKLDNN", True))
OCR_CPU_THREADS = int(config.get("OCR_CPU_THREADS", 0) or 0)
FACE_TRACK_VERIFY_FRAMES = int(config.get("FACE_TRACK_VERIFY_FRAMES", 5))

# ====
# DATABASE MANAGER (SQLite Persistence)
# ====
class DBManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS KnownFaces
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, encoding BLOB)''')
        c.execute('''CREATE TABLE IF NOT EXISTS VideoHistory
                    (face_id INTEGER, file_path TEXT, UNIQUE(face_id, file_path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS WebCache
                    (query TEXT PRIMARY KEY, result TEXT)''')
        self.conn.commit()

    def load_all_faces(self):
        c = self.conn.cursor()
        c.execute("SELECT id, name, encoding FROM KnownFaces")
        return c.fetchall()

    def save_face(self, name: str, encoding_bytes: bytes) -> int:
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO KnownFaces (name, encoding) VALUES (?, ?)", (name, encoding_bytes))
        self.conn.commit()
        c.execute("SELECT id FROM KnownFaces WHERE name=?", (name,))
        res = c.fetchone()
        return res[0] if res else -1

    def delete_face(self, name: str):
        c = self.conn.cursor()
        c.execute("DELETE FROM KnownFaces WHERE name=?", (name,))
        self.conn.commit()

    def update_face_name(self, old_name: str, new_name: str):
        c = self.conn.cursor()
        c.execute("UPDATE KnownFaces SET name=? WHERE name=?", (new_name, old_name))
        self.conn.commit()

    def add_history(self, face_id: int, file_path: str):
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO VideoHistory (face_id, file_path) VALUES (?, ?)", (face_id, file_path))
        self.conn.commit()

    def update_history_paths(self, old_path: str, new_path: str):
        c = self.conn.cursor()
        c.execute("UPDATE VideoHistory SET file_path=? WHERE file_path=?", (new_path, old_path))
        self.conn.commit()

    def get_web_cache(self, query: str) -> Optional[str]:
        c = self.conn.cursor()
        c.execute("SELECT result FROM WebCache WHERE query=?", (query,))
        row = c.fetchone()
        return row[0] if row else None

    def set_web_cache(self, query: str, result: str):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO WebCache (query, result) VALUES (?, ?)", (query, result))
        self.conn.commit()

# ====
# HELPER FUNCTIONS FOR TEXT AND LOGIC
# ====
def is_just_timestamp(text: str) -> bool:
    if not text:
        return True
    clean = re.sub(r'\[Filename\]', '', str(text)).strip()
    return bool(re.fullmatch(r'[\d\s\-_:]+', clean))

def extract_creator(text: str) -> Optional[str]:
    """Extracts creator handles using targeted regex matching."""
    if not text:
        return None
    text = str(text)

    b_match = re.search(r'(?:Download\s+|Download_)?([a-zA-Z0-9_-]{3,})(?:\s+on|_on_).*?Bongacams', text, re.IGNORECASE)
    if b_match:
        return f"Bongacams/{b_match.group(1).strip()}"

    ph_match = re.search(r'([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)\s*pornhub', text, re.IGNORECASE)
    if ph_match:
        user = ph_match.group(1).strip()
        if user.lower() not in ['com', 'www', 'video', 'videos']:
            return f"Pornhub/{user}"

    of_match = re.search(r'(?:OnlyFans|OnlyFars|Onl\s*Fans|hlyFans|OF)(?:\s*[\.\-]?\s*(?:com|om|co|net))?[^\w@]*@?([a-zA-Z0-9_-]{3,})', text, re.IGNORECASE)
    if of_match:
        user = of_match.group(1).strip()
        invalid = ['com', 'chatter', 'byindians_', 'wikipedia', 'geld', 'taiwan', '2026', 'search', 'free', 'leaks', 'profile']
        if user.lower() not in invalid:
            return f"OF/{user}"

    f_match = re.search(r'Fansly(?:\s*[\.\-]?\s*(?:com|om|co|net))?[^\w@]*@?([a-zA-Z0-9_-]{3,})', text, re.IGNORECASE)
    if f_match:
        user = f_match.group(1).strip()
        if user.lower() not in ['creator', 'posting', 'com', '2026', 'hub', 'stock', 'price']:
            return f"Fansly/{user}"

    for s in STUDIOS:
        if s.replace(" ", "").lower() in text.replace(" ", "").lower():
            return f"Studio/{s}"

    return None

def clean_junk(text: str) -> str:
    """Strips known spam phrases that OCR might catch."""
    if not text:
        return ""
    text = str(text).strip()
    junk_phrases = ["OnlyFans Scams", "Allow Flash", "Free RedGIFs", "Fansly Creator Hub", "Die besten Methoden"]
    for phrase in junk_phrases:
        if phrase.lower() in text.lower():
            return ""
    return text

# ====
# LABEL QUALITY: watermark/domain stripping and junk-label rejection
# ====
# OCR reads watermark lines like "onlyfans.com/username" imperfectly: the domain
# gets glued to the handle, characters are cut off at frame edges, or pure
# boilerplate remains. None of that must ever become a folder or person name.
_PLATFORM_WORD_RE = re.compile(
    r'(?:https?\s*:\s*/\s*/\s*\S+|www\s*\.\s*\S+|t\s*\.\s*me\s*/?\S*'
    r'|linktr\s*\.\s*ee\s*/?\S*|linktree\b\S*'
    r'|\b(?:only\s?f(?:a|o)rs?|only\s?fans|onl\s?fans|hlyfans|fansly)\b'
    r'[\s\.\-_:|]*(?:com|om|co|net|to|me)?\b[\s\.\-_:|/\\]*)',
    re.IGNORECASE)

# Domain residue tokens that OCR glues to handles ("OnlyFans.comc ortega00")
_DOMAIN_TOKENS = {"c", "co", "com", "om", "net", "www", "http", "https", "tme", "me", "ee", "to"}

# Generic watermark vocabulary that is never a person/handle name
GENERIC_LABELS = {
    "home", "vip", "hoes", "sex", "sexy", "asian", "dream", "sweet", "spice",
    "spicy", "english", "male", "italy", "vids", "ppvs", "preview", "website",
    "stolen", "fapello", "thothub", "thotaflix", "onlyfanscom", "fanslycom",
    "4lmcom", "4lmt", "almcom", "media", "telegram", "unknown", "unnamed",
    # Common English words that could form false Title Case "real names" from DDG:
    "free", "content", "videos", "video", "hot", "new", "best", "top",
    "girl", "girls", "model", "models", "nude", "naked", "leaks", "leaked",
    # Third-party watermark sites / aggregators and their known companion tokens:
    "of4lm", "risquemega", "baddiesgallery", "viralxxxporn", "stplayer",
    "livecams", "manyvids", "picsart", "bulldoos",
}

# Junk vocabulary from config.json (WEB_BLACKLIST) merged with the built-in
# GENERIC_LABELS, normalized for token/label comparison. STUDIOS stay exempt:
# they are a deliberate whitelist and may overlap the blacklist ('blacked').
WEB_BLACKLIST_NORM = {re.sub(r"[^a-z0-9]", "", w.lower()) for w in WEB_BLACKLIST}
WEB_BLACKLIST_NORM.discard("")
STUDIOS_NORM = {re.sub(r"[^a-z0-9]", "", s.lower()) for s in STUDIOS}
STUDIOS_NORM.discard("")
LABEL_JUNK_NORM = WEB_BLACKLIST_NORM | {re.sub(r"[^a-z0-9]", "", g.lower()) for g in GENERIC_LABELS}
LABEL_JUNK_NORM.discard("")

def is_blacklisted_label(label: str) -> bool:
    '''True if the label consists only of junk vocabulary.

    True on a whole-label match or when EVERY word token is junk
    ('vip hd video', 'sweet dream'). Config studios are exempt
    ('blacked', 'tushy', ...) - see STUDIOS in config.json.'''
    raw = str(label)
    norm = re.sub(r"[^a-z0-9]", "", raw.lower())
    if not norm:
        return True
    if norm in STUDIOS_NORM:
        return False
    if norm in LABEL_JUNK_NORM:
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", raw.lower()) if t]
    if tokens and all(t in LABEL_JUNK_NORM for t in tokens):
        return True
    return False

# Third-party watermark site names that appear as a prefix before the actual handle
# in OCR reads: "OF4LM.COM handle", "TG@OF4LMT handle", "RISQUEMEGA handle",
# "VIRALXXXPORN from handle", etc.
_WATERMARK_PREFIX_RE = re.compile(
    r'^[\s]*(?:tg\s*@?\s*)?'
    r'(?:of4lm|risquemega|baddies_?gallery\S*|viralxxxporn|stplayer|livecams|manyvids|picsart)'
    r'(?:\.com)?(?:\s+from\b)?[\s@._\-|,:]*',
    re.IGNORECASE
)

def strip_platform_prefix(text: str) -> str:
    '''Removes platform/domain boilerplate (onlyfans.com, fansly, t.me, link watermarks)
    and returns the residual handle text, or an empty string if nothing usable remains.'''
    if not text:
        return ""
    text = _WATERMARK_PREFIX_RE.sub("", str(text))
    cleaned = _PLATFORM_WORD_RE.sub(" ", text)
    tokens = []
    for tok in cleaned.split():
        low = tok.lower()
        if len(tok) <= 1 or low in _DOMAIN_TOKENS:
            continue
        # drop short glued domain residue ("comc", "comR", "comel", "comxo.1")
        base = low.split(".")[0]
        if base.startswith(("com", "om", "net")) and len(base) <= 8:
            continue
        if base in ("co", "cor", "con", "corn"):
            continue
        # drop config-blacklisted vocabulary ("vip", "model", "instagram", ...)
        if re.sub(r"[^a-z0-9]", "", low) in LABEL_JUNK_NORM:
            continue
        tokens.append(tok)
    result = " ".join(tokens).strip()
    # Strip glued/residual ".com" or bare "com" OCR tail from the last word
    # e.g. "comatoxzecom" (from "comatozze.com"), "username.com" from watermarks
    result = re.sub(r'\.?com\s*$', '', result, flags=re.IGNORECASE).strip()
    return result

def is_usable_label(label: str) -> bool:
    '''True if a cleaned OCR/filename label is acceptable as a person/folder name.'''
    if not label:
        return False
    low = re.sub(r"\s+", " ", str(label).strip()).lower()
    if len(low) < 5 or len(low) > 40:
        return False
    if not re.search(r"[a-z]", low):
        return False
    if low in GENERIC_LABELS:
        return False
    # config.json blacklist: whole label or all tokens are known junk words
    if is_blacklisted_label(low):
        return False
    # still watermark-ish after stripping? ("stolen from", "visit thotbook_bot")
    if re.search(r"only\s*fans|fansly|thotbook|telegram|stolen|pornhub|bongacams", low):
        return False
    if low.isdigit():
        return False
    # Handles never contain spaces (real names are verified via _REALNAME_RE elsewhere)
    if " " in low:
        return False
    # OnlyFans system placeholder: "u" + 6+ digits (e.g. "u11862532") → never real
    if re.fullmatch(r'u\d{6,}', low):
        return False
    # Labels starting with a digit are not valid handles ("39au…", "-9 it")
    if low[0].isdigit():
        return False
    return True

def extract_date_from_filename(filename: str) -> str:
    match = re.search(r'\d{4}-\d{2}', str(filename))
    return match.group(0) if match else "Unknown_Date"

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Removes invalid characters and enforces a safe length without trailing dots."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned[:max_length].strip().rstrip('. ')  # FIX: no trailing dots/spaces (Windows-safe)
    return cleaned

def extract_words_from_filename(file_path: str) -> str:
    """Fallback: extracts valid keywords directly from the filename if OCR fails."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    clean_name = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß]', ' ', base_name)
    words = []
    for w in clean_name.split():
        if re.fullmatch(r'\d{4}-\d{2}', w):
            words.append(w)
        elif len(w) > 1 and w.lower() not in WEB_BLACKLIST:
            words.append(w)
    return " ".join(words)

def clean_ocr_results(results: List[str]) -> str:
    """Cleans up OCR output while preserving essential handle characters like @, ., -, /"""
    if not results:
        return ""
    valid_words = []
    for text_block in results:
        cleaned_text = re.sub(r'[^\w\s@\.\/\-]', ' ', text_block)
        for word in cleaned_text.split():
            if len(word) > 1 and not word.isnumeric():
                valid_words.append(word)
    return " ".join(valid_words)

def merge_texts(text_list: List[str], threshold: float = 0.90) -> str:
    """Merges highly similar OCR blocks from different frames to avoid duplication."""
    if not text_list:
        return ""
    groups: List[List[str]] = []
    for text in text_list:
        group_found = False
        for group in groups:
            if difflib.SequenceMatcher(None, text, group[0]).ratio() >= threshold:
                group.append(text)
                group_found = True
                break
        if not group_found:
            groups.append([text])
    if not groups:
        return ""
    largest_group = max(groups, key=len)
    return max(set(largest_group), key=largest_group.count)

def calculate_target_seconds(total_duration: float, duration_per_section: int) -> List[int]:
    """Distributes the analysis time evenly across start, middle, and end of the media."""
    total_duration_int = int(total_duration)
    if total_duration_int <= 0:
        return []
    duration = int(duration_per_section)
    seconds_set: Set[int] = set()

    for s in range(0, min(duration, total_duration_int)):
        seconds_set.add(s)
    middle_start = max(0, (total_duration_int // 2) - (duration // 2))
    for s in range(middle_start, min(middle_start + duration, total_duration_int)):
        seconds_set.add(s)
    end_start = max(0, total_duration_int - duration)
    for s in range(end_start, total_duration_int):
        seconds_set.add(s)

    return sorted(list(seconds_set))

def resize_for_ai(image: np.ndarray, max_width: int = MAX_FRAME_WIDTH) -> np.ndarray:
    """Downscales large images/frames to speed up inference without losing critical detail."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    if w > max_width:
        scale = max_width / w
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# ====
# MAIN MEDIA PROCESSOR CLASS
# ====
class VideoProcessor:
    def __init__(self,
                 input_directory: str,
                 db_path: str,
                 target_base_folder: str,
                 max_ocr_seconds: int,
                 max_face_seconds: int,
                 face_threshold: float,
                 enable_web_search: bool = True,
                 ocr_engine: str = "paddleocr"):

        self.input_directory = input_directory
        self.target_base_folder = target_base_folder
        self.max_ocr_seconds = max_ocr_seconds
        self.max_face_seconds = max_face_seconds
        self.face_threshold = face_threshold
        self.enable_web_search = enable_web_search
        self.ocr_engine = ocr_engine.lower()

        emit_json("log", f"Loading AI models ({self.ocr_engine})...")

        if self.ocr_engine == "paddleocr":
            try:
                import logging
                ppocr_logger = logging.getLogger('ppocr')
                ppocr_logger.setLevel(logging.ERROR)
                ppocr_logger.disabled = True
                for handler in ppocr_logger.handlers[:]:
                    ppocr_logger.removeHandler(handler)

                from paddleocr import PaddleOCR
                try:
                    # PaddleOCR >= 3.0 parameter naming
                    import importlib.metadata as _ppmd
                    try:
                        _pp_major = int(_ppmd.version('paddleocr').split('.')[0])
                    except Exception:
                        _pp_major = 2
                    # Run OCR on CPU: Paddle's GPU caching allocator grows up to ~4 GB on the
                    # Quadro P1000 and never releases it, which starves dlib CNN on the same card.
                    if _pp_major >= 3:
                        _ocr_kwargs_cpu = dict(use_textline_orientation=True, lang='en', device='cpu')
                        if OCR_ENABLE_MKLDNN:
                            _ocr_kwargs_cpu["enable_mkldnn"] = True  # MKL-DNN: much faster CPU inference
                        if OCR_CPU_THREADS > 0:
                            _ocr_kwargs_cpu["cpu_threads"] = OCR_CPU_THREADS
                        try:
                            self.reader = PaddleOCR(**_ocr_kwargs_cpu)
                        except TypeError:
                            # older Paddle builds do not know these kwargs -> start without them
                            _ocr_kwargs_cpu.pop("enable_mkldnn", None)
                            _ocr_kwargs_cpu.pop("cpu_threads", None)
                            self.reader = PaddleOCR(**_ocr_kwargs_cpu)
                    else:
                        _ocr_kwargs_cpu = dict(use_textline_orientation=True, lang='en', use_gpu=False)
                        if OCR_ENABLE_MKLDNN:
                            _ocr_kwargs_cpu["enable_mkldnn"] = True  # MKL-DNN: much faster CPU inference
                        if OCR_CPU_THREADS > 0:
                            _ocr_kwargs_cpu["cpu_threads"] = OCR_CPU_THREADS
                        try:
                            self.reader = PaddleOCR(**_ocr_kwargs_cpu)
                        except TypeError:
                            # older Paddle builds do not know these kwargs -> start without them
                            _ocr_kwargs_cpu.pop("enable_mkldnn", None)
                            _ocr_kwargs_cpu.pop("cpu_threads", None)
                            self.reader = PaddleOCR(**_ocr_kwargs_cpu)
                except TypeError:
                    # PaddleOCR 2.x parameter naming
                    self.reader = PaddleOCR(use_angle_cls=True, lang='en')
                emit_json("log", "PaddleOCR pre-warmed successfully.")
            except ImportError as e:
                emit_json("log", f"PaddleOCR missing dependencies: {e}. Falling back to EasyOCR.")
                self.ocr_engine = "easyocr"
            except Exception as e:
                emit_json("log", f"PaddleOCR initialization error: {e}. Falling back to EasyOCR.")
                self.ocr_engine = "easyocr"

        if self.ocr_engine == "easyocr":
            # FIX: only use the GPU if CUDA is actually available
            self.reader = easyocr.Reader(['de', 'en'], gpu=torch.cuda.is_available())
            try:
                _ = self.reader.readtext(np.zeros((100, 100, 3), dtype=np.uint8), detail=0)
                emit_json("log", "EasyOCR model pre-warmed successfully.")
            except Exception as e:
                emit_json("log", f"Warning: Failed to pre-warm EasyOCR model: {e}")

        self.db = DBManager(db_path)
        self.saved_encodings: List[np.ndarray] = []
        self.saved_names: List[str] = []
        self.saved_ids: List[int] = []
        self.final_report: Dict[str, Dict[str, Any]] = {}

        self._load_memory()

    def _load_memory(self) -> None:
        rows = self.db.load_all_faces()
        for row_id, name, encoding_bytes in rows:
            self.saved_ids.append(row_id)
            self.saved_names.append(name)
            self.saved_encodings.append(np.frombuffer(encoding_bytes, dtype=np.float64))
        emit_json("log", f"Memory loaded: {len(self.saved_names)} people already known from DB.")

    def _verify_text_with_web(self, query: str) -> Tuple[str, str]:
        """Verifies OCR text locally (creator regexes) and optionally via web search.
        Returns (folder_name, platform_folder)."""
        if not query or len(query) < 3:
            return query, ""
        if is_just_timestamp(query):
            return "", ""

        cache_key = query.lower()  # FIX: full text as cache key, no truncation collisions

        # Fast path: known creator/platform patterns (no web request needed)
        creator_match = extract_creator(query)
        if creator_match:
            parts = creator_match.split('/', 1)
            pl_name = parts[0]
            handle = parts[1] if len(parts) > 1 else creator_match
            if is_usable_label(strip_platform_prefix(handle)):
                self.db.set_web_cache(cache_key, handle)
                return handle, pl_name
            # junk handle from a truncated watermark read -> fall through

        platform_dir = "OF" if "onlyfans" in query.lower() else "Fansly" if "fansly" in query.lower() else ""

        if is_blacklisted_label(query):
            return "", platform_dir

        # FIX: local platform detection still works when web search is disabled
        if not self.enable_web_search:
            return self._finalize_text_label(query, platform_dir)

        cached = self.db.get_web_cache(cache_key)
        if cached and is_usable_label(cached):
            return cached, platform_dir

        search_queries = [query] if platform_dir else [f"onlyfans {query}", f"fansly {query}"]

        # FIX 2: Page-title suffixes that always indicate a wrong DDG result.
        _WEB_TITLE_JUNK_RE = re.compile(
            r'(\s*[-–|]\s*(Wikipedia|Reddit|YouTube|Twitter|X\.com|Linktree'
            r'|OnlyFans|Fansly|TikTok|Instagram|Patreon|Telegram|ILLEGAL'
            r'|Launchpad|Gallery|Tagged with|VIP cinema|Videos\s+tagged'
            r'|Viralxxxporn|THOTAIFLIX|OFO\s*-|ManyVids)\b.*$'
            r'|\(@[^)]+\)\s*$'   # trailing (@handle) suffix from Twitter
            r')',
            re.IGNORECASE)

        # FIX 3: A valid creator handle pattern: 3–35 chars, only word chars + .~-
        _HANDLE_RE = re.compile(r'^[a-zA-Z0-9_.\-~]{3,35}$')

        # FIX 5: Also accept Title Case real names."
        # Minimum 3 chars per word to block "My Videos", "No Content" etc.
        _REALNAME_RE = re.compile(r'^[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){1,2}$')

        with DDGS(timeout=15) as ddgs:
            for sq in search_queries:
                time.sleep(1.5)
                emit_json("log", f"[WEB] Query: '{sq}'")
                try:
                    results = list(ddgs.text(sq, max_results=2))
                except Exception as e:
                    # FIX: never cache a failed search - allow retries on the next run
                    emit_json("warning", f"Error during web search for '{sq}': {e}")
                    continue
                if not results:
                    emit_json("log", f"[WEB] No results for '{sq}'")
                    continue
                if results:
                    raw_title = results[0].get('title', '').strip()
                    emit_json("log", f"[WEB] Result: '{raw_title}'")

                    # FIX 1: Extract handle from URL-like titles (e.g. "fansly.com/username")
                    url_match = re.search(
                        r'(?:onlyfans|fansly|fapello|thothub)\.com/([a-zA-Z0-9_.\-~]{3,35})',
                        raw_title, re.IGNORECASE)
                    if url_match:
                        raw_title = url_match.group(1)
                        emit_json("log", f"[WEB] URL handle extracted: '{raw_title}'")

                    # FIX 2: Strip known page-title suffixes
                    cleaned_title = _WEB_TITLE_JUNK_RE.sub('', raw_title).strip()

                    # FIX 3: Reject if result does not look like a handle or real name
                    is_handle   = bool(_HANDLE_RE.match(cleaned_title))
                    is_realname = bool(_REALNAME_RE.match(cleaned_title))
                    if not is_handle and not is_realname:
                        emit_json("log", f"[WEB] Rejected (not a handle or real name): '{cleaned_title}'")
                        continue

                    # FIX 4: Web result must not replace a good OCR result —
                    # only use it when the OCR query itself was ambiguous (not a
                    # clean handle already) or when it meaningfully corrects a
                    # truncated read (cleaned_title starts with the query string).
                    ocr_already_good = _HANDLE_RE.match(query.strip())
                    web_improves = (
                        cleaned_title.lower().startswith(query.lower().rstrip())
                        or query.lower().rstrip() in cleaned_title.lower()
                    )
                    if ocr_already_good and not web_improves:
                        # OCR was already a valid handle; web result is different → keep OCR
                        emit_json("log", f"[WEB] Rejected (OCR handle '{query}' already clean, web result '{cleaned_title}' differs)")
                        continue

                    blacklisted = is_blacklisted_label(cleaned_title.lower())
                    # Handles: full is_usable_label check (rejects spaces etc.)
                    # Real names: only blacklist check — spaces are intentional
                    accepted = (len(cleaned_title) >= 3 and not blacklisted
                                and (is_realname or is_usable_label(cleaned_title)))
                    if accepted:
                        emit_json("log", f"[WEB] Accepted: '{cleaned_title}' (from query '{query}')")
                        self.db.set_web_cache(cache_key, cleaned_title)
                        return cleaned_title, platform_dir
                    else:
                        emit_json("log", f"[WEB] Rejected (blacklist/filter): '{cleaned_title}'")

        # FIX: no result found -> return the original query WITHOUT caching the failure
        return self._finalize_text_label(query, platform_dir)

    def _resolve_known_name(self, label: str, exclude: Optional[List[str]] = None) -> Optional[str]:
        '''Returns an existing person name if the OCR label is a mangled variant of it
        (truncated reads, OCR letter swaps, case differences). Keeps the folder/person
        namespace clean instead of creating junk duplicates like comat/comatozze.'''
        if not label:
            return None
        low = label.lower().strip()
        if len(low) < 4:
            return None
        exclude = set(exclude or [])
        best_name, best_ratio = None, 0.0
        for name in self.saved_names:
            if name in exclude or name.lower().startswith("person "):
                continue
            nlow = name.lower()
            if nlow == low:
                return name
            # containment: truncated OCR read of a longer known name
            if (len(low) >= 5 and len(nlow) >= 5 and (low in nlow or nlow in low)
                    and abs(len(nlow) - len(low)) <= 8):
                return name
            r = difflib.SequenceMatcher(None, low, nlow).ratio()
            if r > best_ratio:
                best_name, best_ratio = name, r
        if best_ratio >= 0.86 and len(low) >= 6:
            return best_name
        return None

    def _finalize_text_label(self, label: str, platform_dir: str) -> Tuple[str, str]:
        '''Last gate for every OCR/filename-derived label: strips watermark residue,
        rejects generic junk and reuses an existing person name when the label is
        just a mangled variant of it. Returns (folder_name, platform_folder).'''
        stripped = strip_platform_prefix(label)
        if not is_usable_label(stripped):
            return "", platform_dir
        resolved = self._resolve_known_name(stripped)
        if resolved:
            return resolved, platform_dir
        return stripped, platform_dir

    def _generate_new_person_name(self, text_label: str, exclude: Optional[List[str]] = None) -> str:
        taken = set(self.saved_names)
        if exclude:
            taken.update(exclude)
        if text_label and text_label not in taken:
            return text_label
        counter = len(taken) + 1
        new_name = f"Person {counter}"
        while new_name in taken:
            counter += 1
            new_name = f"Person {counter}"
        return new_name

    @staticmethod
    def _is_usable_label(label: str) -> bool:
        """True if an OCR/web label is handle-like and free of URL junk.

        'fiammisxagain' -> usable; 'OnlyFan.com/fiammisxagain' -> rejected.
        """
        if not label:
            return False
        label = label.strip()
        if len(label) < 3 or len(label) > 40:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\- ]*", label))

    @staticmethod
    def _parse_ocr_result(result: Any) -> List[str]:
        """Extracts raw text strings from PaddleOCR 2.x and 3.x result formats."""
        texts: List[str] = []
        if not result:
            return texts
        for page in result:
            if isinstance(page, dict):
                # PaddleOCR >= 3.0: dict output containing 'rec_texts'
                texts.extend(str(t) for t in page.get('rec_texts', []))
            elif isinstance(page, (list, tuple)):
                # PaddleOCR 2.x: list of [box, (text, confidence)] lines
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        if isinstance(line[1], str):
                            texts.append(line[1])
                        elif isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                            texts.append(str(line[1][0]))
        return texts

    def _emit_preview_frame(self, second: int, phase: str, rgb_frame: np.ndarray) -> None:
        """Sends the currently analyzed frame to the UI as a small JPEG (base64),
        so the web app no longer has to re-decode the video file for its preview."""
        try:
            now = time.time()
            if now - getattr(self, "_last_preview_frame_ts", 0.0) < 2.0:
                return
            h, w = rgb_frame.shape[:2]
            scale = min(1.0, 384.0 / max(1, w))
            small = cv2.resize(rgb_frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ok:
                self._last_preview_frame_ts = now
                emit_json("preview_update", "Preview Frame", second=int(second), phase=phase,
                          frame_b64=base64.b64encode(buf.tobytes()).decode("ascii"))
        except Exception:
            pass
    
    def _get_ocr_text_from_frame(self, rgb_frame: np.ndarray) -> str:
        """Unified OCR method handling different engine output formats with OOM recovery."""
        try:
            if self.ocr_engine == "paddleocr":
                try:
                    result = self.reader.ocr(rgb_frame, cls=True)
                except TypeError:
                    # PaddleOCR >= 3.0 removed the cls parameter
                    result = self.reader.ocr(rgb_frame)
                texts = self._parse_ocr_result(result)
                return clean_ocr_results(texts)
            else:
                results = self.reader.readtext(rgb_frame, detail=0)
                return clean_ocr_results(results) if results else ""
        except Exception as e:
            emit_json("warning", f"OCR Inference Error ({type(e).__name__}): {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ""

    def _perform_ocr_video(self, cap: cv2.VideoCapture, total_duration: float, fps: float) -> str:
        all_detected_texts: List[str] = []
        target_seconds_ocr = calculate_target_seconds(total_duration, self.max_ocr_seconds)
        if not target_seconds_ocr:
            return ""

        for second in target_seconds_ocr:
            emit_json("preview_update", "Extracting Text...", second=second, phase="OCR Analysis")

            cap.set(cv2.CAP_PROP_POS_MSEC, int(second * 1000))
            ret, frame = cap.read()
            if ret:
                frame = resize_for_ai(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._emit_preview_frame(second, "OCR Analysis", rgb_frame)
                full_frame_text = self._get_ocr_text_from_frame(rgb_frame)
                if full_frame_text:
                    all_detected_texts.append(full_frame_text)
                    emit_json("ocr_text", "Extracted Text", text=full_frame_text, second=second)

        return merge_texts(all_detected_texts)

    def _perform_face_recognition_video(self, cap: cv2.VideoCapture, total_duration: float, fps: float,
                                        video_text_label: str, current_video_path: str) -> Set[str]:
        people_counts: Dict[str, int] = {}
        # New faces stay in memory during THIS video and are only written to the DB
        # after passing the MIN_VIDEO_FACE_FRAMES threshold (keeps one-hit ghosts out)
        pending_faces: Dict[str, np.ndarray] = {}
        first_seen: Dict[str, int] = {}
        target_seconds_face = calculate_target_seconds(total_duration, self.max_face_seconds)
        if not target_seconds_face:
            return set()

        known_id_by_name = dict(zip(self.saved_names, self.saved_ids))
        # FIX: new people are kept in memory only and committed to the DB AFTER the
        # ghost check, so one-time random faces no longer pollute the database.
        pending_encodings: List[np.ndarray] = []
        pending_names: List[str] = []

        for second in target_seconds_face:
            emit_json("preview_update", "Detecting Faces...", second=second, phase="Face Recognition")

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
            ret, frame = cap.read()
            if not ret:
                continue

            frame = resize_for_ai(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                self._emit_preview_frame(second, "Face Recognition", rgb_frame)
                face_locations = face_recognition.face_locations(
                    rgb_frame, model=FACE_MODEL, number_of_times_to_upsample=FACE_UPSAMPLE)
                if not face_locations:
                    continue
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            except Exception as e:
                emit_json("warning", f"Face Recognition Error ({type(e).__name__}): {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # FIX: a set prevents counting the same person twice within one frame
            names_this_frame = set()
            for encoding in face_encodings:
                matched_name: Optional[str] = None

                # 1) Match against known faces from the database
                if self.saved_encodings:
                    face_distances = face_recognition.face_distance(self.saved_encodings, encoding)
                    if len(face_distances) > 0:
                        potential_index = int(np.argmin(face_distances))
                        if face_distances[potential_index] <= self.face_threshold:
                            matched_name = self.saved_names[potential_index]

                # 2) Match against new people detected earlier in this same video
                if matched_name is None and pending_encodings:
                    pending_distances = face_recognition.face_distance(pending_encodings, encoding)
                    if len(pending_distances) > 0:
                        pending_index = int(np.argmin(pending_distances))
                        if pending_distances[pending_index] <= self.face_threshold:
                            matched_name = pending_names[pending_index]

                # 3) Register as a new person (in memory only, not in the DB yet)
                if matched_name is None:
                    # Conservative naming: new faces start with a neutral placeholder.
                    # The OCR/video label is only assigned AFTER the ghost check and
                    # only if exactly ONE new person remains (see end of this method).
                    matched_name = self._generate_new_person_name("", exclude=pending_names)
                    pending_encodings.append(encoding)
                    pending_names.append(matched_name)
                    emit_json("face_detected", "New Person Detected", name=matched_name, second=second)

                names_this_frame.add(matched_name)

            if not names_this_frame:
                continue

            # --- Tracking: verify that the detected faces are stable in the scene ---
            individual_trackers = []
            frame_h, frame_w = frame.shape[:2]  # image bounds

            for (top, right, bottom, left) in face_locations:
                # Clamp coordinates strictly to the image area
                left = max(0, left)
                top = max(0, top)
                right = min(frame_w, right)
                bottom = min(frame_h, bottom)

                bbox_w = right - left
                bbox_h = bottom - top

                # Skip invalid or too small boxes
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                bbox = (left, top, bbox_w, bbox_h)

                try:
                    tracker = cv2.TrackerMIL_create()
                except AttributeError:
                    tracker = cv2.legacy.TrackerMIL_create()

                try:
                    tracker.init(frame, bbox)
                    individual_trackers.append(tracker)
                except Exception as e:
                    emit_json("warning", f"Tracker init skipped ({type(e).__name__}): {e}")
                    continue

            tracking_successful = len(individual_trackers) > 0
            for _ in range(max(0, FACE_TRACK_VERIFY_FRAMES)):
                ret_track, frame_track = cap.read()
                if not ret_track:
                    break
                frame_track = resize_for_ai(frame_track)

                for tracker in individual_trackers:
                    success, _ = tracker.update(frame_track)
                    if not success:
                        tracking_successful = False
                        break

                if not tracking_successful:
                    break

            if tracking_successful:
                for name in names_this_frame:
                    if name not in people_counts:
                        first_seen[name] = second
                    people_counts[name] = people_counts.get(name, 0) + 1
                    if name in known_id_by_name:
                        emit_json("face_detected", "Face Recognized", name=name, second=second)

        # --- Ghost filtering: keep only people seen in enough frames ---
        required_frames = max(1, min(MIN_VIDEO_FACE_FRAMES, len(target_seconds_face)))
        valid_people = {name for name, count in people_counts.items() if count >= required_frames}

        # Persist pending faces that passed the threshold; the rest are ghosts and discarded
        for pend_name, pend_enc in pending_faces.items():
            if pend_name in valid_people:
                new_id = self.db.save_face(pend_name, pend_enc.tobytes())
                self.saved_encodings.append(pend_enc)
                self.saved_names.append(pend_name)
                self.saved_ids.append(new_id)
                self.db.add_history(new_id, current_video_path)
                emit_json("face_detected", "Person Saved to Memory", name=pend_name, second=second)

        # --- Conservative text label assignment ---------------------------------
        # The OCR/web label describes the whole VIDEO, not a single face. Use it as
        # a person name only when exactly ONE new (pending) person survived the
        # ghost check and the label is clean and unused; otherwise every new person
        # keeps the neutral "Person N" placeholder assigned during detection.
        valid_pending = [n for n in pending_names if n in valid_people]
        label_free = (
            self._is_usable_label(video_text_label)
            and video_text_label not in self.saved_names
            and video_text_label not in pending_names
        )
        if label_free and len(valid_pending) == 1:
            old_name = valid_pending[0]
            people_counts[video_text_label] = people_counts.pop(old_name)
            first_seen[video_text_label] = first_seen.pop(old_name, 0)
            pending_names[pending_names.index(old_name)] = video_text_label
            valid_people.discard(old_name)
            valid_people.add(video_text_label)
            emit_json("face_detected", f"Person Named from Video Text (was {old_name})",
                      name=video_text_label, second=first_seen.get(video_text_label, 0))
        elif video_text_label and len(valid_pending) > 1:
            emit_json("log", f"Video text label '{video_text_label}' NOT used as name: "
                             f"{len(valid_pending)} new people detected in this video.")

        # FIX: commit only valid new people to the DB (one encoding each); discard ghosts
        for i, name in enumerate(pending_names):
            if name in valid_people and name not in known_id_by_name:
                new_id = self.db.save_face(name, pending_encodings[i].tobytes())
                self.saved_encodings.append(pending_encodings[i])
                self.saved_names.append(name)
                self.saved_ids.append(new_id)
                known_id_by_name[name] = new_id

        for name, count in people_counts.items():
            if name not in valid_people:
                # FIX: report the second at which the ghost was first seen (was misleading before)
                emit_json("ghost_face", "Ignored Ghost", name=name, count=count, second=first_seen.get(name, 0))
            elif name in known_id_by_name:
                self.db.add_history(known_id_by_name[name], current_video_path)

        return valid_people

    def process_media(self) -> None:
        media_files = []
        valid_exts = tuple(ext.replace('*', '').lower() for ext in SUPPORTED_EXTENSIONS)
        image_exts = ('.jpg', '.jpeg', '.png')

        for root, dirs, files in os.walk(self.input_directory):
            if "sorted_videos" in dirs:
                dirs.remove("sorted_videos")
            for filename in files:
                if filename.lower().endswith(valid_exts):
                    media_files.append(os.path.join(root, filename))

        media_files = sorted(set(media_files))  # FIX: deterministic processing order
        if not media_files:
            emit_json("error", f"No supported media files found in '{self.input_directory}'.")
            sys.exit(1)

        total_files = len(media_files)
        emit_json("log", f"Starting analysis of {total_files} media file(s)...")
        # --- Resume support: results of a previously interrupted run are loaded
        # from the checkpoint so already-analyzed files can be skipped ---
        self.checkpoint_path = os.path.join(self.input_directory, ".ai_sorter_checkpoint.json")
        checkpoint: Dict[str, Dict[str, Any]] = {}
        if getattr(self, "resume_analysis", True) and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    checkpoint = json.load(f).get("report", {})
            except Exception:
                checkpoint = {}
        if checkpoint:
            emit_json("log", f"RESUME: {len(checkpoint)} file(s) already analyzed in a previous run - skipping their analysis.")

        for index, file_path in enumerate(media_files, start=1):
            emit_json("progress", "File Selected", file_count=index, total_files=total_files, current_file=file_path)
            if file_path in checkpoint:
                # Resume: reuse the stored result instead of re-analyzing this file
                prev = checkpoint[file_path]
                self.final_report[file_path] = {
                    "raw_text": prev.get("raw_text", ""),
                    "found_text": prev.get("found_text", ""),
                    "platform": prev.get("platform", ""),
                    "people": set(),
                    "final_folder": ""
                }
                continue

            is_image = file_path.lower().endswith(image_exts)
            media_text_label = ""
            platform_folder = ""
            people_in_current_media: Set[str] = set()
            used_raw_text = ""

            if is_image:
                frame = cv2.imread(file_path)
                if frame is None:
                    # FIX: report unreadable images instead of failing silently
                    emit_json("warning", f"Could not read image file: '{os.path.basename(file_path)}'. Sorted by filename only.")
                else:
                    frame = resize_for_ai(frame)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    raw_ocr_text = self._get_ocr_text_from_frame(rgb_frame)
                    filename_text = extract_words_from_filename(file_path)

                    if raw_ocr_text:
                        used_raw_text = raw_ocr_text
                        emit_json("ocr_text", "Extracted Text", text=raw_ocr_text, second=0)
                        media_text_label, platform_folder = self._verify_text_with_web(raw_ocr_text)

                    # FIX: fall back to the filename if OCR was missing OR useless
                    # (timestamp-only text, blacklisted, empty result, etc.)
                    if not media_text_label and filename_text:
                        used_raw_text = f"[Filename] {filename_text}"
                        media_text_label, platform_folder = self._verify_text_with_web(filename_text)
            else:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    # FIX: report unopenable videos instead of failing silently
                    emit_json("warning", f"Could not open video file: '{os.path.basename(file_path)}'. Sorted by filename only.")
                else:
                    try:
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        if fps <= 0.0:
                            fps = 30.0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        if total_frames <= 0:
                            # FIX: broken/stream-like containers have no frame count
                            emit_json("warning", f"Could not determine duration: '{os.path.basename(file_path)}'. Sorted by filename only.")
                        else:
                            total_duration = total_frames / fps

                            raw_ocr_text = self._perform_ocr_video(cap, total_duration, fps)
                            filename_text = extract_words_from_filename(file_path)

                            if raw_ocr_text:
                                used_raw_text = raw_ocr_text
                                media_text_label, platform_folder = self._verify_text_with_web(raw_ocr_text)

                            if not media_text_label and filename_text:
                                used_raw_text = f"[Filename] {filename_text}"
                                media_text_label, platform_folder = self._verify_text_with_web(filename_text)

                            people_in_current_media = self._perform_face_recognition_video(
                                cap, total_duration, fps, media_text_label, file_path)
                    finally:
                        cap.release()

            self.final_report[file_path] = {
                "raw_text": used_raw_text,
                "found_text": media_text_label,
                "platform": platform_folder,
                "people": people_in_current_media,
                "final_folder": ""
            }
            # --- Checkpoint: remember this file's result for a possible resume ---
            checkpoint[file_path] = {
                "raw_text": used_raw_text,
                "found_text": media_text_label,
                "platform": platform_folder,
            }
            try:
                with open(self.checkpoint_path, "w") as f:
                    json.dump({"report": checkpoint}, f)
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Release Paddle's caching allocator as well (no-op on CPU)
            try:
                import paddle
                if hasattr(paddle.device, 'cuda') and hasattr(paddle.device.cuda, 'empty_cache'):
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass

    def sort_media(self) -> None:
        emit_json("log", "STARTING TO SORT MEDIA FILES...")
        os.makedirs(self.target_base_folder, exist_ok=True)

        # FIX: write a manifest of all moves so the web app can undo them precisely
        manifest_path = os.path.join(self.target_base_folder, "manifest.json")
        manifest: Dict[str, str] = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f).get("moves", {})
            except Exception:
                manifest = {}

        for file_path, data in self.final_report.items():
            filename = os.path.basename(file_path)
            web_text = data.get("found_text", "")
            original_platform = data.get("platform", "")

            folder_name = clean_junk(web_text)
            if folder_name and not is_usable_label(strip_platform_prefix(folder_name)):
                folder_name = ""
            folder_name = folder_name or extract_date_from_filename(filename)
            final_platform = original_platform or "Unknown"

            sanitized_folder_name = sanitize_filename(folder_name)
            if not sanitized_folder_name:
                sanitized_folder_name = "Unnamed"  # FIX: never produce an empty folder name
            target_folder = os.path.join(self.target_base_folder, sanitize_filename(final_platform, 100), sanitized_folder_name)
            data["final_folder"] = f"{final_platform}/{sanitized_folder_name}"

            os.makedirs(target_folder, exist_ok=True)
            target_path = os.path.join(target_folder, filename)

            if os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_folder, f"{name}_{str(time.time()).replace('.', '_')}{ext}")

            try:
                shutil.move(file_path, target_path)
                manifest[target_path] = file_path  # remember original location for undo
                self.db.update_history_paths(file_path, target_path)
                emit_json("log", f"Moved '{filename}' to '{data['final_folder']}'")
            except Exception as e:
                emit_json("error", f"ERROR moving '{filename}': {e}")

        try:
            with open(manifest_path, "w") as f:
                json.dump({"moves": manifest}, f, indent=2)
        except Exception as e:
            emit_json("warning", f"Could not write undo manifest: {e}")
        # Run completed: the checkpoint is no longer needed
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
        except OSError:
            pass

# ====
# MAIN EXECUTION BLOCK
# ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered media sorter (OCR + face recognition).")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--disable-web-search", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ocr-engine", type=str, default="paddleocr", choices=["paddleocr", "easyocr"])

    args = parser.parse_args()

    # FIX: guarantee that a termination event reaches the web app, even on crashes,
    # so the UI never keeps showing a "running" state for a dead process.
    try:
        input_directory = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_directory):
            emit_json("error", f"Input directory does not exist: '{input_directory}'")
            sys.exit(1)

        target_base_folder = os.path.join(input_directory, "sorted_videos")
        db_path = os.path.join(input_directory, "face_memory.sqlite")
        use_web = not args.disable_web_search

        processor = VideoProcessor(
            input_directory=input_directory,
            db_path=db_path,
            target_base_folder=target_base_folder,
            max_ocr_seconds=MAX_SECONDS_OCR,
            max_face_seconds=MAX_SECONDS_FACE,
            face_threshold=FACE_RECOGNITION_THRESHOLD,
            enable_web_search=use_web,
            ocr_engine=args.ocr_engine
        )
        processor.resume_analysis = not args.no_resume
        processor.process_media()
        processor.sort_media()
        emit_json("finished", "Script finished.")
    except KeyboardInterrupt:
        emit_json("warning", "Process interrupted by user.")
        emit_json("finished", "Script stopped.")
    except SystemExit:
        raise
    except Exception as e:
        emit_json("error", f"Fatal error: {e}")
        emit_json("finished", "Script finished with an error.")
