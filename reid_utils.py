"""
Shared Re-Identification (OSNet) & FAISS Similarity Utilities
=============================================================
Used by both detector.py (ground camera) and drone_tracker.py (aerial tracker)
so feature extraction, normalization, and FAISS indexing stay 100% consistent.
"""

import os
import cv2
import numpy as np
import torch
import faiss
from PIL import Image


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Select CUDA if available and requested, otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MSMT17_FILENAME = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
    "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
MSMT17_URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/main/{MSMT17_FILENAME}"

WEAPON_ONNX_FILENAME = "weapon_hadi959.onnx"
WEAPON_ONNX_URL = "https://huggingface.co/Hadi959/weapon-detection-yolov8/resolve/main/best.onnx"

WEAPON_CCTV_FILENAME = "weapon_cive202_cctv.pt"
WEAPON_CCTV_URL = "https://huggingface.co/cive202/weapon-detection-yolov8-cctv/resolve/main/best.pt"
WEAPON_CCTV_SHA256 = "b28351f6c7b6ee7a19ca207392ceec5370227e82a639616985127a15d6f4956c"

_UNSAFE_PICKLE_MODULES = {
    "os", "posix", "nt", "subprocess", "sys", "builtins", "eval", "exec",
    "socket", "http", "urllib", "shutil", "pathlib", "pty", "commands",
    "platform", "importlib",
}


def verify_pytorch_checkpoint_safety(path: str, expected_sha256: str = None) -> bool:
    """
    Cybersecurity static audit before torch.load / YOLO():
    1. Optional SHA-256 verification against pinned upstream digest.
    2. Inspect embedded data.pkl opcodes with pickletools for code-execution globals.
    """
    import hashlib
    import pickletools
    import zipfile

    if not os.path.isfile(path):
        return False
    if expected_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected_sha256:
            print(f"Security Warning: SHA-256 mismatch for {path} ({h.hexdigest()})")
            return False

    try:
        with zipfile.ZipFile(path, "r") as z:
            pkl_names = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not pkl_names:
                return False
            for pn in pkl_names:
                recent_strs = []
                for op, arg, _ in pickletools.genops(z.read(pn)):
                    if op.name in (
                        "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
                        "STRING", "BINSTRING", "SHORT_BINSTRING",
                    ):
                        recent_strs.append(str(arg))
                        if len(recent_strs) > 4:
                            recent_strs.pop(0)
                    elif op.name in ("GLOBAL", "INST"):
                        parts = str(arg).replace("\n", " ").split()
                        mod = parts[0].lower().split(".")[0] if parts else ""
                        attr = parts[1].lower() if len(parts) > 1 else ""
                        if mod in _UNSAFE_PICKLE_MODULES or attr in {"eval", "exec", "system", "popen", "spawn"}:
                            print(f"Security Blocked: unsafe pickle global '{arg}' in {path}")
                            return False
                    elif op.name == "STACK_GLOBAL":
                        mod_raw = recent_strs[-2] if len(recent_strs) >= 2 else ""
                        attr_raw = recent_strs[-1] if len(recent_strs) >= 1 else ""
                        mod = mod_raw.lower().split(".")[0]
                        attr = attr_raw.lower()
                        if mod in _UNSAFE_PICKLE_MODULES or attr in {"eval", "exec", "system", "popen", "spawn"}:
                            print(f"Security Blocked: unsafe STACK_GLOBAL '{mod_raw}.{attr_raw}' in {path}")
                            return False
                    elif op.name in ("EXT1", "EXT2", "EXT4"):
                        print(f"Security Blocked: disallowed pickle opcode '{op.name}' in {path}")
                        return False
        return True
    except Exception as e:
        print(f"Security Warning: checkpoint inspection failed for {path} ({e})")
        return False


def ensure_cctv_weapon_model() -> str:
    """
    Download & verify CCTV-trained YOLOv8 weapon model (cive202/weapon-detection-yolov8-cctv)
    using SHA-256 pinning + pickletools opcode audit before loading.
    Falls back to pickle-free ONNX model if download or verification fails.
    """
    cache_dir = os.path.expanduser("~/.cache/torch/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, WEAPON_CCTV_FILENAME)
    if not (os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000):
        try:
            import urllib.request
            print("Downloading CCTV weapon detector (cive202/weapon-detection-yolov8-cctv)...")
            req = urllib.request.Request(WEAPON_CCTV_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print(f"Warning: Could not download CCTV weapon model ({e}); falling back to ONNX.")
            return ensure_weapon_onnx_model()

    if verify_pytorch_checkpoint_safety(dest, expected_sha256=WEAPON_CCTV_SHA256):
        return dest
    try:
        os.remove(dest)
    except OSError:
        pass
    return ensure_weapon_onnx_model()


def ensure_weapon_onnx_model() -> str:
    """Return path to cached pickle-free ONNX weapon detector, downloading if needed."""
    cache_dir = os.path.expanduser("~/.cache/torch/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, WEAPON_ONNX_FILENAME)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    try:
        import urllib.request
        print("Downloading pickle-free ONNX weapon detector (Hadi959/weapon-detection-yolov8)...")
        req = urllib.request.Request(WEAPON_ONNX_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
    except Exception as e:
        print(f"Warning: Could not download ONNX weapon model ({e}); continuing with pose-only mode.")
        return ""


def _ensure_msmt17_weights() -> str:
    """Return path to cached MSMT17 OSNet Re-ID weights, downloading if needed."""
    cache_dir = os.path.expanduser("~/.cache/torch/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, MSMT17_FILENAME)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    try:
        import urllib.request
        print("Downloading official OSNet MSMT17 Person Re-ID weights...")
        req = urllib.request.Request(MSMT17_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
    except Exception as e:
        print(f"Warning: Could not download MSMT17 weights ({e}); falling back to ImageNet.")
        return ""


def load_osnet_model(weights_path: str = None, device: torch.device = None):
    """
    Lazy-load OSNet (osnet_x1_0) for person Re-ID.
    If weights_path is omitted, automatically uses MSMT17 Person Re-ID weights
    (or ImageNet pretrained=True if offline).
    """
    import torchreid

    if device is None:
        device = get_device()

    resolved_path = weights_path if (weights_path and os.path.isfile(weights_path)) else _ensure_msmt17_weights()
    use_imagenet_fallback = not bool(resolved_path)

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        loss="softmax",
        pretrained=use_imagenet_fallback,
    )

    if resolved_path:
        torchreid.utils.load_pretrained_weights(model, resolved_path)
        print(f"Loaded OSNet Re-ID weights from: {resolved_path}")
    else:
        print("Loaded built-in ImageNet OSNet (osnet_x1_0) weights.")

    model = model.to(device)
    model.eval()
    return model, device


def extract_outfit_color_feature(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract upper-body and lower-body 2D (S, V) + Hue signatures (center 50% width
    to ignore background) so black/dark vs white/bright vs colored outfits stay
    clearly separated. Returns an L2-normalized 144-D vector (72 upper + 72 lower).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros(144, dtype=np.float32)

    h, w = image_bgr.shape[:2]
    x1, x2 = int(w * 0.25), int(w * 0.75)
    upper = image_bgr[int(h * 0.18):int(h * 0.50), x1:x2]
    lower = image_bgr[int(h * 0.52):int(h * 0.86), x1:x2]

    parts = []
    for part in (upper, lower):
        if part.size == 0:
            parts.append(np.zeros(72, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        sv_hist = cv2.calcHist([hsv], [1, 2], None, [8, 8], [0, 256, 0, 256]).flatten().astype(np.float32)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten().astype(np.float32)
        sv_hist /= (np.linalg.norm(sv_hist) + 1e-8)
        h_hist /= (np.linalg.norm(h_hist) + 1e-8)
        vec = np.concatenate([0.85 * sv_hist, 0.15 * h_hist]).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-8)
        parts.append(vec)

    # Weight upper body 60%, lower body 40%
    out = np.concatenate([0.775 * parts[0], 0.632 * parts[1]]).astype(np.float32)
    out /= (np.linalg.norm(out) + 1e-8)
    return out


def extract_reid_feature(image_bgr: np.ndarray, model, device: torch.device) -> np.ndarray:
    """
    Extract an L2-normalized 512-D OSNet Re-ID feature vector from a BGR crop.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((128, 256))  # (width, height) = 128x256
    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0

    # ImageNet mean/std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    feat_np = feat.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm
    return feat_np


def create_faiss_index(dimension: int = 512, use_gpu: bool = True):
    """Create a FAISS Inner-Product index (cosine similarity on L2-normalized vectors)."""
    index = faiss.IndexFlatIP(dimension)
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    return index


def draw_hud_panel(frame: np.ndarray, status_text: str, status_color: tuple, lines: list):
    """Draw a semi-transparent dark HUD panel at top-left for crisp readability."""
    h, w = frame.shape[:2]
    panel_w = min(460, w - 20)
    panel_h = 42 + len(lines) * 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), status_color, 2)

    # Status pill
    cv2.rectangle(frame, (18, 18), (18 + panel_w - 16, 44), status_color, -1)
    cv2.putText(frame, status_text, (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 68 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 237, 243), 1)


def draw_label_pill(frame: np.ndarray, text: str, x: int, y: int, bg_color: tuple):
    """Draw a crisp filled label pill above a bounding box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(0, y - th - 10)
    cv2.rectangle(frame, (x, y_top), (x + tw + 12, y_top + th + 10), bg_color, -1)
    cv2.putText(frame, text, (x + 6, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_corner_box(frame: np.ndarray, bbox, color: tuple, thickness: int = 2, corner_len: int = 18):
    """Draw a clean tactical corner-bracket bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cl = min(corner_len, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
    for (px, py, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * cl, py), color, thickness + 1)
        cv2.line(frame, (px, py), (px, py + dy * cl), color, thickness + 1)
