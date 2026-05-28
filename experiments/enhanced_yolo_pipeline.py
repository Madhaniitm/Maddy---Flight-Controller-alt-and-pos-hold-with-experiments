"""
Enhanced YOLO Tier — Research Paper Implementations
====================================================
Techniques implemented:

1. YOLO-World (CVPR 2024, arxiv:2401.17270)
   Open-vocabulary detection for structural drone hazards NOT in COCO-80:
   wall, door, window, wire, steps, pillar, barrier, ceiling, shelf.
   Ref: Cheng et al. (2024) [37]

2. CLAHE Low-Light Enhancement (PMC12273955, 2024-25)
   Contrast-Limited Adaptive Histogram Equalization applied before inference.
   Recovers detections in dim/dark scenes (+2-5% mAP in low-light benchmarks).

3. CLIP Scene Hazard Screening (arxiv:2504.13399, 2025)
   *** EXPERIMENTAL USE ONLY — not in production Tier 2 pipeline ***
   CLIP is retained here for V-series experiment results (V1, V2, V4, V5).
   It proved unreliable on 320×240 ESP32 frames: scores within ±0.013 of
   the uniform (random) baseline of 0.200 across all 5 labels. This failure
   itself is a thesis result — it motivates the LLM cognitive authority claim.
   Use enhanced_yolo_infer(use_clip=False) in production (the default).
   Ref: Clip results used in thesis section on "why LLM is needed as Tier 3".

4. YOLOv11n COCO — Primary Object Detector  [NEW, ref 35, 38, 39]
   Trained COCO-80 model. High recall for person (~90%) and 79 everyday classes.
   Runs first; YOLO-World supplements for structural classes not in COCO.
   Justification: Kim et al. (2024) YOLO-IHD shows COCO-trained YOLO achieves
   80% precision / 68% recall for indoor person vs. near-zero for zero-shot
   YOLO-World text alignment. Ref: [39] PMC10857234.
   System: Ahmmad et al. (2025) [35] uses YOLOv11 + DA v2 + LLM — this work.

5. DepthAnything v2 Metric Indoor  [NEW, ref 36, 42, 43]
   Per-pixel metric depth estimation (real metres, not relative pixel values).
   Samples depth at each YOLO bounding-box centre → replaces broken geometric
   heuristic (est_dist = h / bbox_h × 0.3, ±50% error).
   Model: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (HuggingFace)
   Validated indoor MAE ≈ 7.2 cm (Ahmmad et al. 2025 [35]).
   Ref: Yang et al. (2024) NeurIPS [36]; Bui et al. (2024) IEEE CAI [42].

6. LLM Distance Adjudication — Cognitive Authority
   Prompt-level instruction: DA v2 depth is advisory; LLM visual reasoning
   is the authority. LLM overrides sensor data when image contradicts it.
   Ref: Ahmmad et al. (2025) [35]; arxiv:2602.07680.

Production Tier 2 pipeline (current):
   Frame → CLAHE → YOLOv11n COCO + YOLO-World → merge → DA v2 depth → LLM
   robust_local_detector (14 ms) runs in parallel for emergency triggering.
   CLIP: disabled in production (use_clip=False in enhanced_yolo_infer).

Install notes (for techniques 4 & 5):
   pip install ultralytics            # YOLOv11n via YOLO("yolo11n.pt")
   pip install transformers pillow    # DepthAnything v2 via HuggingFace pipeline
   Model weights auto-download on first run from HuggingFace Hub.
"""

import re
import time
import numpy as np
import cv2
from pathlib import Path

# ── Structural hazard vocabulary for YOLO-World ──────────────────────────────
# Only classes NOT in COCO-80. COCO handles person/chair/table/laptop etc.
# COCO-80 does not include: wall, door, window, wire, pillar, steps, barrier.
STRUCTURAL_CLASSES = [
    "wall", "door", "window", "wire", "cable",
    "pillar", "column", "steps", "staircase",
    "barrier", "ceiling", "shelf",
]

# ── Threat / security vocabulary for YOLO-World (zero-shot) ──────────────────
# These extend YOLO-World's open-vocab capability to semantic security threats
# that are absent from COCO-80.
#
# Approach A — YOLO-World zero-shot (no training needed):
#   Pass THREAT_CLASSES to model.set_classes(). Works out-of-box but recall is
#   LOW on small/blurry 320×240 objects (expect 30–50% recall at best).
#   Use THREAT_WORLD_CONF = 0.15 (same as structural classes).
#   Validated via demo video — not a formal experiment.
#
# Approach B — Fine-tuned YOLOv11n on weapons dataset (higher recall):
#   Open Images V7 has: "Firearm", "Gun", "Handgun", "Rifle", "Knife".
#   Fine-tune YOLOv11n on ~2000 labelled weapon images → ~75-85% recall
#   on typical indoor frames. Load via load_threat_yolo().
#   Weights not bundled — see load_threat_yolo() stub below.
#
# Current system status: demo-video validated, no formal experiment needed.
# The LLM (Tier 3) remains the primary threat-semantic layer; YOLO threat
# classes are supplementary metadata that help the LLM reason faster.
THREAT_CLASSES = [
    "gun", "pistol", "rifle", "firearm", "weapon",
    "knife", "machete", "axe",
    "explosive", "bomb", "grenade",
    "suspicious package",
]

THREAT_WORLD_CONF = 0.15   # zero-shot — same as structural classes

# Combined YOLO-World vocabulary (structural + threat in one set_classes call)
STRUCTURAL_AND_THREAT_CLASSES = STRUCTURAL_CLASSES + THREAT_CLASSES

# ── Threat risk mapping ───────────────────────────────────────────────────────
# Any detected threat class → immediate hazard regardless of distance.
THREAT_RISK_CLASSES = {
    "gun", "pistol", "rifle", "firearm", "weapon",
    "knife", "machete", "axe",
    "explosive", "bomb", "grenade", "suspicious package",
}

# Legacy alias — used by some experiments that pass HAZARD_CLASSES explicitly
HAZARD_CLASSES = [
    "person", "wall", "door", "table", "chair", "box",
    "pillar", "column", "steps", "staircase", "wire", "cable",
    "barrier", "obstacle", "window", "ceiling", "shelf",
]

# ── CLIP scene labels + hazard mapping ──────────────────────────────────────
CLIP_SCENE_LABELS = [
    "open room safe path",        # safe
    "person up close",            # hazard
    "wall blocking path",         # hazard
    "dark or covered lens",       # hazard
    "cluttered room obstacles",   # caution
]

CLIP_HAZARD_MAP = {
    "open room safe path":      "safe",
    "person up close":          "hazard",
    "wall blocking path":       "hazard",
    "dark or covered lens":     "hazard",
    "cluttered room obstacles": "caution",
}

CLIP_CONF_THRESHOLD = 0.204  # just above uniform (0.200 for 5 labels)

# ── Confidence thresholds (Technique 6)  ─────────────────────────────────────
# YOLO internally letterboxes all input to 640×640 regardless of source size.
# Pre-upscaling (e.g. 320×240→960×720) adds a redundant downscale step that
# LOSES features — tested and confirmed to hurt person detection on ESP32 frames.
# The right fix: lower conf thresholds on the original 320×240 frame directly.
# conf=0.20 for COCO (trained model, moderate trust);
# conf=0.15 for YOLO-World (zero-shot text-image alignment, lower scores expected).
COCO_CONF        = 0.20   # COCO trained — moderate threshold
WORLD_CONF       = 0.15   # YOLO-World zero-shot — lower threshold needed
# Legacy: kept for any code that references these
YOLO_UPSCALE_W   = 320    # no upscale — run at native resolution
YOLO_UPSCALE_H   = 240

# ── Navigable openings — not treated as obstacles ────────────────────────────
SAFE_DETECTION_CLASSES = {"door", "window"}


# ════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2: CLAHE preprocessing
# ════════════════════════════════════════════════════════════════════════════

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    CLAHE in LAB colorspace — enhances low-light frames before YOLO inference.
    clipLimit=3.0, tileGridSize=8×8 from PMC12273955 benchmark.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 4: YOLOv11n COCO — primary person/object detector
# ════════════════════════════════════════════════════════════════════════════

def load_coco_yolo():
    """
    Load YOLOv11n trained on COCO-80.
    Primary detector for person, chair, table, laptop, and 76 other classes.
    High recall for person (~90%) vs near-zero for zero-shot YOLO-World.
    Ref: Kim et al. 2024 [39]; Ahmmad et al. 2025 [35].
    Returns (model, "yolo11-coco") or (None, "unavailable").
    """
    repo_root = Path(__file__).parent.parent
    candidates = [
        repo_root / "yolo11n.pt",
        Path("yolo11n.pt"),
    ]
    pt_path = next((p for p in candidates if p.exists()), None)

    try:
        from ultralytics import YOLO
        if pt_path:
            model = YOLO(str(pt_path))
            print(f"[YOLOv11n COCO] Loaded from {pt_path}")
        else:
            model = YOLO("yolo11n.pt")   # auto-download
            print("[YOLOv11n COCO] Downloaded and loaded")
        return model, "yolo11-coco"
    except Exception as e:
        print(f"[YOLOv11n COCO] Load failed: {e}")
        return None, "unavailable"


# ════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 1 + fallback: YOLO-World — structural hazard detector
# ════════════════════════════════════════════════════════════════════════════

def load_enhanced_yolo(classes: list = STRUCTURAL_CLASSES):
    """
    Load YOLO-World for structural hazard classes (wall, door, wire, etc.).
    Falls back to YOLO11n (now treated as COCO fallback) if YOLO-World fails.
    Backward-compatible: existing experiments call load_enhanced_yolo().
    Returns (model, yolo_type_str).
    """
    repo_root = Path(__file__).parent.parent
    world_candidates = [
        repo_root / "yolov8s-worldv2.pt",
        repo_root / "Image verbalization experiments" / "yolov8s-worldv2.pt",
        Path("yolov8s-worldv2.pt"),
    ]
    world_pt = next((p for p in world_candidates if p.exists()), None)

    try:
        from ultralytics import YOLOWorld
        if world_pt:
            model = YOLOWorld(str(world_pt))
        else:
            model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(classes)
        print(f"[YOLO-World] Loaded — {len(classes)} structural classes: {classes}")
        return model, "yolo-world"
    except Exception as e:
        print(f"[YOLO-World] Failed ({e}) — trying YOLOv11n as fallback")
        try:
            from ultralytics import YOLO
            yolo11_candidates = [
                repo_root / "yolo11n.pt",
                Path("yolo11n.pt"),
            ]
            pt = next((p for p in yolo11_candidates if p.exists()), None)
            model = YOLO(str(pt)) if pt else YOLO("yolo11n.pt")
            print("[YOLOv11n] Fallback loaded (YOLO-World unavailable)")
            return model, "yolo11"
        except Exception as e2:
            print(f"[YOLOv11n] Failed ({e2}) — simulation mode")
            return None, "simulation"


# ════════════════════════════════════════════════════════════════════════════
# THREAT DETECTION — YOLO-World vocab extension + fine-tuned YOLOv11 stub
# ════════════════════════════════════════════════════════════════════════════

def load_world_with_threats(include_structural: bool = True):
    """
    Load YOLO-World with THREAT_CLASSES (+ optionally STRUCTURAL_CLASSES).

    Zero-shot threat detection: gun, pistol, rifle, explosive, knife, etc.
    No training required — YOLO-World uses open-vocabulary text-image alignment.

    Limitations on 320×240 ESP32 frames:
      - Recall ~30–50% for small weapon objects (blurry at low resolution)
      - False positive rate is non-negligible — use THREAT_WORLD_CONF=0.15
      - Best treated as supplementary metadata for the LLM, not a hard trigger
      - The LLM (Tier 3) is still the authoritative threat classifier

    Validated: demo video (not a formal experiment).
    The LLM correctly identifies "person pointing gun" from image alone even
    when YOLO-World misses the weapon — confirming Tier 3 cognitive authority.

    Returns (model, "yolo-world-threat") or (None, "unavailable").
    """
    classes = (STRUCTURAL_AND_THREAT_CLASSES if include_structural
               else THREAT_CLASSES)
    repo_root = Path(__file__).parent.parent
    world_candidates = [
        repo_root / "yolov8s-worldv2.pt",
        repo_root / "Image verbalization experiments" / "yolov8s-worldv2.pt",
        Path("yolov8s-worldv2.pt"),
    ]
    world_pt = next((p for p in world_candidates if p.exists()), None)

    try:
        from ultralytics import YOLOWorld
        if world_pt:
            model = YOLOWorld(str(world_pt))
        else:
            model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(classes)
        print(f"[YOLO-World+Threat] Loaded — {len(classes)} classes "
              f"({len(STRUCTURAL_CLASSES)} structural + {len(THREAT_CLASSES)} threat)")
        return model, "yolo-world-threat"
    except Exception as e:
        print(f"[YOLO-World+Threat] Failed: {e}")
        return None, "unavailable"


def load_threat_yolo(weights_path: str = None):
    """
    Load a fine-tuned YOLOv11n model for weapon/threat detection.

    This is higher recall than zero-shot YOLO-World on low-res frames.
    Fine-tune recipe (not done in this work — listed for future extension):
      Dataset : Open Images V7 — classes: Firearm, Gun, Handgun, Rifle, Knife
      Base    : yolo11n.pt (same COCO backbone as primary detector)
      Training: ~50 epochs, ~2000 labelled images → expect ~75–85% recall
      Tool    : ultralytics train data=openimages_weapons.yaml model=yolo11n.pt

    Current status: WEIGHTS NOT AVAILABLE — function returns (None, "unavailable").
    The LLM (Tier 3) serves as the primary threat classifier until weights exist.
    Validated via demo video: LLM correctly describes "gun pointed at camera"
    from raw image without any YOLO assistance.

    Args:
        weights_path: path to fine-tuned .pt file. If None, returns unavailable.

    Returns (model, "yolo11-threat") or (None, "unavailable").
    """
    if weights_path is None:
        print("[YOLOv11-Threat] No weights path provided — fine-tuned model unavailable.")
        print("                 LLM (Tier 3) handles threat classification from image.")
        return None, "unavailable"

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"[YOLOv11-Threat] Fine-tuned model loaded from {weights_path}")
        return model, "yolo11-threat"
    except Exception as e:
        print(f"[YOLOv11-Threat] Load failed: {e}")
        return None, "unavailable"


def is_threat_detection(label: str) -> bool:
    """Return True if the detection label is a security threat class."""
    return label.lower().strip() in THREAT_RISK_CLASSES


# ════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3: CLIP scene hazard screener
# ════════════════════════════════════════════════════════════════════════════

def load_clip():
    """
    Open-CLIP ViT-B-32 (laion2b) for zero-shot scene hazard screening.

    *** EXPERIMENTAL USE ONLY — not in production Tier 2 pipeline ***
    Retained for V-series experiments (V1, V2, V4, V5) which compare CLIP
    against LLM baselines. CLIP was found to be unreliable on 320×240
    ESP32 frames: all 5 scene-label scores within ±0.013 of the 0.200
    uniform baseline — effectively random. This failure result motivates
    the LLM cognitive authority claim in the thesis.

    In production, call enhanced_yolo_infer(use_clip=False) and pass
    clip_model=None, preprocess=None, tokenizer=None.
    Ref: [45] Lugaresi et al. 2019 (MediaPipe) — EfficientDet-Lite0 replaces
         CLIP as the local person-detection emergency trigger.
    """
    try:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        print("[CLIP] ViT-B-32 (laion2b) loaded")
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"[CLIP] Load failed: {e} — CLIP disabled")
        return None, None, None


def clip_screen(clip_model, preprocess, tokenizer, img_bgr: np.ndarray):
    """
    CLIP zero-shot scene classification.
    Returns (scene_label, clip_risk, confidence).
    """
    if clip_model is None:
        return "clip_unavailable", "unknown", 0.0

    import torch
    from PIL import Image as PILImage

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = PILImage.fromarray(img_rgb)
    img_input = preprocess(pil_img).unsqueeze(0)
    txt_input = tokenizer(CLIP_SCENE_LABELS)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_input)
        txt_feat = clip_model.encode_text(txt_input)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (img_feat @ txt_feat.T).squeeze(0).softmax(dim=-1)

    best_idx   = probs.argmax().item()
    best_conf  = probs[best_idx].item()
    best_label = CLIP_SCENE_LABELS[best_idx]

    if best_conf < CLIP_CONF_THRESHOLD:
        return f"uncertain ({best_label[:30]})", "unknown", round(best_conf, 4)

    clip_risk = CLIP_HAZARD_MAP.get(best_label, "unknown")
    return best_label, clip_risk, round(best_conf, 4)


# ════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 5: DepthAnything v2 Metric Indoor
# ════════════════════════════════════════════════════════════════════════════

def load_depth_anything():
    """
    Load DepthAnything v2 Metric Indoor (Small) via HuggingFace Transformers.
    Model: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    Returns real metric depth in metres per pixel.
    Auto-downloads on first call (~100 MB for Small variant).
    Ref: Yang et al. NeurIPS 2024 [36]; Ahmmad et al. 2025 [35] (MAE=7.2cm).

    Returns (pipe, "depth-anything-v2-metric") or (None, "unavailable").
    """
    try:
        from transformers import pipeline as hf_pipeline
        print("[DA v2] Loading Depth Anything V2 Metric Indoor (Small)…")
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[DA v2] Using device: {device}")
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            device=device,
        )
        print("[DA v2] Depth Anything V2 Metric Indoor loaded ✓")
        return pipe, "depth-anything-v2-metric"
    except Exception as e:
        print(f"[DA v2] Load failed: {e}")
        print("       Install: pip install transformers pillow")
        print("       Falling back to geometric heuristic (est_dist = h/bbox_h × 0.3)")
        return None, "unavailable"


def _depth_at_bbox(depth_map: np.ndarray, x1: float, y1: float,
                   x2: float, y2: float) -> float:
    """
    Sample median depth in a 5×5 patch at the bounding-box centre.
    Median is more robust than a single centre pixel.
    """
    h, w = depth_map.shape[:2]
    cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
    r  = 2  # patch radius
    patch = depth_map[max(0, cy-r):cy+r+1, max(0, cx-r):cx+r+1]
    return float(np.median(patch)) if patch.size > 0 else float(depth_map[cy, cx])


def _heuristic_dist(img_h: int, y1: float, y2: float) -> float:
    """Fallback geometric heuristic when DA v2 is unavailable."""
    return round(img_h / max(y2 - y1, 1) * 0.3, 2)


def run_depth_estimation(depth_pipe, img_bgr: np.ndarray):
    """
    Run DepthAnything v2 Metric Indoor on a BGR numpy image.
    Returns depth_map (H×W float32, real metres) or None on failure.

    Important: HuggingFace pipeline returns two outputs:
      result["depth"]           — PIL Image scaled to 0-255 (visualization only)
      result["predicted_depth"] — torch.Tensor with real metric depth in metres
    We use predicted_depth for actual distances.
    """
    if depth_pipe is None:
        return None
    try:
        import torch
        from PIL import Image as PILImage
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        result  = depth_pipe(pil_img)
        # Use predicted_depth — real metric values in metres (not 0-255 scaled)
        depth_tensor = result["predicted_depth"]   # torch.Tensor [1,H,W] or [H,W]
        depth_map    = depth_tensor.squeeze().numpy().astype(np.float32)
        return depth_map
    except Exception as e:
        print(f"[DA v2] Inference error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# DUAL-YOLO INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _iou(box1, box2) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def _run_single_yolo(model, img: np.ndarray, conf: float = COCO_CONF) -> list[dict]:
    """Run one YOLO model, return list of {label, conf, x1,y1,x2,y2}."""
    if model is None:
        return []
    results = model(img, verbose=False, conf=conf)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        dets.append({
            "label": results.names[int(box.cls[0])],
            "conf":  round(float(box.conf[0]), 2),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return dets


def _merge_detections(coco_dets: list[dict], world_dets: list[dict],
                      iou_threshold: float = 0.5) -> list[dict]:
    """
    Merge COCO and YOLO-World detections.
    COCO wins if boxes overlap (higher recall, trained model).
    YOLO-World structural classes are appended without duplicate check
    since they target different object types.
    """
    merged = list(coco_dets)
    for wd in world_dets:
        # Skip if YOLO-World detected a person (COCO does this better)
        if wd["label"] == "person":
            continue
        # Skip if high IoU with any existing detection
        wb = (wd["x1"], wd["y1"], wd["x2"], wd["y2"])
        duplicate = any(
            _iou(wb, (d["x1"], d["y1"], d["x2"], d["y2"])) > iou_threshold
            for d in merged
        )
        if not duplicate:
            merged.append(wd)
    return merged


# ════════════════════════════════════════════════════════════════════════════
# COMBINED ENHANCED INFERENCE — public API
# ════════════════════════════════════════════════════════════════════════════

def enhanced_yolo_infer(
    yolo_model,           # YOLO-World model (structural classes)
    yolo_type: str,
    clip_model,
    preprocess,
    tokenizer,
    jpeg: bytes,
    coco_model=None,      # YOLOv11n COCO model (person + 79 classes)  [NEW]
    depth_pipe=None,      # DepthAnything v2 Metric Indoor pipeline     [NEW]
    use_clip: bool = False,  # CLIP disabled by default in production pipeline
) -> dict:
    """
    Full enhanced Tier 2 pipeline (Techniques 1–5).

    Pipeline:
        Frame → CLAHE → {YOLOv11n COCO, YOLO-World} → merge dets
                      → DA v2 Metric depth → sample depth per bbox
                      → CLIP scene screen (experimental only, disabled by default)
        → metadata dict → LLM (Tier 3, cognitive authority)

    Production use (use_clip=False, the default):
        CLIP is bypassed; clip_label="disabled", clip_risk="disabled", clip_conf=0.0.
        Pass clip_model=None, preprocess=None, tokenizer=None.

    Experimental use (use_clip=True):
        CLIP runs and its fields are populated. Used for V-series experiments
        (V1, V2, V4, V5) that compare CLIP against LLM-only baselines.
        Note: CLIP proved unreliable on 320×240 ESP32 frames (scores within
        ±0.013 of the 0.200 uniform-baseline). Kept for thesis evidence.

    Returns dict:
        yolo_meta  : formatted detection string for LLM prompt
        yolo_ms    : YOLO inference time (ms)
        yolo_type  : model identifier string
        clip_label : CLIP scene classification (or "disabled" if use_clip=False)
        clip_risk  : inferred risk from CLIP (or "disabled" if use_clip=False)
        clip_conf  : CLIP confidence (or 0.0 if use_clip=False)
        img_h      : image height (for downstream use)
        depth_available : bool — True if DA v2 ran, False if heuristic used
    """
    img_raw = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    img     = apply_clahe(img_raw)    # Technique 2: CLAHE
    h       = img.shape[0]

    # ── Technique 3: CLIP scene screen (experimental only) ──────────────────
    # CLIP is disabled in production (use_clip=False by default).
    # Results from CLIP experiments show scores within ±0.013 of the 0.200
    # uniform baseline on 320×240 ESP32 frames — effectively random.
    # These results are retained as thesis evidence motivating the LLM layer.
    # Ref: [45] Lugaresi et al. 2019; V-series experiment files.
    if use_clip:
        clip_label, clip_risk, clip_conf = clip_screen(
            clip_model, preprocess, tokenizer, img
        )
    else:
        clip_label, clip_risk, clip_conf = "disabled", "disabled", 0.0

    # ── Technique 5: Depth estimation (runs once per frame) ─────────────────
    t_depth_start = time.perf_counter()
    depth_map     = run_depth_estimation(depth_pipe, img)
    depth_ms      = (time.perf_counter() - t_depth_start) * 1000.0
    depth_available = depth_map is not None

    # ── Techniques 1 + 4: Dual-YOLO inference ───────────────────────────────
    if yolo_model is None and coco_model is None:
        # Full simulation mode
        return {
            "yolo_meta":       "YOLO detections: none",
            "yolo_ms":         round(abs(np.random.normal(20, 5)), 2),
            "yolo_type":       yolo_type,
            "clip_label":      clip_label,   # "disabled" if use_clip=False
            "clip_risk":       clip_risk,
            "clip_conf":       clip_conf,
            "img_h":           h,
            "depth_available": False,
        }

    # ── Technique 6: Lowered conf thresholds (no upscaling) ────────────────
    # YOLO letterboxes all inputs to 640×640 internally regardless of source.
    # Pre-upscaling 320×240→960×720 was tested and found to REDUCE person
    # detection recall (YOLO's 640→640 letterbox path is better than
    # 960→640 downscale path). Native resolution + lower conf is optimal.
    # COCO_CONF=0.20, WORLD_CONF=0.15 — both benchmarked on ESP32 frames.
    t0 = time.perf_counter()

    # Run COCO YOLOv11n (person + 79 classes) — primary
    coco_dets = _run_single_yolo(coco_model, img, conf=COCO_CONF)

    # Run YOLO-World (structural classes: wall, door, wire…) — supplement
    world_dets = _run_single_yolo(yolo_model, img, conf=WORLD_CONF)

    yolo_ms = (time.perf_counter() - t0) * 1000.0

    # Merge: COCO wins on overlap; YOLO-World adds structural objects
    all_dets = _merge_detections(coco_dets, world_dets)

    # ── Format detection metadata with real depth (or heuristic fallback) ───
    dets_str = []
    for d in all_dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]

        if depth_available:
            dist = round(_depth_at_bbox(depth_map, x1, y1, x2, y2), 2)
            dist_str = f"depth_m={dist}m [DA v2]"
        else:
            dist = _heuristic_dist(h, y1, y2)
            dist_str = f"est_dist~{dist}m [heuristic]"

        src = "COCO" if d in coco_dets else "YOLO-W"
        dets_str.append(
            f"{d['label']} (conf={d['conf']}, {dist_str}, "
            f"src={src}, bbox=[{x1},{y1},{x2},{y2}])"
        )

    yolo_meta = ("YOLO detections: " + "; ".join(dets_str)) if dets_str \
                else "YOLO detections: none"

    # Note which depth method was used
    depth_note = "DA v2 Metric Indoor" if depth_available else "geometric heuristic"

    return {
        "yolo_meta":       yolo_meta,
        "yolo_ms":         round(yolo_ms, 2),
        "depth_ms":        round(depth_ms, 2),
        "depth_note":      depth_note,
        "yolo_type":       yolo_type,
        "clip_label":      clip_label,
        "clip_risk":       clip_risk,
        "clip_conf":       clip_conf,
        "img_h":           h,
        "depth_available": depth_available,
    }


# ════════════════════════════════════════════════════════════════════════════
# RULE-BASED RISK (yolo_enhanced_only condition — no LLM)
# ════════════════════════════════════════════════════════════════════════════

def enhanced_rule_risk(yolo_meta: str, clip_risk: str) -> str:
    """
    Rule-based risk combining dual-YOLO detections and CLIP scene label.
    Used in G-series yolo_only condition.
    YOLO primary; CLIP fallback when YOLO returns none.

    Threat class handling (gun, pistol, explosive, etc.):
        Any detection from THREAT_RISK_CLASSES → immediate hazard, no distance check.
        YOLO-World zero-shot recall for threats on 320×240 is ~30–50%; the LLM
        (Tier 3) is the primary threat classifier — this rule is a best-effort
        supplementary signal. Validated via demo video.
    """
    meta = yolo_meta.lower()

    if "none" not in meta:
        # ── Threat classes → immediate hazard (no distance required) ─────────
        # gun, pistol, rifle, firearm, weapon, knife, explosive, bomb, grenade…
        for cls in THREAT_RISK_CLASSES:
            if cls in meta:
                print(f"[rule_risk] Threat detected: '{cls}' → hazard")
                return "hazard"

        # ── Hard stop structural hazards ──────────────────────────────────────
        for cls in ["pillar", "column", "barrier", "steps", "wire", "cable", "staircase"]:
            if cls in meta:
                return "hazard"
        if "wall" in meta:
            m = re.search(r'wall[^;]*conf=([\d.]+)', meta)
            if m and float(m.group(1)) > 0.40:
                return "hazard"

        # ── Person proximity check ────────────────────────────────────────────
        if "person" in meta:
            # Use DA v2 depth if available, else heuristic
            m_depth = re.search(r'person[^;]*depth_m=([\d.]+)m', meta)
            m_heur  = re.search(r'person[^;]*est_dist~([\d.]+)m', meta)
            dist = None
            if m_depth:
                dist = float(m_depth.group(1))
            elif m_heur:
                dist = float(m_heur.group(1))
            if dist is not None and dist < 1.0:
                return "hazard"
            return "caution"

        # Navigable openings (door, window) are safe
        detected = re.findall(r'(\w+)\s*\(conf=', meta)
        if detected and all(d in SAFE_DETECTION_CLASSES for d in detected):
            return "safe"
        return "caution"

    # YOLO found nothing — trust CLIP
    if clip_risk in ("hazard", "caution"):
        return clip_risk
    return "safe"


# ════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS (Tier 3 — cognitive authority)
# ════════════════════════════════════════════════════════════════════════════

LLM_ONLY_PROMPT = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors.\n"
    "No sensor metadata available — your ONLY input is the camera image. "
    "Look at it directly and reason from what you see.\n\n"
    "RISK CLASSIFICATION GUIDE:\n"
    "  hazard : Person very close (<1m) OR obstacle directly in ~1m flight path "
    "within 2m OR lens fully covered. → Stop immediately.\n"
    "  caution: Dim/dark scene (not fully black), OR background clutter at floor/"
    "table level not blocking the 1m corridor. → Slow down or hover.\n"
    "  safe   : Flight path at ~1m is clear. Open rooms, navigable doorways/windows, "
    "objects on tables/shelves, persons visually >3m away. → Proceed.\n\n"
    "Pilot action options:\n"
    "  HOVER | PITCH_FORWARD | PITCH_BACK | ROLL_LEFT | ROLL_RIGHT |\n"
    "  YAW_LEFT | YAW_RIGHT | ASCEND | DESCEND | LAND\n\n"
    "Respond in this exact format:\n"
    "Description: <what you see — 1-2 sentences>\n"
    "Proximity: <your visual estimate of closest object/person>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <command>"
)

# ── Production prompt (no CLIP — default) ───────────────────────────────────
# Use this for all production runs. CLIP is removed from Tier 2 because it
# produced near-random scores on 320×240 ESP32 frames (±0.013 of 0.200 baseline).
# Fill only: {yolo_meta}
COMBINED_PROMPT_TEMPLATE = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "operating at ~1m altitude indoors.\n\n"
    "YOUR ROLE IN THE ARCHITECTURE:\n"
    "  Tier 1 (reflexes)  : PID controller — motor corrections at 4 kHz.\n"
    "  Tier 2 (perception): YOLOv11n (COCO) + YOLO-World + DepthAnything v2 —\n"
    "                       fast object detection with metric depth; passes metadata to you.\n"
    "                       Emergency triggers: robust_local_detector (14 ms, no API).\n"
    "  Tier 3 (YOU)       : Cognitive reasoning — analyse the image, form your own\n"
    "                       judgment, make the final safety decision.\n\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE. Look at it directly and reason from it.\n\n"
    "Supplementary sensor data from Tier 2 (advisory only — you are the authority):\n"
    "  {yolo_meta}\n"
    "  (depth_m = DepthAnything v2 Metric Indoor, MAE≈7cm; est_dist = geometric fallback)\n\n"
    "SENSOR INTERPRETATION RULES:\n"
    "- depth_m values are from DepthAnything v2 Metric Indoor (real metres, MAE ≈7 cm).\n"
    "  If unavailable, est_dist~Xm is a bounding-box heuristic — treat with lower trust.\n"
    "- YOLO can MISS objects — if you see something the sensors did not detect, it exists.\n"
    "  Report it and classify based on what you see.\n"
    "- src=COCO means detected by trained YOLOv11n (high confidence).\n"
    "  src=YOLO-W means zero-shot YOLO-World (structural class, lower recall).\n"
    "- If YOLO detections = none: do NOT assume safe. Look at the image yourself.\n"
    "- Local detector (Tier 1.5) can false-positive or miss — treat as a hint, not a command.\n"
    "  YOUR VISUAL ANALYSIS OF THE IMAGE overrides all sensor data including local detector.\n\n"
    "RISK CLASSIFICATION (your visual analysis is primary evidence):\n"
    "  hazard : Person very close (<1m) OR obstacle directly blocks ~1m flight path\n"
    "           within ~2m OR lens fully covered. → Stop immediately.\n"
    "  caution: Dim/dark scene (not fully black), OR clutter at floor/table level not\n"
    "           blocking 1m corridor. → Slow down or hover.\n"
    "  safe   : Flight path at ~1m is clear. Open rooms, navigable doors/windows,\n"
    "           objects on tables/shelves, persons visually >3m away. → Proceed.\n\n"
    "Pilot action options:\n"
    "  HOVER | PITCH_FORWARD | PITCH_BACK | ROLL_LEFT | ROLL_RIGHT |\n"
    "  YAW_LEFT | YAW_RIGHT | ASCEND | DESCEND | LAND\n\n"
    "RESPOND in this exact format:\n"
    "Description: <1-2 sentences — your own visual analysis of the image>\n"
    "Sensor note: <discrepancy between image and sensors (YOLO/local detector), or 'consistent'>\n"
    "Proximity: <your visual estimate of closest object/person, cross-check with depth_m>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <command>"
)

# ── Experimental prompt (CLIP included) — for V-series experiments only ──────
# Fill: {yolo_meta}, {clip_label}, {clip_conf}, {clip_risk}
# Use enhanced_yolo_infer(use_clip=True) to populate these fields.
COMBINED_PROMPT_TEMPLATE_CLIP = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "operating at ~1m altitude indoors.\n\n"
    "YOUR ROLE IN THE ARCHITECTURE:\n"
    "  Tier 1 (reflexes)  : PID controller — motor corrections at 4 kHz.\n"
    "  Tier 2 (perception): YOLOv11n (COCO) + YOLO-World + DepthAnything v2 + CLIP —\n"
    "                       fast detection, metric depth, scene label; passes all to you.\n"
    "  Tier 3 (YOU)       : Cognitive reasoning — analyse the image, form your own\n"
    "                       judgment, make the final safety decision.\n\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE. Look at it directly and reason from it.\n\n"
    "Supplementary sensor data from Tier 2 (advisory only — you are the authority):\n"
    "  {yolo_meta}\n"
    "  (depth_m = DepthAnything v2 Metric Indoor, MAE≈7cm; est_dist = geometric fallback)\n"
    "  CLIP scene label: {clip_label} (confidence={clip_conf}, inferred risk={clip_risk})\n\n"
    "SENSOR INTERPRETATION RULES:\n"
    "- depth_m values are from DepthAnything v2 Metric Indoor (real metres, MAE ≈7 cm).\n"
    "  If unavailable, est_dist~Xm is a bounding-box heuristic — treat with lower trust.\n"
    "- YOLO can MISS objects — if you see something the sensors did not detect, it exists.\n"
    "  Report it and classify based on what you see.\n"
    "- src=COCO means detected by trained YOLOv11n (high confidence).\n"
    "  src=YOLO-W means zero-shot YOLO-World (structural class, lower recall).\n"
    "- CLIP describes scene type — do not let it override your corridor-level judgment.\n"
    "  Note: CLIP was found unreliable on 320×240 frames (scores ≈ uniform baseline).\n"
    "- Local detector (Tier 1.5) can false-positive or miss — treat as a hint, not a command.\n"
    "  YOUR VISUAL ANALYSIS OF THE IMAGE overrides all sensor data including local detector.\n"
    "- If YOLO detections = none: do NOT assume safe. Look at the image yourself.\n\n"
    "RISK CLASSIFICATION (your visual analysis is primary evidence):\n"
    "  hazard : Person very close (<1m) OR obstacle directly blocks ~1m flight path\n"
    "           within ~2m OR lens fully covered. → Stop immediately.\n"
    "  caution: Dim/dark scene (not fully black), OR clutter at floor/table level not\n"
    "           blocking 1m corridor. → Slow down or hover.\n"
    "  safe   : Flight path at ~1m is clear. Open rooms, navigable doors/windows,\n"
    "           objects on tables/shelves, persons visually >3m away. → Proceed.\n\n"
    "Pilot action options:\n"
    "  HOVER | PITCH_FORWARD | PITCH_BACK | ROLL_LEFT | ROLL_RIGHT |\n"
    "  YAW_LEFT | YAW_RIGHT | ASCEND | DESCEND | LAND\n\n"
    "RESPOND in this exact format:\n"
    "Description: <1-2 sentences — your own visual analysis of the image>\n"
    "Sensor note: <discrepancy between image and sensors (YOLO/CLIP/local detector), or 'consistent'>\n"
    "Proximity: <your visual estimate of closest object/person, cross-check with depth_m>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <command>"
)
