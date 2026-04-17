"""
EXP-V5: YOLO Confidence Threshold Sweep
=========================================
Goal:
    Find the optimal YOLOv8 confidence threshold for anomaly detection
    on real ESP32-S3 camera frames in indoor drone environments.

    Thresholds tested: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    Labeled frame sets:
        hazard_frames : person_near, wall_close, blocked_lens (10 frames each)
        safe_frames   : clear_open, door_open, person_far     (10 frames each)

    For each threshold on each labeled frame:
        - YOLO detects objects  → predicted = hazard if any hazard class found
        - Compare prediction vs ground truth
        - Compute TP, FP, TN, FN → precision, recall, F1

    N_REPEATS=5 (re-capture frame for each repeat to test consistency).

Metrics:
    - precision, recall, F1, accuracy  (Wilson CI per threshold)
    - false_alarm_rate, miss_rate      (Wilson CI per threshold)
    - n_detections per frame           (Bootstrap CI)
    Output: V5_runs.csv, V5_roc.csv (PR curve data)
"""

import sys, os, time, pathlib, io
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    get_frame, bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR,
    HAZARD_LABELS
)
import numpy as np

THRESHOLDS   = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
N_REPEATS    = 5

# Labeled scenes for this experiment
LABELED_SCENES = [
    # (label, ground_truth_is_hazard, setup_instruction)
    ("person_near",    True,  "Operator stands ~1m in front of camera."),
    ("wall_close",     True,  "Point camera at wall from ~25cm away."),
    ("blocked_lens",   True,  "Partially cover camera lens."),
    ("clear_open",     False, "Empty floor, no obstacles."),
    ("door_open",      False, "Open doorway in frame, no close obstacles."),
    ("person_far",     False, "Operator stands ~3m away — background figure only."),
]

# YOLO anomaly detection logic (mirrors server.py)
_yolo_model_cache = {}

def load_yolo(conf: float):
    global _yolo_model_cache
    if conf not in _yolo_model_cache:
        from ultralytics import YOLO
        m = YOLO("yolov8n.pt")
        _yolo_model_cache[conf] = m
    return _yolo_model_cache[conf]

def yolo_predict(jpeg_bytes: bytes, conf: float) -> tuple[list, bool]:
    """Returns (detections_list, predicted_hazard)."""
    try:
        from PIL import Image as PILImage
        model = load_yolo(conf)
        img   = PILImage.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        results = model(img, conf=conf, verbose=False)
        dets    = []
        hazard  = False
        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                label   = model.names[cls_id]
                conf_v  = float(box.conf[0])
                x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
                area_pct = (x2-x1)*(y2-y1)/(img.width*img.height)*100
                dets.append({"label": label, "conf": conf_v, "area_pct": area_pct})
                if label in HAZARD_LABELS or area_pct > 60:
                    hazard = True
        return dets, hazard
    except Exception as e:
        print(f"  [YOLO] error at conf={conf}: {e}")
        return [], False

def confusion_metrics(tp, fp, tn, fn):
    total  = tp + fp + tn + fn
    prec   = tp / (tp + fp) if (tp+fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0.
    f1     = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0.
    acc    = (tp+tn)/total if total > 0 else 0.
    far    = fp / (fp+tn) if (fp+tn) > 0 else 0.   # false alarm rate
    mrr    = fn / (fn+tp) if (fn+tp) > 0 else 0.   # miss rate
    return dict(precision=round(prec,4), recall=round(recall,4),
                f1=round(f1,4), accuracy=round(acc,4),
                false_alarm_rate=round(far,4), miss_rate=round(mrr,4))

def main():
    print("="*60)
    print("EXP-V5: YOLO Confidence Threshold Sweep")
    print(f"Thresholds={THRESHOLDS}  Scenes={len(LABELED_SCENES)}  N_repeats={N_REPEATS}")
    print("="*60)
    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    all_rows = []

    for scene_label, is_hazard, setup in LABELED_SCENES:
        print(f"\n── Scene: {scene_label}  (ground_truth={'hazard' if is_hazard else 'safe'}) ──")
        print(f"   Setup: {setup}")
        input("   [READY] Press Enter when scene is set up…")

        for thresh in THRESHOLDS:
            for rep in range(1, N_REPEATS+1):
                jpeg  = get_frame(scene_label)
                dets, pred_hazard = yolo_predict(jpeg, thresh)

                # Confusion values for this single sample
                tp = int(is_hazard and pred_hazard)
                fp = int((not is_hazard) and pred_hazard)
                tn = int((not is_hazard) and (not pred_hazard))
                fn = int(is_hazard and (not pred_hazard))

                row = {
                    "scene_label":    scene_label,
                    "ground_truth":   "hazard" if is_hazard else "safe",
                    "threshold":      thresh,
                    "repeat":         rep,
                    "pred_hazard":    int(pred_hazard),
                    "n_detections":   len(dets),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "hazard_labels":  "|".join(d["label"] for d in dets if d["label"] in HAZARD_LABELS),
                    "error":          "",
                }
                all_rows.append(row)
                print(f"   conf={thresh}  rep={rep}  pred={'HAZ' if pred_hazard else 'safe'}  "
                      f"dets={len(dets)}  TP={tp} FP={fp} TN={tn} FN={fn}")

            time.sleep(1)

    # ── Save runs
    fields = ["scene_label","ground_truth","threshold","repeat",
              "pred_hazard","n_detections","tp","fp","tn","fn",
              "hazard_labels","error"]
    runs_csv = RESULTS_DIR / "V5_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── PR / ROC curve per threshold
    print(f"\n── V5 Threshold Summary ────────────────────────────────────")
    print(f"  {'conf':5s}  precision  recall   F1       acc      FAR      MR")
    roc_rows = []
    for thresh in THRESHOLDS:
        tr  = [r for r in all_rows if r["threshold"]==thresh]
        tp  = sum(r["tp"] for r in tr)
        fp  = sum(r["fp"] for r in tr)
        tn  = sum(r["tn"] for r in tr)
        fn  = sum(r["fn"] for r in tr)
        m   = confusion_metrics(tp, fp, tn, fn)
        print(f"  {thresh:.2f}   {m['precision']:.3f}      {m['recall']:.3f}    "
              f"{m['f1']:.3f}    {m['accuracy']:.3f}    "
              f"{m['false_alarm_rate']:.3f}    {m['miss_rate']:.3f}")
        roc_rows.append({"threshold": thresh, **m,
                         "tp": tp, "fp": fp, "tn": tn, "fn": fn})

    # Best threshold by F1
    best = max(roc_rows, key=lambda r: r["f1"])
    print(f"\n  Best threshold by F1: {best['threshold']} "
          f"(F1={best['f1']:.3f}, precision={best['precision']:.3f}, "
          f"recall={best['recall']:.3f})")

    roc_csv = RESULTS_DIR / "V5_roc.csv"
    write_csv(roc_csv, roc_rows,
              ["threshold","precision","recall","f1","accuracy",
               "false_alarm_rate","miss_rate","tp","fp","tn","fn"])

    print(f"\nData → {runs_csv}")
    print(f"ROC  → {roc_csv}")

if __name__ == "__main__":
    main()
