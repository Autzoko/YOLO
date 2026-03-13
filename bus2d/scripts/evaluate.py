#!/usr/bin/env python3
"""
Step 6 — Comprehensive evaluation of the trained BUS-2D model.

Produces:
  - AP@0.5 and AP@0.5:0.95 per class
  - PR curve per class
  - Confusion matrix
  - Confidence threshold sweep (precision/recall/F1)
  - Per-volume recall (flag volumes with zero detections)

All outputs saved to runs/bus2d/exp1/eval/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def resolve_env(s, env=None):
    if env is None:
        env = {}
    if not isinstance(s, str):
        return s
    while "${" in s:
        start = s.index("${")
        end = s.index("}", start)
        key = s[start + 2 : end]
        val = env.get(key, os.environ.get(key, ""))
        s = s[:start] + str(val) + s[end + 1 :]
    return s


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    env = {"data_root": cfg["data_root"], "output_root": cfg["output_root"]}

    def _resolve(obj):
        if isinstance(obj, str):
            return resolve_env(obj, env)
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    return _resolve(cfg)


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def xywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_labels(label_dir, img_dir):
    """Load all ground truth labels. Returns dict: filename -> list of (class_id, xyxy)."""
    gt = {}
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        img_path = os.path.join(img_dir, f"{stem}.png")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        boxes = []
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                xyxy = xywh_to_xyxy(cx, cy, w, h, img_w, img_h)
                boxes.append((cls, xyxy))
        gt[stem] = boxes
    return gt


def compute_ap(precision, recall):
    """Compute AP using 101-point interpolation (COCO style)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 101-point interpolation
    recall_pts = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_pts:
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            ap += mpre[idx[0]]
    ap /= 101.0
    return ap


def evaluate_at_iou(gt_all, preds_all, iou_thresh, class_names):
    """Evaluate detections at a specific IoU threshold. Returns per-class (AP, precision_array, recall_array).
    """
    n_classes = len(class_names)
    results = {}

    for cls_id in range(n_classes):
        # Collect all predictions for this class, sorted by confidence
        all_preds = []
        n_gt = 0
        gt_matched = {}  # image -> set of matched gt indices

        for img_name in set(list(gt_all.keys()) + list(preds_all.keys())):
            gt_boxes = [(c, b) for c, b in gt_all.get(img_name, []) if c == cls_id]
            pred_boxes = [(conf, b) for c, conf, b in preds_all.get(img_name, []) if c == cls_id]
            n_gt += len(gt_boxes)
            gt_matched[img_name] = set()

            for conf, pb in pred_boxes:
                all_preds.append((conf, img_name, pb, gt_boxes))

        # Sort by confidence descending
        all_preds.sort(key=lambda x: -x[0])

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        matched_per_img = defaultdict(set)

        for i, (conf, img_name, pb, gt_boxes) in enumerate(all_preds):
            best_iou = 0
            best_j = -1
            for j, (_, gb) in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j not in matched_per_img[img_name]:
                tp[i] = 1
                matched_per_img[img_name].add(best_j)
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        recall = cum_tp / (n_gt + 1e-10) if n_gt > 0 else cum_tp * 0

        ap = compute_ap(precision, recall) if n_gt > 0 else 0.0

        results[cls_id] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "n_gt": n_gt,
            "n_pred": len(all_preds),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BUS-2D model")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    class_map = cfg["classes"]
    class_names = {int(k): v[1] for k, v in class_map.items()}
    n_classes = len(class_names)

    eval_dir = eval_cfg["output_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    slices_root = cfg["slices_root"]
    img_dir = os.path.join(slices_root, "images", args.split)
    lbl_dir = os.path.join(slices_root, "labels", args.split)
    meta_dir = os.path.join(slices_root, "metadata", args.split)

    print("=" * 60)
    print("BUS-2D Evaluation")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Split: {args.split}")
    print(f"Image dir: {img_dir}")
    print(f"Output dir: {eval_dir}")

    # Load GT
    print("\nLoading ground truth...")
    gt_all = load_gt_labels(lbl_dir, img_dir)
    n_images = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
    n_gt_boxes = sum(len(v) for v in gt_all.values())
    print(f"  Images: {n_images}")
    print(f"  GT boxes: {n_gt_boxes}")

    if args.dry_run:
        print("\n[DRY RUN] Would run inference on all images, compute metrics, save plots.")
        return

    # Run inference
    print("\nRunning inference...")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    preds_all = {}  # img_name -> list of (class_id, confidence, xyxy)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    batch_size = 32

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i : i + batch_size]
        batch_paths = [os.path.join(img_dir, f) for f in batch_files]
        results = model.predict(batch_paths, conf=0.01, iou=0.5, verbose=False)

        for fname, result in zip(batch_files, results):
            stem = fname[:-4]
            boxes = result.boxes
            preds = []
            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    cls = int(boxes.cls[j].item())
                    conf = float(boxes.conf[j].item())
                    xyxy = boxes.xyxy[j].cpu().numpy().tolist()
                    preds.append((cls, conf, xyxy))
            preds_all[stem] = preds

    print(f"  Predictions: {sum(len(v) for v in preds_all.values())} boxes")

    # ── AP@0.5 and AP@0.5:0.95 per class ───────────────────────────
    print("\nComputing AP metrics...")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    ap50_results = evaluate_at_iou(gt_all, preds_all, 0.5, class_names)
    ap_50_95 = {}
    for cls_id in range(n_classes):
        aps = []
        for iou_t in iou_thresholds:
            r = evaluate_at_iou(gt_all, preds_all, iou_t, class_names)
            aps.append(r[cls_id]["ap"])
        ap_50_95[cls_id] = np.mean(aps)

    print(f"\n{'Class':<12} {'AP@0.5':>8} {'AP@0.5:0.95':>12} {'N_GT':>6}")
    print("-" * 42)
    for cid in range(n_classes):
        name = class_names[cid]
        ap50 = ap50_results[cid]["ap"]
        ap5095 = ap_50_95[cid]
        ngt = ap50_results[cid]["n_gt"]
        print(f"{name:<12} {ap50:>8.3f} {ap5095:>12.3f} {ngt:>6}")
    mAP50 = np.mean([ap50_results[c]["ap"] for c in range(n_classes)])
    mAP5095 = np.mean([ap_50_95[c] for c in range(n_classes)])
    print(f"{'mAP':<12} {mAP50:>8.3f} {mAP5095:>12.3f}")

    # ── PR curves per class ─────────────────────────────────────────
    print("\nGenerating PR curves...")
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    for cid in range(n_classes):
        ax = axes[cid]
        r = ap50_results[cid]
        if len(r["recall"]) > 0:
            ax.plot(r["recall"], r["precision"], linewidth=2)
        ax.set_title(f"{class_names[cid]} (AP={r['ap']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(eval_dir, "pr_curves.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"  Saved: {pr_path}")

    # ── Confusion matrix ────────────────────────────────────────────
    print("Generating confusion matrix...")
    # Simple confusion: for each GT box, find best matching prediction
    conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background

    for img_name, gt_boxes in gt_all.items():
        pred_boxes = preds_all.get(img_name, [])
        matched_pred = set()

        for gt_cls, gt_xyxy in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            for pi, (p_cls, p_conf, p_xyxy) in enumerate(pred_boxes):
                if p_conf < 0.25:
                    continue
                iou = compute_iou(gt_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pi

            if best_iou >= 0.5 and best_pred_idx >= 0:
                pred_cls = pred_boxes[best_pred_idx][0]
                conf_matrix[gt_cls, pred_cls] += 1
                matched_pred.add(best_pred_idx)
            else:
                conf_matrix[gt_cls, n_classes] += 1  # FN (background)

        # Unmatched predictions → FP
        for pi, (p_cls, p_conf, _) in enumerate(pred_boxes):
            if pi not in matched_pred and p_conf >= 0.25:
                conf_matrix[n_classes, p_cls] += 1

    labels = [class_names[i] for i in range(n_classes)] + ["BG"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks(range(n_classes + 1))
    ax.set_yticks(range(n_classes + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (IoU≥0.5, conf≥0.25)")
    for i in range(n_classes + 1):
        for j in range(n_classes + 1):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved: {cm_path}")

    # ── Confidence threshold sweep ──────────────────────────────────
    print("Running confidence threshold sweep...")
    conf_start = eval_cfg["conf_thresholds"]["start"]
    conf_stop = eval_cfg["conf_thresholds"]["stop"]
    conf_step = eval_cfg["conf_thresholds"]["step"]
    thresholds = np.arange(conf_start, conf_stop + conf_step / 2, conf_step)

    sweep_results = []
    for thresh in thresholds:
        tp_total, fp_total, fn_total = 0, 0, 0
        for img_name, gt_boxes in gt_all.items():
            preds = [(c, conf, b) for c, conf, b in preds_all.get(img_name, []) if conf >= thresh]
            matched = set()
            for gt_cls, gt_xyxy in gt_boxes:
                found = False
                for pi, (p_cls, p_conf, p_xyxy) in enumerate(preds):
                    if pi in matched:
                        continue
                    if compute_iou(gt_xyxy, p_xyxy) >= 0.5 and p_cls == gt_cls:
                        tp_total += 1
                        matched.add(pi)
                        found = True
                        break
                if not found:
                    fn_total += 1
            fp_total += len(preds) - len(matched)

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        sweep_results.append(
            {
                "threshold": round(float(thresh), 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )

    # Plot sweep
    _fig, ax = plt.subplots(figsize=(10, 6))
    ts = [r["threshold"] for r in sweep_results]
    ax.plot(ts, [r["precision"] for r in sweep_results], label="Precision", marker="o", ms=3)
    ax.plot(ts, [r["recall"] for r in sweep_results], label="Recall", marker="s", ms=3)
    ax.plot(ts, [r["f1"] for r in sweep_results], label="F1", marker="^", ms=3, linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Confidence Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sweep_path = os.path.join(eval_dir, "conf_sweep.png")
    plt.savefig(sweep_path, dpi=150)
    plt.close()
    print(f"  Saved: {sweep_path}")

    # Best F1
    best = max(sweep_results, key=lambda x: x["f1"])
    print(
        f"  Best F1={best['f1']:.4f} at conf={best['threshold']:.2f} "
        f"(P={best['precision']:.4f}, R={best['recall']:.4f})"
    )

    # ── Per-volume recall ───────────────────────────────────────────
    print("\nComputing per-volume recall...")
    vol_stats = defaultdict(lambda: {"n_gt": 0, "n_det": 0})

    for img_name, gt_boxes in gt_all.items():
        # Get volume_id from metadata
        meta_path = os.path.join(meta_dir, f"{img_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            vid = meta.get("volume_id", "unknown")
        else:
            vid = img_name.rsplit("_y", 1)[0]  # fallback

        vol_stats[vid]["n_gt"] += len(gt_boxes)
        # Count detections at best F1 threshold
        preds = [(c, conf, b) for c, conf, b in preds_all.get(img_name, []) if conf >= best["threshold"]]
        for gt_cls, gt_xyxy in gt_boxes:
            for p_cls, p_conf, p_xyxy in preds:
                if compute_iou(gt_xyxy, p_xyxy) >= 0.5:
                    vol_stats[vid]["n_det"] += 1
                    break

    zero_det_vols = []
    vol_recall_data = []
    for vid, stats in sorted(vol_stats.items()):
        if stats["n_gt"] > 0:
            recall = stats["n_det"] / stats["n_gt"]
            vol_recall_data.append(
                {"volume_id": vid, "recall": round(recall, 4), "n_gt": stats["n_gt"], "n_det": stats["n_det"]}
            )
            if stats["n_det"] == 0:
                zero_det_vols.append(vid)

    if zero_det_vols:
        print(f"  WARNING: {len(zero_det_vols)} volumes with ZERO detections:")
        for v in zero_det_vols:
            print(f"    - {v}")
    else:
        print("  All volumes have at least one detection.")

    # ── Save all results ────────────────────────────────────────────
    eval_results = {
        "ap_per_class": {
            class_names[c]: {
                "AP@0.5": round(ap50_results[c]["ap"], 4),
                "AP@0.5:0.95": round(ap_50_95[c], 4),
                "n_gt": ap50_results[c]["n_gt"],
            }
            for c in range(n_classes)
        },
        "mAP@0.5": round(mAP50, 4),
        "mAP@0.5:0.95": round(mAP5095, 4),
        "conf_sweep": sweep_results,
        "best_f1": best,
        "per_volume_recall": vol_recall_data,
        "zero_detection_volumes": zero_det_vols,
    }

    results_path = os.path.join(eval_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nAll results saved to: {eval_dir}/")
    print("  eval_results.json  — metrics JSON")
    print("  pr_curves.png      — PR curves")
    print("  confusion_matrix.png — confusion matrix")
    print("  conf_sweep.png     — threshold sweep")


if __name__ == "__main__":
    main()
