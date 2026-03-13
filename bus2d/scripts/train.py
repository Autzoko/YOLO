#!/usr/bin/env python3
"""
Step 5 — Launch YOLO training (local or called from SLURM).

Uses the ultralytics Python API.  Multi-GPU is handled automatically
when device=[0,1,2,3] (ultralytics launches DDP internally).
"""

import argparse
import os
import sys

import yaml

# Add ultralytics repo root to path
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


def main():
    parser = argparse.ArgumentParser(description="Train BUS-2D YOLO model")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument("--dataset-yaml", default=None, help="Override dataset YAML path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]

    dataset_yaml = args.dataset_yaml or os.path.join(os.path.dirname(__file__), "..", "bus2d_dataset.yaml")

    from ultralytics import YOLO

    # Load pretrained model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading pretrained: {tcfg['model']}")
        model = YOLO(tcfg["model"])

    # Compute class weights (inverse frequency) from split_info if available
    split_info_path = os.path.join(cfg["slices_root"], "split_info.json")
    cls_weights = None
    if os.path.exists(split_info_path):
        import json

        with open(split_info_path) as f:
            split_info = json.load(f)
        train_dist = split_info.get("train_class_dist", {})
        # Only count positive classes (exclude -1 = negative)
        pos_counts = {int(k): v for k, v in train_dist.items() if int(k) >= 0}
        if pos_counts:
            total = sum(pos_counts.values())
            n_classes = len(pos_counts)
            cls_weights = []
            for cid in range(5):
                cnt = pos_counts.get(cid, 1)
                # Inverse frequency weighting, normalized
                w = total / (n_classes * cnt)
                cls_weights.append(round(w, 3))
            print(f"Class weights (inverse freq): {cls_weights}")

    # Train
    print(f"\nDataset YAML: {dataset_yaml}")
    print(f"Model: {tcfg['model']}")
    print(f"Image size: {tcfg['imgsz']}")
    print(f"Batch: {tcfg['batch']}")
    print(f"Epochs: {tcfg['epochs']}")
    print(f"Device: {tcfg['device']}")

    train_kwargs = dict(
        data=dataset_yaml,
        imgsz=tcfg["imgsz"],
        batch=tcfg["batch"],
        epochs=tcfg["epochs"],
        patience=tcfg["patience"],
        workers=tcfg["workers"],
        device=tcfg["device"],
        project=tcfg["project"],
        name=tcfg["name"],
        exist_ok=True,
        save=True,
        val=True,
        # Augmentation — most handled by YOLO internally
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
    )

    # Pass class weights via cls loss scaling if available
    # YOLO doesn't natively support per-class weights, but we can boost cls loss
    # For rare classes (4b, 4c), the effective approach is copy-paste augmentation
    if cls_weights:
        # Use copy_paste augmentation to help rare classes
        train_kwargs["copy_paste"] = 0.3

    results = model.train(**train_kwargs)
    print("\nTraining complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
