#!/usr/bin/env python3
"""
One-off script: Regenerate ABUS slices only (corrected coronal axis 0).
Reads the existing manifest + split_info to maintain the same train/val assignment.
"""

import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import SimpleITK as sitk

# Reuse functions from the main pipeline
sys.path.insert(0, os.path.dirname(__file__))
from preprocess_extract import (
    augment_slice,
    bbox_from_mask_slice,
    load_config,
    normalize_volume,
    yolo_label_line,
)


def load_abus_volume(data_path):
    img = sitk.ReadImage(data_path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def load_abus_mask(mask_path):
    img = sitk.ReadImage(mask_path)
    return sitk.GetArrayFromImage(img).astype(np.uint8)


def extract_slices_abus_fixed(vol_info, volume, mask, cfg, split, rng):
    """Extract coronal slices along AXIS 0 (corrected)."""
    se_cfg = cfg["slice_extraction"]
    aug_cfg = cfg["augmentation"]
    results = []
    vol_id = vol_info["volume_id"]
    occupied_y_ranges = []

    for les in vol_info["lesions"]:
        y_min = les["y_min"]
        y_max = les["y_max"]
        occupied_y_ranges.append((y_min, y_max))

        # Middle third
        y_span = y_max - y_min
        mid_start = y_min + int(y_span / 3)
        mid_end = y_min + int(2 * y_span / 3)

        # Clamp to axis 0 (coronal sweep, dim ~330)
        mid_start = max(mid_start, 0)
        mid_end = min(mid_end, volume.shape[0] - 1)

        n_slices = mid_end - mid_start + 1
        n_slices = max(se_cfg["min_slices"], min(se_cfg["max_slices"], n_slices))
        if mid_end - mid_start + 1 > n_slices:
            indices = np.linspace(mid_start, mid_end, n_slices, dtype=int)
        else:
            indices = list(range(mid_start, mid_end + 1))

        for y_idx in indices:
            # CORRECTED: slice axis 0 → (axial, lateral) = (608, 865) image
            slice_2d = volume[y_idx, :, :]
            mask_2d = mask[y_idx, :, :]

            bb = bbox_from_mask_slice(mask_2d)
            label_lines = []
            if bb is not None:
                cx, cy, w, h = bb
                img_h, img_w = slice_2d.shape
                label_lines.append(
                    yolo_label_line(les["class_id"], cx, cy, w, h, img_w, img_h)
                )

            img = slice_2d if slice_2d.dtype == np.uint8 else slice_2d.astype(np.uint8)
            if split == "train":
                img = augment_slice(img, aug_cfg, rng)

            fname = f"{vol_id}_y{y_idx:04d}"
            meta = {
                "volume_id": vol_id,
                "dataset": "abus",
                "slice_index": int(y_idx),
                "class": les["class_id"],
                "birads": les["birads"],
            }
            results.append((fname, img, label_lines, meta))

    # Negative slices
    n_pos = len(results)
    n_neg_target = int(n_pos * se_cfg["negative_ratio"])
    buffer = se_cfg["negative_buffer"]

    all_y = set()
    for y_min, y_max in occupied_y_ranges:
        for y in range(max(0, y_min - buffer), min(volume.shape[0], y_max + buffer + 1)):
            all_y.add(y)
    valid_neg_y = [y for y in range(volume.shape[0]) if y not in all_y]

    if valid_neg_y and n_neg_target > 0:
        neg_indices = rng.choice(
            valid_neg_y,
            size=min(n_neg_target, len(valid_neg_y)),
            replace=len(valid_neg_y) < n_neg_target,
        )
        for y_idx in neg_indices:
            slice_2d = volume[y_idx, :, :]
            img = slice_2d if slice_2d.dtype == np.uint8 else slice_2d.astype(np.uint8)
            if split == "train":
                img = augment_slice(img, aug_cfg, rng)
            fname = f"{vol_id}_y{y_idx:04d}_neg"
            meta = {
                "volume_id": vol_id,
                "dataset": "abus",
                "slice_index": int(y_idx),
                "class": -1,
                "birads": "none",
            }
            results.append((fname, img, [], meta))

    return results


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    cfg = load_config(config_path)

    manifest_path = cfg["manifest_path"]
    slices_root = cfg["slices_root"]
    split_info_path = os.path.join(slices_root, "split_info.json")

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(split_info_path) as f:
        split_info = json.load(f)

    train_ids = set(split_info["train_volume_ids"])
    val_ids = set(split_info["val_volume_ids"])

    # Filter ABUS-only volumes
    abus_vols = [v for v in manifest["volumes"] if v["dataset"] == "abus"]
    print(f"ABUS volumes to regenerate: {len(abus_vols)}")

    preproc = cfg["preprocessing"]
    rng = np.random.RandomState(cfg["split"]["seed"])
    class_map = cfg["classes"]

    total_slices = 0
    class_counts = {"train": Counter(), "val": Counter()}

    for i, vol in enumerate(abus_vols):
        vid = vol["volume_id"]
        if vid in train_ids:
            split = "train"
        elif vid in val_ids:
            split = "val"
        else:
            print(f"  [SKIP] {vid}: not in train or val split")
            continue

        print(f"  [{i+1}/{len(abus_vols)}] {vid} → {split}")

        try:
            volume = load_abus_volume(vol["data_path"])
            mask = load_abus_mask(vol["mask_path"])
            volume = normalize_volume(
                volume, preproc["clip_percentiles"][0], preproc["clip_percentiles"][1]
            )
            slices = extract_slices_abus_fixed(vol, volume, mask, cfg, split, rng)
        except Exception as e:
            print(f"    [ERROR] {vid}: {e}")
            continue

        img_dir = os.path.join(slices_root, "images", split)
        lbl_dir = os.path.join(slices_root, "labels", split)
        meta_dir = os.path.join(slices_root, "metadata", split)

        for fname, img, label_lines, meta in slices:
            cv2.imwrite(os.path.join(img_dir, f"{fname}.png"), img)
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                f.write("\n".join(label_lines))
            with open(os.path.join(meta_dir, f"{fname}.json"), "w") as f:
                json.dump(meta, f)

            total_slices += 1
            cls = meta.get("class", -1)
            class_counts[split][cls] += 1

        del volume, mask

    print(f"\nRegeneration complete: {total_slices} ABUS slices")
    for split in ["train", "val"]:
        print(f"\n  {split}:")
        for cid in sorted(class_counts[split]):
            name = class_map[cid][1] if cid in class_map else "negative"
            print(f"    {name}: {class_counts[split][cid]}")

    # Verify new slice dimensions
    sample = os.path.join(slices_root, "images", "train")
    abus_files = [f for f in os.listdir(sample) if f.startswith("abus_")]
    if abus_files:
        from PIL import Image
        test_img = Image.open(os.path.join(sample, abus_files[0]))
        print(f"\nVerification — new ABUS slice size (W x H): {test_img.size}")
        print(f"  Expected: 865 x 608")


if __name__ == "__main__":
    main()
