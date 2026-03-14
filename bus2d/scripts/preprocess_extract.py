#!/usr/bin/env python3
"""
Step 2 — Preprocess volumes and extract 2D coronal slices with YOLO labels.

Reads manifest.json, processes each volume:
  1. Resample Duying Y-axis from 3mm → 1mm
  2. Normalize (clip [1st,99th] percentile → uint8)
  3. Extract positive coronal slices (middle 1/3 of lesion Y-extent)
  4. Extract negative coronal slices (outside lesion Y-range, 10-slice buffer)
  5. Apply augmentation to training slices
  6. Save PNG images + YOLO .txt labels

Output structure (YOLO format):
  slices_root/
    images/train/  images/val/
    labels/train/  labels/val/
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml
from scipy.ndimage import zoom

# ── config loading (shared with scan_manifest) ─────────────────────

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


# ── volume loading ──────────────────────────────────────────────────

def load_abus_volume(data_path):
    """Load ABUS NRRD volume.  Returns 3D numpy array (X, Y, Z) float32."""
    img = sitk.ReadImage(data_path)
    arr = sitk.GetArrayFromImage(img)  # (Z?, Y?, X?) — see note below
    # SimpleITK returns (k, j, i) for image axes (i, j, k).
    # For ABUS, spacing is (1,1,1) so axis order doesn't matter for resampling,
    # but matters for coronal slicing.  We'll treat the array as (dim0, dim1, dim2)
    # and use dim1 as the Y-axis (coronal slice = fix dim1).
    return arr.astype(np.float32)


def load_abus_mask(mask_path):
    """Load ABUS NRRD mask.  Returns 3D bool/uint8 array."""
    img = sitk.ReadImage(mask_path)
    return sitk.GetArrayFromImage(img).astype(np.uint8)


def load_duying_volume(nii_path):
    """Load Duying NIfTI volume.  Returns 3D numpy array and spacing."""
    img = nib.load(nii_path)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    pixdim = img.header["pixdim"]
    spacing = [float(pixdim[1]), float(pixdim[2]), float(pixdim[3])]
    return arr, spacing


# ── preprocessing ───────────────────────────────────────────────────

def resample_y(volume, src_spacing_y, tgt_spacing_y):
    """Resample volume along axis 1 (Y) from src to tgt spacing."""
    factor = src_spacing_y / tgt_spacing_y
    if abs(factor - 1.0) < 0.01:
        return volume
    # zoom factors: (1, factor, 1)
    return zoom(volume, (1, factor, 1), order=1).astype(volume.dtype)


def normalize_volume(volume, lo_pct=1, hi_pct=99):
    """Clip to [lo_pct, hi_pct] percentile, scale to uint8."""
    lo = np.percentile(volume, lo_pct)
    hi = np.percentile(volume, hi_pct)
    if hi - lo < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    clipped = np.clip(volume, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return scaled


# ── bbox helpers ────────────────────────────────────────────────────

def bbox_from_mask_slice(mask_2d):
    """
    Compute bounding box from a 2D binary mask.
    Returns (cx, cy, w, h) in pixel coords, or None if empty.
    """
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    cx = x_min + w / 2
    cy = y_min + h / 2
    return cx, cy, w, h


def yolo_label_line(class_id, cx, cy, w, h, img_w, img_h):
    """Format a YOLO label line: class cx cy w h (all normalized)."""
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"


# ── augmentation ────────────────────────────────────────────────────

def augment_slice(img, aug_cfg, rng):
    """Apply augmentation to a single 2D uint8 image.  Returns augmented copy.

    NOTE: Horizontal flip is NOT applied here because bounding box labels
    are computed BEFORE this function is called and would become misaligned.
    Flipping is handled during training with proper bbox adjustment.
    """
    out = img.copy()

    # Brightness/contrast jitter
    if rng.random() < aug_cfg["brightness_contrast_p"]:
        alpha = 1.0 + rng.uniform(-aug_cfg["contrast_limit"], aug_cfg["contrast_limit"])
        beta = rng.uniform(-aug_cfg["brightness_limit"], aug_cfg["brightness_limit"]) * 255
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if rng.random() < aug_cfg["gaussian_noise_p"]:
        lo, hi = aug_cfg["gaussian_noise_var_limit"]
        sigma = rng.uniform(lo, hi)
        noise = rng.normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


# ── slice extraction ────────────────────────────────────────────────

def extract_slices_abus(vol_info, volume, mask, cfg, split, rng):
    """
    Extract coronal slices from an ABUS volume.

    volume / mask: 3D arrays with Y = axis 1
    Returns list of (filename, image_2d, label_lines, metadata).
    """
    se_cfg = cfg["slice_extraction"]
    aug_cfg = cfg["augmentation"]
    results = []
    vol_id = vol_info["volume_id"]

    # Get Y extent from bbx_labels (already in manifest)
    occupied_y_ranges = []

    for les in vol_info["lesions"]:
        y_min = les["y_min"]
        y_max = les["y_max"]
        occupied_y_ranges.append((y_min, y_max))

        # Middle third
        y_span = y_max - y_min
        mid_start = y_min + int(y_span / 3)
        mid_end = y_min + int(2 * y_span / 3)

        # Clamp to axis 0 (coronal sweep direction, dim=330)
        mid_start = max(mid_start, 0)
        mid_end = min(mid_end, volume.shape[0] - 1)

        n_slices = mid_end - mid_start + 1
        n_slices = max(se_cfg["min_slices"], min(se_cfg["max_slices"], n_slices))
        if mid_end - mid_start + 1 > n_slices:
            # Subsample evenly
            indices = np.linspace(mid_start, mid_end, n_slices, dtype=int)
        else:
            indices = list(range(mid_start, mid_end + 1))

        for y_idx in indices:
            # Coronal slice: fix axis 0 (coronal sweep), get (axial, lateral) image
            slice_2d = volume[y_idx, :, :]
            mask_2d = mask[y_idx, :, :]

            # Compute bbox from mask on this slice
            bb = bbox_from_mask_slice(mask_2d)
            label_lines = []
            if bb is not None:
                cx, cy, w, h = bb
                img_h, img_w = slice_2d.shape
                label_lines.append(
                    yolo_label_line(les["class_id"], cx, cy, w, h, img_w, img_h)
                )

            # Normalize to uint8 (volume already normalized)
            img = slice_2d if slice_2d.dtype == np.uint8 else slice_2d.astype(np.uint8)

            # Augment training slices
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

    # Find valid negative coronal indices (axis 0)
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
            results.append((fname, img, [], meta))  # empty labels

    return results


def extract_slices_duying(vol_info, volume, cfg, split, rng):
    """
    Extract coronal slices from a Duying volume (already resampled to 1mm isotropic).

    volume: 3D array (X, Y_resampled, Z)
    Lesion y_min/y_max from manifest are in original 3mm voxel coords.
    After resampling by factor 3, y_voxel_1mm = y_voxel_3mm * 3.
    """
    se_cfg = cfg["slice_extraction"]
    aug_cfg = cfg["augmentation"]
    results = []
    vol_id = vol_info["volume_id"]
    orig_spacing_y = vol_info["spacing_mm"][1]
    resample_factor = orig_spacing_y / cfg["preprocessing"]["resample_y_mm"]

    occupied_y_ranges = []

    for les in vol_info["lesions"]:
        # Convert Y range from original voxel coords to resampled coords
        y_min_rs = int(les["y_min"] * resample_factor)
        y_max_rs = int(les["y_max"] * resample_factor)
        y_min_rs = max(0, y_min_rs)
        y_max_rs = min(volume.shape[1] - 1, y_max_rs)
        occupied_y_ranges.append((y_min_rs, y_max_rs))

        # Middle third
        y_span = y_max_rs - y_min_rs
        mid_start = y_min_rs + int(y_span / 3)
        mid_end = y_min_rs + int(2 * y_span / 3)
        mid_start = max(mid_start, 0)
        mid_end = min(mid_end, volume.shape[1] - 1)

        n_slices = mid_end - mid_start + 1
        n_slices = max(se_cfg["min_slices"], min(se_cfg["max_slices"], n_slices))
        if mid_end - mid_start + 1 > n_slices:
            indices = np.linspace(mid_start, mid_end, n_slices, dtype=int)
        else:
            indices = list(range(mid_start, mid_end + 1))

        # Get 2D bbox from coronal annotation (in pixel coords, X/Z)
        cbbox = les.get("coronal_bbox_px", {})
        cx = cbbox.get("cx", 0)
        cz = cbbox.get("cz", 0)
        lx = cbbox.get("lx", 0)
        lz = cbbox.get("lz", 0)

        for y_idx in indices:
            if y_idx < 0 or y_idx >= volume.shape[1]:
                continue
            # Coronal slice: fix axis 1, get (X, Z) image
            slice_2d = volume[:, y_idx, :]
            img_h, img_w = slice_2d.shape  # (X_dim, Z_dim)

            # YOLO label: the bbox is in (X, Z) pixel space
            # In the 2D image: rows = X axis, cols = Z axis
            # YOLO expects: cx_norm, cy_norm, w_norm, h_norm
            # where x is horizontal (cols=Z) and y is vertical (rows=X)
            # So: yolo_cx = cz / img_w, yolo_cy = cx / img_h
            label_lines = []
            if lx > 0 and lz > 0:
                yolo_cx = cz / img_w
                yolo_cy = cx / img_h
                yolo_w = lz / img_w
                yolo_h = lx / img_h
                # Clamp to [0, 1]
                yolo_cx = max(0, min(1, yolo_cx))
                yolo_cy = max(0, min(1, yolo_cy))
                yolo_w = max(0, min(1, yolo_w))
                yolo_h = max(0, min(1, yolo_h))
                label_lines.append(
                    f"{les['class_id']} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}"
                )

            img = slice_2d if slice_2d.dtype == np.uint8 else slice_2d.astype(np.uint8)
            if split == "train":
                img = augment_slice(img, aug_cfg, rng)

            fname = f"{vol_id}_y{y_idx:04d}"
            meta = {
                "volume_id": vol_id,
                "dataset": "duying",
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
        for y in range(max(0, y_min - buffer), min(volume.shape[1], y_max + buffer + 1)):
            all_y.add(y)
    valid_neg_y = [y for y in range(volume.shape[1]) if y not in all_y]

    if valid_neg_y and n_neg_target > 0:
        neg_indices = rng.choice(
            valid_neg_y,
            size=min(n_neg_target, len(valid_neg_y)),
            replace=len(valid_neg_y) < n_neg_target,
        )
        for y_idx in neg_indices:
            slice_2d = volume[:, y_idx, :]
            img = slice_2d if slice_2d.dtype == np.uint8 else slice_2d.astype(np.uint8)
            if split == "train":
                img = augment_slice(img, aug_cfg, rng)
            fname = f"{vol_id}_y{y_idx:04d}_neg"
            meta = {
                "volume_id": vol_id,
                "dataset": "duying",
                "slice_index": int(y_idx),
                "class": -1,
                "birads": "none",
            }
            results.append((fname, img, [], meta))

    return results


# ── train/val split ─────────────────────────────────────────────────

def volume_level_split(volumes, cfg):
    """
    Split volumes into train/val at the volume level.
    Stratified by (dataset, primary_class).
    Returns (train_vols, val_vols).
    """
    from collections import defaultdict

    split_cfg = cfg["split"]
    rng = np.random.RandomState(split_cfg["seed"])
    val_frac = split_cfg["val_fraction"]

    # Stratify key = (dataset, primary_class)
    strata = defaultdict(list)
    for v in volumes:
        primary_cls = v["lesions"][0]["class_id"] if v["lesions"] else -1
        key = (v["dataset"], primary_cls)
        strata[key].append(v)

    train_vols, val_vols = [], []
    for key, vols in strata.items():
        rng.shuffle(vols)
        n_val = max(1, int(len(vols) * val_frac))
        val_vols.extend(vols[:n_val])
        train_vols.extend(vols[n_val:])

    # Hard assertion: zero overlap
    train_ids = {v["volume_id"] for v in train_vols}
    val_ids = {v["volume_id"] for v in val_vols}
    assert train_ids.isdisjoint(val_ids), "FATAL: Train/val overlap detected!"

    return train_vols, val_vols


# ── main processing loop ───────────────────────────────────────────

def process_volumes(volumes, split, cfg, output_root, rng):
    """Process all volumes for a given split.  Returns (n_slices, class_counts)."""
    from collections import Counter

    img_dir = os.path.join(output_root, "images", split)
    lbl_dir = os.path.join(output_root, "labels", split)
    meta_dir = os.path.join(output_root, "metadata", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    preproc = cfg["preprocessing"]
    n_slices = 0
    class_counts = Counter()

    for i, vol in enumerate(volumes):
        vid = vol["volume_id"]
        ds = vol["dataset"]
        print(f"  [{split}] {i+1}/{len(volumes)}: {vid} ({ds})")

        try:
            if ds == "abus":
                volume = load_abus_volume(vol["data_path"])
                mask = load_abus_mask(vol["mask_path"])
                volume = normalize_volume(
                    volume, preproc["clip_percentiles"][0], preproc["clip_percentiles"][1]
                )
                slices = extract_slices_abus(vol, volume, mask, cfg, split, rng)
            else:  # duying
                raw_vol, spacing = load_duying_volume(vol["data_path"])
                # Resample Y axis
                raw_vol = resample_y(raw_vol, spacing[1], preproc["resample_y_mm"])
                # Normalize
                raw_vol = normalize_volume(
                    raw_vol, preproc["clip_percentiles"][0], preproc["clip_percentiles"][1]
                )
                slices = extract_slices_duying(vol, raw_vol, cfg, split, rng)

        except Exception as e:
            print(f"    [ERROR] {vid}: {e}")
            continue

        for fname, img, label_lines, meta in slices:
            # Save PNG
            cv2.imwrite(os.path.join(img_dir, f"{fname}.png"), img)
            # Save YOLO label (empty file for negatives)
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                f.write("\n".join(label_lines))
            # Save metadata JSON
            with open(os.path.join(meta_dir, f"{fname}.json"), "w") as f:
                json.dump(meta, f)

            n_slices += 1
            cls = meta.get("class", -1)
            class_counts[cls] += 1

        # free memory
        try:
            del raw_vol
        except NameError:
            pass

    return n_slices, class_counts


def main():
    parser = argparse.ArgumentParser(description="Preprocess and extract slices")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json (default: from config)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = args.manifest or cfg["manifest_path"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    volumes = manifest["volumes"]
    class_map = cfg["classes"]

    print("=" * 60)
    print("BUS-2D Preprocessing & Slice Extraction")
    print("=" * 60)
    print(f"Volumes in manifest: {len(volumes)}")

    # Split
    print("\nPerforming volume-level train/val split...")
    train_vols, val_vols = volume_level_split(volumes, cfg)

    print(f"  Train volumes: {len(train_vols)}")
    print(f"  Val volumes:   {len(val_vols)}")

    # Estimate slices
    se_cfg = cfg["slice_extraction"]
    est_pos = 0
    for v in volumes:
        for les in v["lesions"]:
            y_span = les["y_max"] - les["y_min"]
            mid_slices = max(
                se_cfg["min_slices"],
                min(se_cfg["max_slices"], int(y_span / 3)),
            )
            est_pos += mid_slices
    est_neg = int(est_pos * se_cfg["negative_ratio"])
    print(f"\nEstimated slices: ~{est_pos} positive + ~{est_neg} negative = ~{est_pos + est_neg}")

    output_root = cfg["slices_root"]
    print(f"Output directory: {output_root}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    resp = input("\nProceed with extraction? [y/N] ").strip().lower()
    if resp != "y":
        print("Aborted.")
        return

    rng = np.random.RandomState(cfg["split"]["seed"])

    # Process train
    print("\n--- Processing TRAIN split ---")
    n_train, train_cls = process_volumes(train_vols, "train", cfg, output_root, rng)

    # Process val (no augmentation applied — split=="val" skips it)
    print("\n--- Processing VAL split ---")
    n_val, val_cls = process_volumes(val_vols, "val", cfg, output_root, rng)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Train slices: {n_train}")
    for cid in sorted(train_cls):
        name = class_map[cid][1] if cid in class_map else "negative"
        print(f"  {name}: {train_cls[cid]}")
    print(f"\nVal slices: {n_val}")
    for cid in sorted(val_cls):
        name = class_map[cid][1] if cid in class_map else "negative"
        print(f"  {name}: {val_cls[cid]}")

    # Also save split info
    split_info = {
        "train_volume_ids": [v["volume_id"] for v in train_vols],
        "val_volume_ids": [v["volume_id"] for v in val_vols],
        "train_slices": n_train,
        "val_slices": n_val,
        "train_class_dist": {str(k): v for k, v in train_cls.items()},
        "val_class_dist": {str(k): v for k, v in val_cls.items()},
    }
    split_path = os.path.join(output_root, "split_info.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit info saved to: {split_path}")


if __name__ == "__main__":
    main()
