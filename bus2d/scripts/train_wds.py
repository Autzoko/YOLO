#!/usr/bin/env python3
"""
Train BUS-2D YOLO model reading directly from WebDataset tar shards.

Subclasses DetectionTrainer to override get_dataloader() with a
WebDataset-backed streaming dataloader. No file extraction needed.
"""

import argparse
import io
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import webdataset as wds
import yaml
from torch.utils.data import DataLoader

# Add ultralytics repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader


# ── Config loading ──────────────────────────────────────────────────

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


# ── WebDataset helpers ──────────────────────────────────────────────

def decode_sample(sample, imgsz=640):
    """
    Decode a single WebDataset sample into YOLO-compatible format.

    Each sample has keys like "png", "txt", "json".
    Returns dict matching YOLODataset.__getitem__ output after Format transform.
    """
    # Decode image
    img_bytes = sample["png"]
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR, (H, W, 3)
    if img is None:
        # Return empty sample
        img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    ori_h, ori_w = img.shape[:2]

    # Resize to imgsz (letterbox-style: scale to fit, pad)
    scale = min(imgsz / ori_h, imgsz / ori_w)
    new_h, new_w = int(ori_h * scale), int(ori_w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to imgsz x imgsz
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top = pad_h // 2
    left = pad_w // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    # Convert to CHW uint8 tensor
    img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).contiguous()  # (3, H, W)

    # Parse YOLO labels
    label_text = sample.get("txt", b"").decode("utf-8").strip()
    bboxes = []
    classes = []
    if label_text:
        for line in label_text.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # Adjust normalized bbox for letterbox padding
                # Original normalized coords → absolute in original image
                abs_cx = cx * ori_w
                abs_cy = cy * ori_h
                abs_w = w * ori_w
                abs_h = h * ori_h

                # Scale to resized image
                abs_cx *= scale
                abs_cy *= scale
                abs_w *= scale
                abs_h *= scale

                # Shift for padding
                abs_cx += left
                abs_cy += top

                # Back to normalized coords in padded image
                norm_cx = abs_cx / imgsz
                norm_cy = abs_cy / imgsz
                norm_w = abs_w / imgsz
                norm_h = abs_h / imgsz

                bboxes.append([norm_cx, norm_cy, norm_w, norm_h])
                classes.append(cls)

    n = len(classes)
    cls_tensor = torch.tensor(classes, dtype=torch.float32) if n > 0 else torch.zeros(0, dtype=torch.float32)
    bbox_tensor = torch.tensor(bboxes, dtype=torch.float32) if n > 0 else torch.zeros((0, 4), dtype=torch.float32)
    batch_idx = torch.zeros(n, dtype=torch.long)

    # im_file for logging
    im_file = sample.get("__key__", "unknown")

    # ratio_pad: (scale, (left_pad, top_pad)) — used by validator to map preds back
    ratio_pad = (scale, (left, top))

    return {
        "img": img_tensor,
        "cls": cls_tensor,
        "bboxes": bbox_tensor,
        "batch_idx": batch_idx,
        "im_file": im_file,
        "ori_shape": (ori_h, ori_w),
        "resized_shape": (imgsz, imgsz),
        "ratio_pad": ratio_pad,
    }


def yolo_collate_fn(batch):
    """
    Collate decoded samples into the batch format YOLO expects.

    batch: list of dicts from decode_sample()
    Returns: single dict with stacked/concatenated tensors.
    """
    new_batch = {}
    new_batch["img"] = torch.stack([b["img"] for b in batch], dim=0)
    new_batch["im_file"] = [b["im_file"] for b in batch]
    new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
    new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
    new_batch["ratio_pad"] = [b["ratio_pad"] for b in batch]

    # Concatenate annotations with batch_idx offset
    all_cls = []
    all_bboxes = []
    all_batch_idx = []
    for i, b in enumerate(batch):
        all_cls.append(b["cls"])
        all_bboxes.append(b["bboxes"])
        all_batch_idx.append(b["batch_idx"] + i)

    new_batch["cls"] = torch.cat(all_cls, dim=0)
    new_batch["bboxes"] = torch.cat(all_bboxes, dim=0)
    new_batch["batch_idx"] = torch.cat(all_batch_idx, dim=0)

    return new_batch


def make_wds_dataloader(shard_dir, batch_size, imgsz, workers, shuffle=True, rank=0, world_size=1):
    """
    Create a WebDataset-backed dataloader for YOLO training.

    shard_dir: path containing .tar files
    Returns: DataLoader yielding YOLO-compatible batch dicts.
    """
    import glob
    shard_pattern = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    if not shard_pattern:
        raise FileNotFoundError(f"No .tar files found in {shard_dir}")

    # Count total samples from the tar files (approximate)
    # WebDataset needs this for epoch length
    total_samples = 0
    import tarfile
    for tp in shard_pattern:
        with tarfile.open(tp, "r") as t:
            pngs = [m for m in t.getnames() if m.endswith(".png")]
            total_samples += len(pngs)

    print(f"  WebDataset: {len(shard_pattern)} shards, ~{total_samples} samples")

    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=shuffle, nodesplitter=wds.split_by_node)
        .shuffle(1000 if shuffle else 0)
        .to_tuple("__key__", "png", "txt")
        .map(lambda x: decode_sample({"__key__": x[0], "png": x[1], "txt": x[2]}, imgsz=imgsz))
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=yolo_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Attach length for YOLO progress bar
    loader.dataset_length = total_samples
    return loader, total_samples


# ── Custom Trainer ──────────────────────────────────────────────────

class WDSDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that reads from WebDataset tar shards."""

    def __init__(self, *args, shards_dir=None, **kwargs):
        self.shards_dir = shards_dir
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        """
        Override to skip file-system validation.
        Return a minimal data dict that satisfies YOLO's expectations.
        """
        # Read the dataset YAML for class names only
        with open(self.args.data) as f:
            data = yaml.safe_load(f)
        # Ensure required keys exist
        data.setdefault("nc", len(data.get("names", {})))
        data.setdefault("channels", 3)
        data.setdefault("train", "train")
        data.setdefault("val", "val")
        return data

    def build_dataset(self, img_path, mode="train", batch=None):
        """Not used — we override get_dataloader directly."""
        return None

    def plot_training_labels(self):
        """Skip — label stats not available from streaming shards."""
        pass

    def plot_training_samples(self, batch, ni):
        """Skip — avoid issues with streaming batch format."""
        pass

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Override to return WebDataset dataloader."""
        shuffle = mode == "train"
        split = "train" if mode == "train" else "val"
        shard_dir = os.path.join(self.shards_dir, split)

        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)
        imgsz = self.args.imgsz

        workers = self.args.workers
        if mode != "train":
            workers = min(workers, 2)

        loader, n_samples = make_wds_dataloader(
            shard_dir=shard_dir,
            batch_size=batch_size,
            imgsz=imgsz,
            workers=workers,
            shuffle=shuffle,
            rank=rank,
            world_size=getattr(self, "world_size", 1),
        )

        # YOLO needs __len__ on the dataloader for progress bars
        class LenWrapper:
            def __init__(self, loader, length, batch_size):
                self._loader = loader
                self._length = max(1, length // batch_size)
                # Copy attributes YOLO might access
                self.dataset = type("DS", (), {
                    "__len__": lambda self_: length,
                    "labels": [],
                    "label_files": [],
                })()

            def __iter__(self):
                return iter(self._loader)

            def __len__(self):
                return self._length

            def __getattr__(self, name):
                return getattr(self._loader, name)

        return LenWrapper(loader, n_samples, batch_size)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BUS-2D from WebDataset shards")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument(
        "--shards-dir",
        default="/scratch/ll5582/Data/Ultrasound/YOLO/shards",
        help="Path to shards/ directory with train/ and val/ subdirs",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=os.path.join(os.path.dirname(__file__), "..", "bus2d_dataset.yaml"),
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device, e.g. '0' or '0,1,2,3'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]

    model_path = args.resume or tcfg["model"]
    epochs = args.epochs or tcfg["epochs"]
    batch = args.batch or tcfg["batch"]
    device = args.device or tcfg["device"]
    if isinstance(device, str) and "," in device:
        device = [int(d) for d in device.split(",")]

    print("=" * 60)
    print("BUS-2D Training (WebDataset shards)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Shards: {args.shards_dir}")
    print(f"Epochs: {epochs}, Batch: {batch}, Device: {device}")

    overrides = dict(
        model=model_path,
        data=args.dataset_yaml,
        imgsz=tcfg["imgsz"],
        batch=batch,
        epochs=epochs,
        patience=tcfg["patience"],
        workers=tcfg["workers"],
        device=device,
        project=tcfg["project"],
        name=tcfg["name"],
        exist_ok=True,
        save=True,
        val=True,
        resume=args.resume or False,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,         # Disable mosaic — not compatible with streaming
        close_mosaic=0,
        copy_paste=0.0,     # Disable — not compatible with streaming
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mode="train",
        task="detect",
    )

    trainer = WDSDetectionTrainer(
        overrides=overrides,
        shards_dir=args.shards_dir,
    )
    trainer.train()

    print("\nTraining complete!")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
