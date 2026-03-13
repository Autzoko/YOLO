#!/usr/bin/env python3
"""
Step 3 — Pack extracted slices into WebDataset tar shards.

Each shard: ~500 MB.  Each sample contains:
  - image.png   (the coronal slice)
  - label.txt   (YOLO label, may be empty for negatives)
  - meta.json   (volume_id, dataset, slice_index, class)
"""

import argparse
import json
import math
import os
import random
import tarfile
from collections import Counter

import yaml


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


def gather_samples(split_dir):
    """Gather all (image, label, meta) triplets from the extracted slice directory. Returns list of dicts with paths.
    """
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")
    meta_dir = os.path.join(split_dir, "metadata")

    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        meta_path = os.path.join(meta_dir, f"{stem}.json")

        if not os.path.exists(lbl_path):
            continue

        samples.append(
            {
                "stem": stem,
                "img_path": img_path,
                "lbl_path": lbl_path,
                "meta_path": meta_path if os.path.exists(meta_path) else None,
            }
        )

    return samples


def estimate_sample_size(samples, n=50):
    """Estimate average sample size from a random subset."""
    import random as _r

    subset = _r.sample(samples, min(n, len(samples)))
    total = 0
    for s in subset:
        total += os.path.getsize(s["img_path"])
        total += os.path.getsize(s["lbl_path"])
        if s["meta_path"]:
            total += os.path.getsize(s["meta_path"])
    return total / len(subset)


def write_shards(samples, output_dir, target_mb, split_name):
    """Write samples into tar shards of approximately target_mb each."""
    os.makedirs(output_dir, exist_ok=True)

    avg_size = estimate_sample_size(samples)
    samples_per_shard = max(1, int(target_mb * 1024 * 1024 / avg_size))
    n_shards = math.ceil(len(samples) / samples_per_shard)

    print(f"  Avg sample size: {avg_size / 1024:.1f} KB")
    print(f"  Samples per shard: ~{samples_per_shard}")
    print(f"  Total shards: {n_shards}")

    shard_idx = 0
    sample_count = 0
    tar = None
    shard_path = None

    for i, sample in enumerate(samples):
        if sample_count % samples_per_shard == 0:
            if tar is not None:
                tar.close()
            shard_path = os.path.join(output_dir, f"{split_name}-{shard_idx:05d}.tar")
            tar = tarfile.open(shard_path, "w")
            shard_idx += 1
            sample_count = 0

        stem = sample["stem"]

        # Add image
        _add_file_to_tar(tar, sample["img_path"], f"{stem}.png")
        # Add label
        _add_file_to_tar(tar, sample["lbl_path"], f"{stem}.txt")
        # Add metadata
        if sample["meta_path"]:
            _add_file_to_tar(tar, sample["meta_path"], f"{stem}.json")

        sample_count += 1

    if tar is not None:
        tar.close()

    return shard_idx


def _add_file_to_tar(tar, src_path, arcname):
    """Add a file to an open tar archive."""
    info = tar.gettarinfo(src_path, arcname=arcname)
    with open(src_path, "rb") as f:
        tar.addfile(info, f)


def main():
    parser = argparse.ArgumentParser(description="Pack slices into WebDataset shards")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    slices_root = cfg["slices_root"]
    shards_root = cfg["shards_root"]
    target_mb = cfg["sharding"]["target_shard_mb"]
    seed = cfg["sharding"]["shuffle_seed"]

    print("=" * 60)
    print("BUS-2D WebDataset Shard Packer")
    print("=" * 60)

    class_map = cfg["classes"]
    manifest_data = {"splits": {}}

    for split in ["train", "val"]:
        os.path.join(slices_root, split) if os.path.isdir(os.path.join(slices_root, split)) else slices_root
        # Images are in slices_root/images/{split}/
        img_dir = os.path.join(slices_root, "images", split)
        if not os.path.isdir(img_dir):
            print(f"\n  [WARN] No {split} images found at {img_dir}, skipping")
            continue

        print(f"\n--- {split.upper()} ---")
        # Gather samples from the flat structure
        samples = []
        lbl_dir = os.path.join(slices_root, "labels", split)
        meta_dir = os.path.join(slices_root, "metadata", split)

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            stem = fname[:-4]
            samples.append(
                {
                    "stem": stem,
                    "img_path": os.path.join(img_dir, fname),
                    "lbl_path": os.path.join(lbl_dir, f"{stem}.txt"),
                    "meta_path": os.path.join(meta_dir, f"{stem}.json")
                    if os.path.exists(os.path.join(meta_dir, f"{stem}.json"))
                    else None,
                }
            )

        print(f"  Samples: {len(samples)}")

        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(samples)

        # Class distribution
        cls_counts = Counter()
        for s in samples:
            if s["meta_path"] and os.path.exists(s["meta_path"]):
                with open(s["meta_path"]) as f:
                    meta = json.load(f)
                cls_counts[meta.get("class", -1)] += 1

        print("  Class distribution:")
        for cid in sorted(cls_counts):
            name = class_map[cid][1] if cid in class_map else "negative"
            print(f"    {name}: {cls_counts[cid]}")

        out_dir = os.path.join(shards_root, split)

        if args.dry_run:
            avg_sz = estimate_sample_size(samples) if samples else 0
            n_shards = math.ceil(len(samples) / max(1, int(target_mb * 1024 * 1024 / max(1, avg_sz)))) if samples else 0
            print(f"  [DRY RUN] Would write ~{n_shards} shards to {out_dir}")
            manifest_data["splits"][split] = {
                "samples": len(samples),
                "class_distribution": {
                    (class_map[k][1] if k in class_map else "negative"): v for k, v in cls_counts.items()
                },
            }
            continue

        n_shards = write_shards(samples, out_dir, target_mb, split)
        print(f"  Written {n_shards} shards to {out_dir}")

        manifest_data["splits"][split] = {
            "samples": len(samples),
            "shards": n_shards,
            "shard_dir": out_dir,
            "class_distribution": {
                (class_map[k][1] if k in class_map else "negative"): v for k, v in cls_counts.items()
            },
        }

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    resp = input("\nShards created. Write manifest.json? [y/N] ").strip().lower()
    if resp != "y":
        return

    manifest_path = os.path.join(shards_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Shard manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
