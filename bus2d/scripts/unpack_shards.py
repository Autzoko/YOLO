#!/usr/bin/env python3
"""
Unpack WebDataset tar shards into YOLO directory structure.

Usage:
    python unpack_shards.py --shards-dir /path/to/shards --output-dir /path/to/output

Reads train/ and val/ tar shards, extracts into:
    output_dir/
        images/train/  images/val/
        labels/train/  labels/val/
"""

import argparse
import os
import tarfile
import sys


def unpack_split(shards_dir, output_dir, split):
    """Unpack all tar shards for a split into YOLO images/labels structure."""
    shard_dir = os.path.join(shards_dir, split)
    if not os.path.isdir(shard_dir):
        print(f"  [SKIP] {shard_dir} not found")
        return 0

    img_dir = os.path.join(output_dir, "images", split)
    lbl_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    tar_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".tar")])
    n_samples = 0

    for tf_name in tar_files:
        tf_path = os.path.join(shard_dir, tf_name)
        with tarfile.open(tf_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".png"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, img_dir)
                    n_samples += 1
                elif member.name.endswith(".txt"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, lbl_dir)
        print(f"    Unpacked {tf_name}")

    return n_samples


def main():
    parser = argparse.ArgumentParser(description="Unpack shards to YOLO format")
    parser.add_argument("--shards-dir", required=True, help="Path to shards/ directory")
    parser.add_argument("--output-dir", required=True, help="Path to output YOLO directory")
    args = parser.parse_args()

    print("Unpacking shards to YOLO format")
    print(f"  Shards: {args.shards_dir}")
    print(f"  Output: {args.output_dir}")

    for split in ["train", "val"]:
        print(f"\n  [{split}]")
        n = unpack_split(args.shards_dir, args.output_dir, split)
        print(f"  → {n} images")

    print("\nDone.")


if __name__ == "__main__":
    main()
