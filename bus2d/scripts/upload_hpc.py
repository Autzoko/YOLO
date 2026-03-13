#!/usr/bin/env python3
"""
Step 4 — Upload processed data to NYU Jubail HPC via rsync.

Uploads slices_root (YOLO-format images/labels) and shards_root to the HPC.
Always does a dry-run first and requires confirmation.
"""

import argparse
import os
import subprocess
import sys

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


def rsync_transfer(local_path, remote_dest, flags, dry_run=False):
    """Run rsync.  If dry_run, append --dry-run flag."""
    cmd = ["rsync"] + flags.split()
    if dry_run:
        cmd.append("--dry-run")
    # Ensure trailing slash on local path for directory sync
    local = local_path.rstrip("/") + "/"
    cmd.extend([local, remote_dest])

    print(f"\n  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Upload to HPC via rsync")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    )
    parser.add_argument("--skip-shards", action="store_true", help="Only upload YOLO slices")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hpc = cfg["hpc"]
    host = hpc["host"]
    remote_root = hpc["remote_root"]
    flags = hpc["rsync_flags"]

    print("=" * 60)
    print("BUS-2D HPC Upload")
    print("=" * 60)

    transfers = [
        ("YOLO slices", cfg["slices_root"], f"{host}:{remote_root}/slices"),
    ]
    if not args.skip_shards:
        transfers.append(
            ("WebDataset shards", cfg["shards_root"], f"{host}:{remote_root}/shards"),
        )

    # Also upload config, dataset yaml, and slurm script
    bus2d_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transfers.append(
        ("Config + scripts", bus2d_dir, f"{host}:{remote_root}/bus2d"),
    )

    # Dry run first
    print("\n--- DRY RUN ---")
    for name, local, remote in transfers:
        if not os.path.isdir(local):
            print(f"\n  [SKIP] {name}: {local} does not exist")
            continue
        print(f"\n  Syncing {name}: {local} → {remote}")
        rsync_transfer(local, remote, flags, dry_run=True)

    resp = input("\nProceed with actual upload? [y/N] ").strip().lower()
    if resp != "y":
        print("Aborted.")
        return

    print("\n--- UPLOADING ---")
    for name, local, remote in transfers:
        if not os.path.isdir(local):
            continue
        print(f"\n  Syncing {name}...")
        rc = rsync_transfer(local, remote, flags, dry_run=False)
        if rc != 0:
            print(f"  [ERROR] rsync failed for {name} (exit code {rc})")
            sys.exit(rc)

    print("\nUpload complete.")


if __name__ == "__main__":
    main()
