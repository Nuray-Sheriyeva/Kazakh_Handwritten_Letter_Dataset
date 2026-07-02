"""
make_split_folders.py

Uses the split assignments in metadata.csv to physically organize your
128x128 images into separate train/val/test folders, each containing
per-letter subfolders. This does NOT re-decide the split -- it just
copies files into folders based on the split column that make_metadata.py
already assigned (so sheets stay grouped correctly, no leakage).

Resulting structure:
    output_root/
        train/
            а/  (only images from sheets assigned to train)
            б/
            ...
        val/
            а/
            ...
        test/
            а/
            ...

Your original dataset_root_128/ is left untouched -- this only reads from
it and copies (not moves) into the new structure.

Usage:
    python make_split_folders.py <dataset_root_128> <metadata.csv> <output_root>

Example:
    python make_split_folders.py dataset_root_128 dataset_root_128/metadata.csv dataset_split
"""

import sys
import csv
import shutil
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print("Usage: python make_split_folders.py <dataset_root_128> <metadata.csv> <output_root>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    metadata_path = Path(sys.argv[2])
    output_root = Path(sys.argv[3])

    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: metadata.csv not found at: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: metadata.csv is empty.")
        sys.exit(1)

    counts = {"train": 0, "val": 0, "test": 0}
    missing = []

    for row in rows:
        split = row["split"]
        filepath = row["filepath"]  # e.g. "а/Sheet_14_letter_r00_c01.png"

        src = dataset_root / filepath
        if not src.exists():
            missing.append(filepath)
            continue

        dst_dir = output_root / split / row["label"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        counts[split] += 1

    print("----------------------------------------")
    print("Done copying files into split folders.")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counts[split]} images -> {output_root / split}")
    if missing:
        print(f"\nWARNING: {len(missing)} files listed in metadata.csv were not found on disk:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
