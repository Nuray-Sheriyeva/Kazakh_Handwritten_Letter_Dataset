"""
make_metadata.py

Builds metadata.csv for the normalized 128x128 dataset:
    filepath, label, sheet_id, split

Split is assigned by SHEET, not by individual image -- every image from a
given sheet (i.e. from the same writer) ends up entirely in train, entirely
in val, or entirely in test. This prevents the same handwriting style from
appearing in both training and test data, which would inflate accuracy by
letting the model partially recognize a specific writer's style rather than
the letter itself.

sheet_id is parsed directly from filenames like:
    Sheet_14_letter_r00_c01.png  ->  sheet_id = 14

Usage:
    python make_metadata.py <dataset_root_128> <output_csv_path> [--seed N] [--train 0.8] [--val 0.1]

Example:
    python make_metadata.py dataset_root_128 dataset_root_128/metadata.csv
    python make_metadata.py dataset_root_128 metadata.csv --seed 42 --train 0.8 --val 0.1

(test ratio is whatever remains after train+val)

Requires: no external dependencies (standard library only)
"""

import sys
import csv
import re
import random
import argparse
from pathlib import Path
from collections import defaultdict

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SHEET_ID_PATTERN = re.compile(r"(?i)sheet[_\-]?(\d+)")


def extract_sheet_id(filename: str):
    match = SHEET_ID_PATTERN.search(filename)
    if match:
        return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    if args.train + args.val >= 1.0:
        print("ERROR: --train + --val must be less than 1.0 (remainder becomes test)")
        sys.exit(1)

    letter_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not letter_dirs:
        print(f"ERROR: no subfolders found inside {dataset_root}")
        sys.exit(1)

    # --- Step 1: collect every image, and every distinct sheet_id seen ---
    records = []  # list of dicts: filepath, label, sheet_id
    unparsed = []

    for letter_dir in letter_dirs:
        label = letter_dir.name
        images = sorted(
            [p for p in letter_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        for img_path in images:
            sheet_id = extract_sheet_id(img_path.name)
            if sheet_id is None:
                unparsed.append(str(img_path))
                continue
            records.append(
                {
                    "filepath": f"{label}/{img_path.name}",
                    "label": label,
                    "sheet_id": sheet_id,
                }
            )

    if unparsed:
        print(f"WARNING: could not parse sheet_id from {len(unparsed)} filenames:")
        for u in unparsed[:10]:
            print(f"  {u}")
        if len(unparsed) > 10:
            print(f"  ... and {len(unparsed) - 10} more")
        print(
            "These files will be EXCLUDED from metadata.csv. Check their "
            "naming pattern if this number is large.\n"
        )

    all_sheet_ids = sorted(set(r["sheet_id"] for r in records), key=lambda x: int(x))
    print(f"Found {len(records)} images across {len(all_sheet_ids)} sheets.")

    # --- Step 2: shuffle sheets (not images) and split ---
    rng = random.Random(args.seed)
    shuffled_sheets = all_sheet_ids.copy()
    rng.shuffle(shuffled_sheets)

    n_sheets = len(shuffled_sheets)
    n_train = round(n_sheets * args.train)
    n_val = round(n_sheets * args.val)
    # remainder goes to test
    n_test = n_sheets - n_train - n_val

    train_sheets = set(shuffled_sheets[:n_train])
    val_sheets = set(shuffled_sheets[n_train : n_train + n_val])
    test_sheets = set(shuffled_sheets[n_train + n_val :])

    print(
        f"Sheet split -> train: {len(train_sheets)}, "
        f"val: {len(val_sheets)}, test: {len(test_sheets)}"
    )

    def sheet_to_split(sheet_id):
        if sheet_id in train_sheets:
            return "train"
        elif sheet_id in val_sheets:
            return "val"
        else:
            return "test"

    for r in records:
        r["split"] = sheet_to_split(r["sheet_id"])

    # --- Step 3: write metadata.csv ---
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "sheet_id", "split"])
        writer.writeheader()
        writer.writerows(records)

    # --- Step 4: summary report ---
    split_counts = defaultdict(int)
    label_split_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        split_counts[r["split"]] += 1
        label_split_counts[r["label"]][r["split"]] += 1

    print("\n----------------------------------------")
    print(f"metadata.csv written to: {args.output_csv}")
    print(f"Total images: {len(records)}")
    for split in ["train", "val", "test"]:
        pct = 100 * split_counts[split] / len(records) if records else 0
        print(f"  {split}: {split_counts[split]} images ({pct:.1f}%)")

    print("\nPer-letter split counts (first 5 letters shown as a sanity check):")
    for label in sorted(label_split_counts.keys())[:5]:
        counts = label_split_counts[label]
        print(f"  {label}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
