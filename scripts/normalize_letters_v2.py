"""
normalize_letters_v2.py

Same purpose as normalize_letters.py (safely normalize crops to 128x128
without ever cutting off a letter), PLUS a fix for a real bug found via
visual review: leftover grid-line slivers at the edge of some original
crops were being misread as "ink," which (a) stretched the detected
letter bounding box out to that edge unnecessarily, and (b) polluted the
background-color estimate, producing a visible gray/silver border in the
normalized output instead of clean white.

WHAT'S NEW vs v1:
    - Before computing the letter's bounding box, connected components
      that are thin, long, AND touch the outer edge of the original crop
      are treated as border/grid-line artifacts and excluded from the
      bbox calculation.
    - Those identified line pixels are also explicitly erased (painted to
      the estimated background color) in the final output, in case any
      of them would otherwise survive inside the safety-margin crop.
    - A new border_lines_removed.csv logs every file where a line was
      detected and removed, with its position/size, so you can spot-check
      a sample and confirm real letter strokes were never misclassified.

Everything else (safety margin, pad-to-square, resize-not-crop, the
edge-touch verification pass) works the same as before.

Usage:
    python normalize_letters_v2.py <dataset_root> <output_root>

Requires: opencv-python, numpy
    pip install opencv-python numpy
"""

import sys
import csv
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SIZE = 128
SAFETY_MARGIN_PX = 10
PADDING_RATIO = 0.12
MIN_NOISE_AREA_PX = 3
EDGE_TOUCH_TOLERANCE_PX = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# --- border-line detection thresholds ---
BORDER_TOUCH_PX = 2          # component must be within this many px of the
                              # image edge to be considered "touching" it
LINE_LENGTH_FRACTION = 0.5   # component must span at least this fraction
                              # of the image width/height to count as a line
LINE_THICKNESS_MAX_PX = 8    # component's thickness (the short dimension)
                              # must be at or below this to count as a line
LINE_THICKNESS_MAX_FRACTION = 0.04  # ...or this fraction of image size,
                                     # whichever is larger


def detect_ink_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_val, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if np.count_nonzero(mask) > mask.size * 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def clean_mask_remove_noise(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= MIN_NOISE_AREA_PX:
            cleaned[labels == label_id] = 255
    return cleaned


def remove_border_line_artifacts(mask: np.ndarray):
    """
    Identifies connected components that are thin, long, and touch the
    outer edge of the frame -- the signature of a leftover grid/border
    line rather than a letter stroke. Returns (cleaned_mask, line_mask,
    removed_components_info).

    Deliberately conservative: a component only counts as a line if it
    BOTH touches the border AND has a strong line-like aspect ratio.
    Genuine letter strokes are essentially never long, uniformly thin,
    AND exactly flush with the crop's outer edge at the same time, so
    this should not remove real letter content.
    """
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = mask.copy()
    line_mask = np.zeros_like(mask)
    removed = []

    thickness_limit_h = max(LINE_THICKNESS_MAX_PX, int(LINE_THICKNESS_MAX_FRACTION * h))
    thickness_limit_w = max(LINE_THICKNESS_MAX_PX, int(LINE_THICKNESS_MAX_FRACTION * w))

    for label_id in range(1, num_labels):
        x, y, cw, ch, area = stats[label_id]

        touches_border = (
            x <= BORDER_TOUCH_PX
            or y <= BORDER_TOUCH_PX
            or (x + cw) >= (w - BORDER_TOUCH_PX)
            or (y + ch) >= (h - BORDER_TOUCH_PX)
        )
        if not touches_border:
            continue

        is_horizontal_line = cw >= LINE_LENGTH_FRACTION * w and ch <= thickness_limit_h
        is_vertical_line = ch >= LINE_LENGTH_FRACTION * h and cw <= thickness_limit_w

        if is_horizontal_line or is_vertical_line:
            component_pixels = labels == label_id
            cleaned[component_pixels] = 0
            line_mask[component_pixels] = 255
            removed.append(
                {
                    "x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
                    "area": int(area),
                    "orientation": "horizontal" if is_horizontal_line else "vertical",
                }
            )

    return cleaned, line_mask, removed


def get_ink_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def process_one_image(img_path: Path):
    info = {
        "file": img_path.name,
        "orig_w": None, "orig_h": None,
        "bbox_w": None, "bbox_h": None,
        "scale_factor": None,
        "flagged_edge_touch": False,
        "line_removed": False,
        "line_count": 0,
        "error": "",
    }

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        info["error"] = "failed to load image"
        return None, info, []

    h, w = img.shape
    info["orig_w"], info["orig_h"] = w, h

    mask = detect_ink_mask(img)
    mask = clean_mask_remove_noise(mask)
    mask_clean, line_mask, removed_lines = remove_border_line_artifacts(mask)

    info["line_removed"] = len(removed_lines) > 0
    info["line_count"] = len(removed_lines)

    bbox = get_ink_bbox(mask_clean)
    if bbox is None:
        info["error"] = "no ink detected after removing border-line artifacts"
        return None, info, removed_lines

    x_min, y_min, x_max, y_max = bbox

    x_min_safe = max(0, x_min - SAFETY_MARGIN_PX)
    y_min_safe = max(0, y_min - SAFETY_MARGIN_PX)
    x_max_safe = min(w - 1, x_max + SAFETY_MARGIN_PX)
    y_max_safe = min(h - 1, y_max + SAFETY_MARGIN_PX)

    cropped = img[y_min_safe : y_max_safe + 1, x_min_safe : x_max_safe + 1].copy()
    cropped_line_mask = line_mask[y_min_safe : y_max_safe + 1, x_min_safe : x_max_safe + 1]
    bbox_h, bbox_w = cropped.shape
    info["bbox_w"], info["bbox_h"] = bbox_w, bbox_h

    # --- estimate background color from border strips, EXCLUDING any
    #     pixels identified as leftover line artifacts, so the estimate
    #     isn't dragged toward gray/dark ---
    border_coords = np.zeros_like(cropped, dtype=bool)
    border_coords[0, :] = True
    border_coords[-1, :] = True
    border_coords[:, 0] = True
    border_coords[:, -1] = True
    valid_bg_coords = border_coords & (cropped_line_mask == 0)

    if np.any(valid_bg_coords):
        bg_value = int(np.median(cropped[valid_bg_coords]))
    else:
        # fallback: every border pixel was line-flagged (rare) -- use the
        # median of the whole crop instead
        bg_value = int(np.median(cropped))

    # --- erase any remaining line-artifact pixels inside the crop ---
    if np.any(cropped_line_mask):
        cropped[cropped_line_mask == 255] = bg_value

    # --- pad to square with breathing room ---
    side = max(bbox_w, bbox_h)
    side = int(side * (1 + PADDING_RATIO))
    side = max(side, bbox_w, bbox_h)

    square = np.full((side, side), bg_value, dtype=np.uint8)
    y_offset = (side - bbox_h) // 2
    x_offset = (side - bbox_w) // 2
    square[y_offset : y_offset + bbox_h, x_offset : x_offset + bbox_w] = cropped

    info["scale_factor"] = round(TARGET_SIZE / side, 4)

    interpolation = cv2.INTER_AREA if side > TARGET_SIZE else cv2.INTER_LANCZOS4
    final_img = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE), interpolation=interpolation)

    # --- verification: confirm no ink touches the final border ---
    final_mask = detect_ink_mask(final_img)
    final_mask = clean_mask_remove_noise(final_mask)
    if (
        np.any(final_mask[:EDGE_TOUCH_TOLERANCE_PX, :])
        or np.any(final_mask[-EDGE_TOUCH_TOLERANCE_PX:, :])
        or np.any(final_mask[:, :EDGE_TOUCH_TOLERANCE_PX])
        or np.any(final_mask[:, -EDGE_TOUCH_TOLERANCE_PX:])
    ):
        info["flagged_edge_touch"] = True

    return final_img, info, removed_lines


def main():
    if len(sys.argv) != 3:
        print("Usage: python normalize_letters_v2.py <dataset_root> <output_root>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not dataset_root.exists():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    letter_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not letter_dirs:
        print(f"ERROR: no subfolders found inside {dataset_root}")
        sys.exit(1)

    log_rows, flagged_rows, error_rows, line_rows = [], [], [], []
    total_processed = 0

    for letter_dir in letter_dirs:
        letter = letter_dir.name
        out_letter_dir = output_root / letter
        out_letter_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            [p for p in letter_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        print(f"Processing '{letter}': {len(images)} images...")

        for img_path in images:
            final_img, info, removed_lines = process_one_image(img_path)
            info["letter"] = letter

            if final_img is None:
                error_rows.append(info)
                print(f"  SKIPPED {img_path.name}: {info['error']}")
                continue

            out_path = out_letter_dir / img_path.name
            cv2.imwrite(str(out_path), final_img)
            total_processed += 1
            log_rows.append(info)

            if info["flagged_edge_touch"]:
                flagged_rows.append(info)

            for line_info in removed_lines:
                row = dict(line_info)
                row["letter"] = letter
                row["file"] = img_path.name
                line_rows.append(row)

    fieldnames = [
        "letter", "file", "orig_w", "orig_h", "bbox_w", "bbox_h",
        "scale_factor", "flagged_edge_touch", "line_removed", "line_count", "error",
    ]
    with open(output_root / "normalization_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    with open(output_root / "review_needed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    with open(output_root / "errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    line_fieldnames = ["letter", "file", "x", "y", "w", "h", "area", "orientation"]
    with open(output_root / "border_lines_removed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=line_fieldnames)
        writer.writeheader()
        writer.writerows(line_rows)

    print("\n----------------------------------------")
    print(f"Total images processed successfully: {total_processed}")
    print(f"Border-line artifacts removed from: {len(set((r['letter'], r['file']) for r in line_rows))} files")
    print(f"Still flagged for manual review (ink near edge): {len(flagged_rows)}")
    print(f"Skipped due to errors: {len(error_rows)}")
    print(f"\nNormalized dataset saved to: {output_root}")
    print(f"Full log:               normalization_log.csv")
    print(f"Needs manual review:    review_needed.csv")
    print(f"Errors:                 errors.csv")
    print(f"Border lines removed:   border_lines_removed.csv  <- spot check a sample of these")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
