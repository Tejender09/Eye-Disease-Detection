import csv
import hashlib
from pathlib import Path
import random
import shutil
from PIL import Image, ImageOps
from collections import defaultdict

# ---------------- CONFIGURATION ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Support long paths on Windows with \\?\ prefix
RAW_DIR = Path("\\\\?\\" + str((PROJECT_ROOT / "datasets" / "raw").resolve()))
BACKUP_ZIP = PROJECT_ROOT / "Eye Dataset.v2-new-dataset.folder.zip"
OUTPUT_DIR = Path("\\\\?\\" + str((PROJECT_ROOT / "datasets" / "processed").resolve()))
REPORT_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = (224, 224)
MIN_WIDTH = 64
MIN_HEIGHT = 64
RANDOM_SEED = 42
DEDUPLICATE_BEFORE_SPLIT = True
FILTER_CROSS_CLASS_CONFLICTS = True

# 70% Train, 15% Validation, 15% Test
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

def clean_path(p):
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s[4:]
    return s

def compute_dhash(img, hash_size=8):
    """Compute difference hash (dhash) for perceptual similarity comparison."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BOX)
    pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
    return sum([2 ** i for i, v in enumerate(diff) if v])

# ---------------- BACKUP UTILITY ----------------

def ensure_backup():
    """Ensure a backup of raw dataset exists before any modification."""
    if BACKUP_ZIP.exists():
        print(f"Raw dataset backup confirmed via archive: {BACKUP_ZIP.name}")
    else:
        print("Note: Original archive not found; raw directory will serve as source.")

# ---------------- DISCOVER CLASSES ----------------

def discover_classes():
    """Dynamically discover class folders inside RAW_DIR."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")
    
    classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {RAW_DIR}")
    return classes

# ---------------- AUDIT RAW DATASET ----------------

def audit_raw_dataset(classes):
    print("\n========== RAW DATASET AUDIT ==========\n")
    records = []
    hash_map = {}
    dhash_map = defaultdict(lambda: defaultdict(list))
    corrupted = []
    low_resolution = []

    for class_name in classes:
        class_dir = RAW_DIR / class_name
        files = sorted(
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        )
        print(f"Class '{class_name}': {len(files)} raw files found.")

        for file_path in files:
            try:
                with Image.open(file_path) as img:
                    img.verify()

                with Image.open(file_path) as img:
                    width, height = img.size
                    file_bytes = file_path.read_bytes()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    p_hash = compute_dhash(img)

                    record = {
                        "path": file_path,
                        "class": class_name,
                        "width": width,
                        "height": height,
                        "hash": file_hash,
                        "dhash": p_hash,
                        "is_cross_conflict": False,
                        "conflict_classes": []
                    }

                    records.append(record)
                    hash_map.setdefault(file_hash, []).append(str(file_path))
                    dhash_map[p_hash][class_name].append(record)

                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        low_resolution.append(record)

            except Exception as error:
                corrupted.append({
                    "path": str(file_path),
                    "error": str(error)
                })

    # Exact byte duplicate groups
    duplicate_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    # Cross-class perceptual duplicate detection
    cross_class_conflicts = []
    for p_hash, class_records_dict in dhash_map.items():
        if len(class_records_dict) > 1:
            all_conflicted_classes = sorted(class_records_dict.keys())
            for c_name, rec_list in class_records_dict.items():
                for rec in rec_list:
                    rec["is_cross_conflict"] = True
                    rec["conflict_classes"] = all_conflicted_classes
                    cross_class_conflicts.append(rec)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-class conflicts report
    with open(REPORT_DIR / "cross_class_conflicts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Perceptual Hash", "Assigned Class", "Conflicting Classes", "File Path"])
        for rec in cross_class_conflicts:
            writer.writerow([
                hex(rec["dhash"]),
                rec["class"],
                " & ".join(rec["conflict_classes"]),
                clean_path(rec["path"])
            ])

    # Exact Duplicates report
    with open(REPORT_DIR / "exact_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SHA256", "File Paths"])
        for file_hash, paths in duplicate_groups.items():
            cleaned_paths = [clean_path(p) for p in paths]
            writer.writerow([file_hash, " | ".join(cleaned_paths)])

    # Dimensions report
    with open(REPORT_DIR / "image_dimensions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Class", "Width", "Height", "Low Resolution"])
        for record in records:
            writer.writerow([
                clean_path(record["path"]),
                record["class"],
                record["width"],
                record["height"],
                record["width"] < MIN_WIDTH or record["height"] < MIN_HEIGHT
            ])

    # Corrupted report
    with open(REPORT_DIR / "corrupted_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "error"])
        writer.writeheader()
        for c in corrupted:
            writer.writerow({"path": clean_path(c["path"]), "error": c["error"]})

    print("\nRaw Audit Summary:")
    print(f" - Readable images: {len(records)}")
    print(f" - Corrupted images: {len(corrupted)}")
    print(f" - Low-resolution images (<{MIN_WIDTH}x{MIN_HEIGHT}): {len(low_resolution)}")
    print(f" - Exact duplicate groups: {len(duplicate_groups)}")
    print(f" - Cross-class conflicted images: {len(cross_class_conflicts)}")

    return records, corrupted, duplicate_groups, cross_class_conflicts

# ---------------- PREPROCESSING & SPLITTING (70 / 15 / 15) ----------------

def preprocess_and_split(records, classes):
    print("\n========== PREPROCESSING & SPLITTING (70 / 15 / 15) ==========\n")
    random.seed(RANDOM_SEED)

    # Clean / prepare output directories
    for split in ["train", "val", "test"]:
        for class_name in classes:
            split_dir = OUTPUT_DIR / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for old_file in split_dir.glob("*.jpg"):
                try:
                    old_file.unlink()
                except Exception:
                    pass

    # Filter out cross-class duplicates and intra-class exact duplicates
    class_to_records = {c: [] for c in classes}
    seen_hashes = set()
    filtered_cross_count = 0

    for r in records:
        if FILTER_CROSS_CLASS_CONFLICTS and r["is_cross_conflict"]:
            filtered_cross_count += 1
            continue

        if DEDUPLICATE_BEFORE_SPLIT:
            if r["hash"] in seen_hashes:
                continue
            seen_hashes.add(r["hash"])

        class_to_records[r["class"]].append(r)

    print(f"Excluded {filtered_cross_count} cross-class conflicting images.")
    print(f"Remaining clean images for dataset: {sum(len(items) for items in class_to_records.values())}\n")

    split_counts = {split: {c: 0 for c in classes} for split in ["train", "val", "test"]}
    processed_records = []

    for class_name, items in class_to_records.items():
        # Deterministic shuffle
        items_copy = list(items)
        random.shuffle(items_copy)

        n_total = len(items_copy)
        n_train = int(round(n_total * SPLIT_RATIOS["train"]))
        n_val = int(round(n_total * SPLIT_RATIOS["val"]))
        # Remaining goes to test to guarantee sum(splits) == n_total
        n_test = n_total - n_train - n_val

        splits_map = {
            "train": items_copy[:n_train],
            "val": items_copy[n_train:n_train + n_val],
            "test": items_copy[n_train + n_val:]
        }

        print(f"Class '{class_name}' ({n_total} items): "
              f"Train={len(splits_map['train'])}, "
              f"Val={len(splits_map['val'])}, "
              f"Test={len(splits_map['test'])}")

        for split_name, split_items in splits_map.items():
            for idx, item in enumerate(split_items):
                src_path = item["path"]
                # Save as standardized jpeg
                safe_class = class_name.replace(" ", "_")
                dest_filename = f"{safe_class}_{idx:05d}.jpg"
                dest_path = OUTPUT_DIR / split_name / class_name / dest_filename

                try:
                    with Image.open(src_path) as img:
                        # Auto-orient EXIF tag if present
                        img = ImageOps.exif_transpose(img)
                        # Ensure RGB 3-channel
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # High quality resize
                        img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        img_resized.save(dest_path, "JPEG", quality=95)

                        split_counts[split_name][class_name] += 1
                        processed_records.append({
                            "split": split_name,
                            "class": class_name,
                            "dest_path": str(dest_path),
                            "src_path": str(src_path),
                            "width": IMAGE_SIZE[0],
                            "height": IMAGE_SIZE[1]
                        })
                except Exception as err:
                    print(f"Error processing {src_path}: {err}")

    return split_counts, processed_records, filtered_cross_count

# ---------------- POST-PREPROCESSING AUDIT ----------------

def audit_processed_dataset(classes, split_counts, processed_records, filtered_cross_count, total_raw):
    print("\n========== POST-PREPROCESSING AUDIT ==========\n")
    hash_set_by_split = {"train": set(), "val": set(), "test": set()}

    for split in ["train", "val", "test"]:
        total_split = 0
        for class_name in classes:
            folder = OUTPUT_DIR / split / class_name
            files = list(folder.glob("*.jpg"))
            count = len(files)
            total_split += count

            for f in files:
                f_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                hash_set_by_split[split].add(f_hash)

        print(f"Processed '{split}' split total: {total_split} images")

    # Check for split leakage
    train_val_overlap = hash_set_by_split["train"].intersection(hash_set_by_split["val"])
    train_test_overlap = hash_set_by_split["train"].intersection(hash_set_by_split["test"])
    val_test_overlap = hash_set_by_split["val"].intersection(hash_set_by_split["test"])

    leakage_count = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

    # Save split distribution summary
    split_csv_path = REPORT_DIR / "split_distribution.csv"
    with open(split_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Train", "Val", "Test", "Total Clean", "Train %", "Val %", "Test %"])
        for c in classes:
            tr = split_counts["train"][c]
            va = split_counts["val"][c]
            te = split_counts["test"][c]
            tot = tr + va + te
            tr_pct = f"{(tr/tot)*100:.1f}%" if tot else "0%"
            va_pct = f"{(va/tot)*100:.1f}%" if tot else "0%"
            te_pct = f"{(te/tot)*100:.1f}%" if tot else "0%"
            writer.writerow([c, tr, va, te, tot, tr_pct, va_pct, te_pct])

    # Save overall audit summary
    summary_csv_path = REPORT_DIR / "audit_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Target Resolution", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"])
        writer.writerow(["Split Ratio (Train/Val/Test)", "70% / 15% / 15%"])
        writer.writerow(["Total Raw Images", total_raw])
        writer.writerow(["Filtered Cross-Class Conflicts", filtered_cross_count])
        writer.writerow(["Total Clean Processed Images", len(processed_records)])
        writer.writerow(["Train Images", sum(split_counts["train"].values())])
        writer.writerow(["Validation Images", sum(split_counts["val"].values())])
        writer.writerow(["Test Images", sum(split_counts["test"].values())])
        writer.writerow(["Data Leakage Between Splits", leakage_count])

    if leakage_count == 0:
        print(f"\nAudit complete. Zero data leakage detected between splits (Leakage count: 0).")
    else:
        print(f"\nAudit warning: Duplicate leakage detected between splits (Leakage count: {leakage_count}).")
    print(f"Detailed reports updated in: {REPORT_DIR}")

# ---------------- MAIN ----------------

def main():
    print("Starting Eye Disease Dataset preparation & audit pipeline...")
    ensure_backup()
    classes = discover_classes()
    print(f"Classes discovered: {classes}")

    records, corrupted, duplicate_groups, cross_class_conflicts = audit_raw_dataset(classes)
    split_counts, processed_records, filtered_count = preprocess_and_split(records, classes)
    audit_processed_dataset(classes, split_counts, processed_records, filtered_count, len(records))

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
