
from pathlib import Path
from PIL import Image, ImageOps
from collections import Counter
import hashlib
import csv

# ---------------- CONFIGURATION ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

CLASSES = [
    "Cataract",
    "Conjunctivitis",
    "Eyelid",
    "Normal",
    "Uveitis"
]

IMAGE_SIZE = (224, 224)
MIN_WIDTH = 64
MIN_HEIGHT = 64

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ---------------- SETUP ----------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True, exist_ok=True
        )

# ---------------- AUDIT ----------------

records = []
hash_map = {}
corrupted = []
low_resolution = []

print("\n========== DATASET AUDIT ==========\n")

for class_name in CLASSES:
    class_dir = RAW_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Missing class folder: {class_dir}"
        )

    files = sorted(
        f for f in class_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
    )

    print(f"{class_name}: {len(files)} files")

    for file_path in files:
        try:
            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                width, height = img.size

                # Exact file-content hash
                file_hash = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()

                record = {
                    "path": file_path,
                    "class": class_name,
                    "width": width,
                    "height": height,
                    "hash": file_hash
                }

                records.append(record)
                hash_map.setdefault(file_hash, []).append(
                    str(file_path)
                )

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    low_resolution.append(record)

        except Exception as error:
            corrupted.append({
                "path": str(file_path),
                "error": str(error)
            })

# ---------------- DUPLICATE REPORT ----------------

duplicate_groups = {
    h: paths for h, paths in hash_map.items()
    if len(paths) > 1
}

with open(
    REPORT_DIR / "exact_duplicates.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["SHA256", "File Paths"])

    for file_hash, paths in duplicate_groups.items():
        writer.writerow([file_hash, " | ".join(paths)])

# ---------------- DIMENSION REPORT ----------------

with open(
    REPORT_DIR / "image_dimensions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "File", "Class", "Width", "Height",
        "Low Resolution"
    ])

    for record in records:
        writer.writerow([
            str(record["path"]),
            record["class"],
            record["width"],
            record["height"],
            record["width"] < MIN_WIDTH
            or record["height"] < MIN_HEIGHT
        ])

# ---------------- CORRUPTED REPORT ----------------

with open(
    REPORT_DIR / "corrupted_images.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["path", "error"]
    )
    writer.writeheader()
    writer.writerows(corrupted)

# ---------------- SUMMARY ----------------

print("\n========== AUDIT SUMMARY ==========")
print("Readable images:", len(records))
print("Corrupted images:", len(corrupted))
print("Low-resolution images:", len(low_resolution))
print("Exact duplicate groups:", len(duplicate_groups))

print("\nClass distribution:")
for class_name in CLASSES:
    count = sum(
        r["class"] == class_name for r in records
    )
    print(f"{class_name}: {count}")

print("\nReports saved in:", REPORT_DIR)
print("\nAudit complete.")
print("No raw images were modified or deleted.")
