
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "datasets" / "raw" / "external_eye"
PROCESSED_DIR = ROOT / "datasets" / "processed" / "external_eye"
SPLIT_DIR = ROOT / "datasets" / "splits"
REPORT_DIR = ROOT / "reports"

IMAGE_SIZE = (224, 224)
MIN_IMAGE_SIZE = 64

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ==========================================
# 2. IMAGE HASH
# ==========================================

def calculate_hash(image_path):

    hasher = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# ==========================================
# 3. COLLECT AND CLEAN DATASET
# ==========================================

def collect_images():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DIR}"
        )

    records = []

    corrupt_images = []
    small_images = []
    duplicate_images = []

    seen_hashes = {}

    print("\nScanning dataset...")

    for class_dir in sorted(RAW_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        for image_path in sorted(class_dir.rglob("*")):

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Check image readability
            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    width, height = img.size

            except (OSError, ValueError, UnidentifiedImageError):
                corrupt_images.append(str(image_path))
                continue

            # Exclude images smaller than 64x64
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                small_images.append(str(image_path))
                continue

            # Exact duplicate detection
            image_hash = calculate_hash(image_path)

            if image_hash in seen_hashes:

                previous_label = seen_hashes[image_hash]["label"]

                if previous_label != class_name:
                    duplicate_images.append(str(image_path))
                    duplicate_images.append(
                        seen_hashes[image_hash]["path"]
                    )
                else:
                    duplicate_images.append(str(image_path))

                continue

            seen_hashes[image_hash] = {
                "label": class_name,
                "path": str(image_path)
            }

            records.append({
                "path": image_path,
                "label": class_name,
                "hash": image_hash
            })

    cleaning_report = {
        "total_images_found": (
            len(records)
            + len(corrupt_images)
            + len(small_images)
            + len(duplicate_images)
        ),
        "corrupt_images": len(corrupt_images),
        "small_images_excluded": len(small_images),
        "duplicate_images_removed": len(duplicate_images),
        "final_usable_images": len(records)
    }

    print("\nCleaning completed.")

    return records, cleaning_report


# ==========================================
# 4. DATASET SPLITTING
# ==========================================

def split_dataset(records):

    df = pd.DataFrame(records)

    class_counts = df["label"].value_counts()

    if len(class_counts) < 2:
        raise ValueError("At least two classes are required.")

    if (class_counts < 3).any():
        raise ValueError(
            "Each class needs at least 3 images."
        )

    # 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # 15% validation, 15% testing
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_images(df):

    print("\nPreprocessing images...")

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_records = []

    for index, row in df.iterrows():

        try:

            with Image.open(row["path"]) as img:

                img = img.convert("RGB")

                img = img.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                output_dir = (
                    PROCESSED_DIR
                    / row["split"]
                    / row["label"]
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                image_name = (
                    f"{index:06d}_{row['hash'][:10]}.jpg"
                )

                output_path = output_dir / image_name

                img.save(
                    output_path,
                    format="JPEG",
                    quality=95
                )

                output_records.append({
                    "image_id": image_name,
                    "label": row["label"],
                    "split": row["split"],
                    "original_path": str(
                        row["path"].relative_to(ROOT)
                    ),
                    "processed_path": str(
                        output_path.relative_to(ROOT)
                    ),
                    "sha256": row["hash"]
                })

        except (OSError, ValueError) as error:

            raise RuntimeError(
                f"Preprocessing failed for {row['path']}: {error}"
            )

    return pd.DataFrame(output_records)


# ==========================================
# 6. GENERATE REPORTS
# ==========================================

def generate_reports(df, cleaning_report):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Individual split CSV files
    for split in ["train", "val", "test"]:

        split_df = df[df["split"] == split]

        split_df.to_csv(
            SPLIT_DIR / f"{split}.csv",
            index=False
        )

    # Class-wise counts
    counts = pd.crosstab(
        df["label"],
        df["split"]
    ).reindex(
        columns=["train", "val", "test"],
        fill_value=0
    )

    counts["total"] = counts.sum(axis=1)

    counts.to_csv(
        REPORT_DIR / "class_counts.csv"
    )

    # Cleaning report
    pd.DataFrame(
        [cleaning_report]
    ).to_csv(
        REPORT_DIR / "cleaning_report.csv",
        index=False
    )

    # Split summary
    df.groupby(
        ["label", "split"]
    ).size().reset_index(
        name="image_count"
    ).to_csv(
        REPORT_DIR / "split_summary.csv",
        index=False
    )

    # Print results
    print("\nCLASS DISTRIBUTION")
    print(counts)

    print("\nCLEANING REPORT")

    for key, value in cleaning_report.items():
        print(f"{key}: {value}")


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 45)
    print("EYE DISEASE DATA PREPARATION")
    print("=" * 45)

    records, cleaning_report = collect_images()

    if not records:
        raise ValueError("No usable images found.")

    print(
        f"\nFinal usable images: {len(records)}"
    )

    split_df = split_dataset(records)

    processed_df = preprocess_images(split_df)

    if len(processed_df) != len(split_df):
        raise RuntimeError(
            "Processed image count does not match split count."
        )

    generate_reports(
        processed_df,
        cleaning_report
    )

    print("\n" + "=" * 45)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 45)

    print(
        f"Total processed images: {len(processed_df)}"
    )

    print(f"Processed folder: {PROCESSED_DIR}")
    print(f"Reports folder: {REPORT_DIR}")
    print(f"Splits folder: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
