
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "reports"
SPLIT_DIR = ROOT / "datasets" / "splits"

DUPLICATE_FILE = REPORT_DIR / "perceptual_duplicates.csv"

OUTPUT_CSV = REPORT_DIR / "cross_split_audit.csv"
CONTACT_DIR = REPORT_DIR / "leakage_contact_sheets"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv"
}

PAIRS_PER_SHEET = 15
THUMBNAIL_SIZE = (240, 175)


# ==========================================
# 2. LOAD SPLIT INFORMATION
# ==========================================

def normalize_path(path):
    return str(path).replace("\\", "/").lower()


def load_split_mapping():

    split_mapping = {}

    for split_name, csv_path in SPLIT_FILES.items():

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        required = {"original_path", "split"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"Invalid columns in {csv_path}"
            )

        for _, row in df.iterrows():

            key = normalize_path(row["original_path"])

            split_mapping[key] = row["split"]

    return split_mapping


# ==========================================
# 3. LOAD DUPLICATE REPORT
# ==========================================

def load_duplicate_report():

    if not DUPLICATE_FILE.exists():
        raise FileNotFoundError(
            f"Duplicate report not found: {DUPLICATE_FILE}"
        )

    df = pd.read_csv(DUPLICATE_FILE)

    required = {
        "image_1",
        "image_2",
        "hamming_distance",
        "hash_similarity_percent",
        "pair_type"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Duplicate report is missing required columns."
        )

    return df


# ==========================================
# 4. MATCH IMAGES WITH THEIR SPLITS
# ==========================================

def identify_cross_split_pairs(df, split_mapping):

    results = []

    for _, row in df.iterrows():

        image_1 = normalize_path(row["image_1"])
        image_2 = normalize_path(row["image_2"])

        split_1 = split_mapping.get(
            image_1, "Not Found"
        )

        split_2 = split_mapping.get(
            image_2, "Not Found"
        )

        is_cross_split = (
            split_1 != "Not Found"
            and split_2 != "Not Found"
            and split_1 != split_2
        )

        if is_cross_split:

            risk = "POTENTIAL CROSS-SPLIT LEAKAGE"

        elif split_1 == "Not Found" or split_2 == "Not Found":

            risk = "PATH NOT FOUND"

        else:

            risk = "SAME SPLIT"

        results.append({
            "image_1": row["image_1"],
            "image_2": row["image_2"],
            "class_1": row.get("class_1", ""),
            "class_2": row.get("class_2", ""),
            "split_1": split_1,
            "split_2": split_2,
            "hamming_distance": row["hamming_distance"],
            "similarity_percent": row["hash_similarity_percent"],
            "pair_type": row["pair_type"],
            "risk": risk,
            "manual_decision": "PENDING",
            "review_notes": ""
        })

    return pd.DataFrame(results)


# ==========================================
# 5. IMAGE LOADING
# ==========================================

def load_thumbnail(image_path):

    full_path = ROOT / Path(str(image_path))

    try:

        with Image.open(full_path) as img:

            img = img.convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE)

            canvas = Image.new(
                "RGB",
                THUMBNAIL_SIZE,
                "white"
            )

            x = (THUMBNAIL_SIZE[0] - img.width) // 2
            y = (THUMBNAIL_SIZE[1] - img.height) // 2

            canvas.paste(img, (x, y))

            return canvas

    except Exception as error:

        print(f"Cannot load {image_path}: {error}")

        return Image.new(
            "RGB",
            THUMBNAIL_SIZE,
            "#cccccc"
        )


# ==========================================
# 6. CREATE VISUAL CONTACT SHEETS
# ==========================================

def create_contact_sheets(audit_df):

    CONTACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs = audit_df[
        audit_df["risk"] == "POTENTIAL CROSS-SPLIT LEAKAGE"
    ].reset_index(drop=True)

    if pairs.empty:

        print("No cross-split pairs found.")
        return

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    rows_per_sheet = PAIRS_PER_SHEET
    pair_height = 225

    total_sheets = math.ceil(
        len(pairs) / rows_per_sheet
    )

    for sheet_number in range(total_sheets):

        start = sheet_number * rows_per_sheet
        end = min(
            start + rows_per_sheet,
            len(pairs)
        )

        sheet_pairs = pairs.iloc[start:end]

        sheet = Image.new(
            "RGB",
            (540, len(sheet_pairs) * pair_height),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for local_index, (_, row) in enumerate(
            sheet_pairs.iterrows()
        ):

            y = local_index * pair_height

            image_1 = load_thumbnail(row["image_1"])
            image_2 = load_thumbnail(row["image_2"])

            sheet.paste(image_1, (10, y + 35))
            sheet.paste(image_2, (280, y + 35))

            pair_number = start + local_index + 1

            draw.text(
                (10, y + 5),
                f"PAIR {pair_number}",
                fill="black",
                font=font
            )

            draw.text(
                (10, y + 215),
                f"{row['class_1']} | {row['split_1']}",
                fill="black",
                font=font
            )

            draw.text(
                (280, y + 215),
                f"{row['class_2']} | {row['split_2']}",
                fill="black",
                font=font
            )

            draw.text(
                (10, y + 195),
                f"Hamming distance: {row['hamming_distance']} | "
                f"Similarity: {row['similarity_percent']}%",
                fill="blue",
                font=font
            )

        output_path = CONTACT_DIR / (
            f"contact_sheet_{sheet_number + 1}.jpg"
        )

        sheet.save(
            output_path,
            quality=90
        )

        print("Created:", output_path)


# ==========================================
# 7. GENERATE AUDIT REPORT
# ==========================================

def generate_report(audit_df):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    cross_split = audit_df[
        audit_df["risk"] == "POTENTIAL CROSS-SPLIT LEAKAGE"
    ]

    same_split = audit_df[
        audit_df["risk"] == "SAME SPLIT"
    ]

    missing = audit_df[
        audit_df["risk"] == "PATH NOT FOUND"
    ]

    print("\n" + "=" * 50)
    print("DATASET LEAKAGE AUDIT")
    print("=" * 50)

    print("Total pairs:", len(audit_df))
    print("Cross-split pairs:", len(cross_split))
    print("Same-split pairs:", len(same_split))
    print("Unmatched pairs:", len(missing))

    print("\nCross-split distribution:")

    print(
        cross_split.groupby(
            ["split_1", "split_2"]
        ).size()
    )

    print("\nHamming distance distribution:")

    print(
        cross_split["hamming_distance"].value_counts()
        .sort_index()
    )

    print("\nAudit report saved:", OUTPUT_CSV)


# ==========================================
# 8. MAIN
# ==========================================

def main():

    print("Starting dataset leakage audit...")

    split_mapping = load_split_mapping()

    duplicate_df = load_duplicate_report()

    audit_df = identify_cross_split_pairs(
        duplicate_df,
        split_mapping
    )

    generate_report(audit_df)

    create_contact_sheets(audit_df)

    print("\nAudit completed successfully!")


if __name__ == "__main__":
    main()
