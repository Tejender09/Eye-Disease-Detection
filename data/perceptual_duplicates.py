
from pathlib import Path
from PIL import Image
import pandas as pd

# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "datasets" / "raw" / "external_eye"
REPORT = ROOT / "reports" / "perceptual_duplicates.csv"

HASH_SIZE = 8
THRESHOLD = 5

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =====================================
# 1. CALCULATE DIFFERENCE HASH
# =====================================

def dhash(image_path):

    with Image.open(image_path) as img:

        img = img.convert("L")
        img = img.resize((HASH_SIZE + 1, HASH_SIZE))

        pixels = img.load()
        value = 0

        for y in range(HASH_SIZE):
            for x in range(HASH_SIZE):

                value = (value << 1) | (
                    pixels[x, y] > pixels[x + 1, y]
                )

        return value


# =====================================
# 2. HAMMING DISTANCE
# =====================================

def hamming_distance(hash1, hash2):

    return bin(hash1 ^ hash2).count("1")


# =====================================
# 3. SCAN DATASET
# =====================================

def scan_images():

    images = []

    print("\nScanning dataset...")

    for path in sorted(DATASET.rglob("*")):

        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:

            image_hash = dhash(path)

            images.append({
                "path": str(path.relative_to(ROOT)),
                "class": path.parent.name,
                "hash": image_hash
            })

        except Exception as error:

            print("Skipped:", path, error)

    print("Images scanned:", len(images))

    return images


# =====================================
# 4. COMPARE IMAGE PAIRS
# =====================================

def compare_images(images):

    results = []

    total = len(images)

    print("\nComparing image pairs...")

    for i in range(total):

        for j in range(i + 1, total):

            first = images[i]
            second = images[j]

            distance = hamming_distance(
                first["hash"],
                second["hash"]
            )

            if distance <= THRESHOLD:

                same_class = (
                    first["class"] == second["class"]
                )

                similarity = (
                    1 - distance / (HASH_SIZE * HASH_SIZE)
                ) * 100

                results.append({

                    "image_1": first["path"],
                    "class_1": first["class"],

                    "image_2": second["path"],
                    "class_2": second["class"],

                    "hamming_distance": distance,

                    "hash_similarity_percent": round(
                        similarity, 2
                    ),

                    "pair_type": (
                        "Same Class"
                        if same_class
                        else "Cross Class"
                    ),

                    "manual_review_status": (
                        "Pending"
                    ),

                    "review_decision": "",

                    "review_notes": ""

                })

    return results


# =====================================
# 5. GENERATE DETAILED REPORT
# =====================================

def generate_report(results):

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "image_1",
        "class_1",
        "image_2",
        "class_2",
        "hamming_distance",
        "hash_similarity_percent",
        "pair_type",
        "manual_review_status",
        "review_decision",
        "review_notes"
    ]

    df = pd.DataFrame(results, columns=columns)

    if not df.empty:

        df = df.sort_values(
            by=[
                "pair_type",
                "hamming_distance"
            ],
            ascending=[True, True]
        )

    df.to_csv(REPORT, index=False)

    print("\n========== DUPLICATE AUDIT ==========")

    print("Total possible similar pairs:", len(df))

    if not df.empty:

        print(
            "Same-class pairs:",
            (df["pair_type"] == "Same Class").sum()
        )

        print(
            "Cross-class pairs:",
            (df["pair_type"] == "Cross Class").sum()
        )

        print("\nDistance distribution:")

        print(
            df["hamming_distance"]
            .value_counts()
            .sort_index()
        )

    print("\nReport saved:", REPORT)

    print("\nNo images were deleted or modified.")


# =====================================
# 6. MAIN
# =====================================

def main():

    print("=" * 45)
    print("PERCEPTUAL DUPLICATE AUDIT")
    print("=" * 45)

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    images = scan_images()

    results = compare_images(images)

    generate_report(results)


if __name__ == "__main__":
    main()
