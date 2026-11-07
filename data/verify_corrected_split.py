
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "reports" / "temporary_final"
    / "corrected_dataset_manifest.csv"
)

DUPLICATES = (
    ROOT / "reports" / "perceptual_duplicates.csv"
)

OUTPUT = (
    ROOT / "reports" / "temporary_final"
    / "final_leakage_audit.csv"
)


def normalize(path):
    return str(path).replace("\\", "/").strip().lower()


def main():

    manifest = pd.read_csv(MANIFEST)
    duplicates = pd.read_csv(DUPLICATES)

    mapping = {}

    for _, row in manifest.iterrows():

        mapping[normalize(row["original_path"])] = {
            "split": row["new_split"],
            "group": row["group_id"],
            "label": row["label"]
        }

    results = []

    for _, row in duplicates.iterrows():

        image1 = normalize(row["image_1"])
        image2 = normalize(row["image_2"])

        first = mapping.get(image1)
        second = mapping.get(image2)

        if first is None or second is None:

            status = "PATH_NOT_FOUND"

        elif first["split"] != second["split"]:

            status = "CROSS_SPLIT_CANDIDATE"

        elif first["group"] == second["group"]:

            status = "SAME_GROUP_SAME_SPLIT"

        else:

            status = "SAME_SPLIT_DIFFERENT_GROUPS"

        results.append({
            "image_1": row["image_1"],
            "image_2": row["image_2"],
            "class_1": row["class_1"],
            "class_2": row["class_2"],
            "hamming_distance": row["hamming_distance"],
            "split_1": first["split"] if first else "Not Found",
            "split_2": second["split"] if second else "Not Found",
            "group_1": first["group"] if first else None,
            "group_2": second["group"] if second else None,
            "status": status
        })

    result = pd.DataFrame(results)

    result.to_csv(OUTPUT, index=False)

    print("\nFINAL LEAKAGE VERIFICATION")
    print("=" * 45)

    print("\n", result["status"].value_counts())

    print("\nCross-split candidates:")

    cross = result[
        result["status"] == "CROSS_SPLIT_CANDIDATE"
    ]

    print(cross[
        [
            "image_1",
            "image_2",
            "hamming_distance",
            "split_1",
            "split_2"
        ]
    ].to_string(index=False))

    print("\nReport saved:", OUTPUT)


if __name__ == "__main__":
    main()
