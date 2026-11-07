
import itertools
import shutil
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "datasets" / "splits"
DUPLICATE_FILE = ROOT / "reports" / "perceptual_duplicates.csv"

OUTPUT_DIR = ROOT / "datasets" / "temporary_final" / "external_eye"
REPORT_DIR = ROOT / "reports" / "temporary_final"

RANDOM_STATE = 42

# Automatically group same-class pairs with distance <= 2
MAX_GROUP_DISTANCE = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLIT_FILES = [
    SPLIT_DIR / "train.csv",
    SPLIT_DIR / "val.csv",
    SPLIT_DIR / "test.csv"
]


# ==========================================
# 2. PATH NORMALIZATION
# ==========================================

def normalize_path(path):

    return str(path).replace("\\", "/").strip().lower()


# ==========================================
# 3. LOAD EXISTING SPLITS
# ==========================================

def load_dataset():

    frames = []

    for file in SPLIT_FILES:

        if not file.exists():
            raise FileNotFoundError(file)

        frames.append(pd.read_csv(file))

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df["path_key"] = df["original_path"].apply(
        normalize_path
    )

    if df["path_key"].duplicated().any():
        raise ValueError(
            "An original image occurs more than once in the split CSVs."
        )

    if df["sha256"].duplicated().any():
        raise ValueError(
            "Exact duplicate images exist across the original splits."
        )

    print("Total images:", len(df))
    print("Classes:", sorted(df["label"].unique()))

    return df


# ==========================================
# 4. UNION-FIND GROUPING
# ==========================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item for item in items
        }

        self.rank = {
            item: 0 for item in items
        }

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:

            self.rank[root_a] += 1


# ==========================================
# 5. CREATE IMAGE GROUPS
# ==========================================

def create_groups(df):

    duplicate_df = pd.read_csv(DUPLICATE_FILE)

    required = {
        "image_1",
        "image_2",
        "class_1",
        "class_2",
        "hamming_distance"
    }

    if not required.issubset(duplicate_df.columns):

        raise ValueError(
            "Perceptual duplicate CSV has missing columns."
        )

    uf = UnionFind(df["path_key"].tolist())

    available = set(df["path_key"])

    grouped_edges = []
    cross_class_edges = []
    uncertain_edges = []
    unmatched_edges = []

    for _, row in duplicate_df.iterrows():

        image_1 = normalize_path(row["image_1"])
        image_2 = normalize_path(row["image_2"])

        class_1 = str(row["class_1"]).lower()
        class_2 = str(row["class_2"]).lower()

        distance = int(row["hamming_distance"])

        if image_1 not in available or image_2 not in available:

            unmatched_edges.append(row.to_dict())
            continue

        if class_1 != class_2:

            cross_class_edges.append(row.to_dict())
            continue

        if distance <= MAX_GROUP_DISTANCE:

            uf.union(image_1, image_2)

            grouped_edges.append(row.to_dict())

        elif distance <= 5:

            uncertain_edges.append(row.to_dict())

    group_roots = [
    uf.find(item) for item in df["path_key"]
]

    df["group_id"] = pd.factorize(
    pd.Series(group_roots)
)[0]

    # Every group must contain only one class
    group_classes = df.groupby("group_id")["label"].nunique()

    if (group_classes > 1).any():

        raise ValueError(
            "A group contains multiple classes. Check labels."
        )

    print("\nGROUPING RESULTS")

    print("Automatically grouped pairs:", len(grouped_edges))
    print("Cross-class pairs flagged:", len(cross_class_edges))
    print("Uncertain same-class pairs:", len(uncertain_edges))
    print("Unmatched pairs:", len(unmatched_edges))
    print("Unique groups:", df["group_id"].nunique())

    return (
        df,
        grouped_edges,
        cross_class_edges,
        uncertain_edges,
        unmatched_edges
    )


# ==========================================
# 6. STRATIFIED GROUP SPLITTING
# ==========================================

def split_groups(df):

    X = df["path_key"].values
    y = df["label"].values
    groups = df["group_id"].values

    unique_groups = df["group_id"].nunique()

    if unique_groups < 10:
        raise ValueError(
            "Not enough independent groups for 10-fold splitting."
        )

    # First create approximately 70% train and 30% temporary.
    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(
        splitter.split(X, y, groups)
    )

    # Choose 3 folds that best approximate 30% of each class.
    class_counts = df["label"].value_counts().to_dict()

    best_score = float("inf")
    best_temp_indices = None

    for selected_folds in itertools.combinations(range(10), 3):

        temp_indices = sorted(
            set(
                index
                for fold_number in selected_folds
                for index in folds[fold_number][1]
            )
        )

        temp_df = df.iloc[temp_indices]

        score = 0.0

        for label, total in class_counts.items():

            actual = (
                temp_df["label"].eq(label).sum()
            )

            target = total * 0.30

            score += abs(actual - target) / max(total, 1)

        score += abs(
            len(temp_indices) - len(df) * 0.30
        ) / len(df)

        if score < best_score:

            best_score = score
            best_temp_indices = temp_indices

    train_indices = sorted(
        set(range(len(df))) - set(best_temp_indices)
    )

    train_df = df.iloc[train_indices].copy()
    temp_df = df.iloc[best_temp_indices].copy()

    # Divide temporary groups into validation and testing.
    if temp_df["group_id"].nunique() < 2:
        raise ValueError(
            "Not enough temporary groups for validation and testing."
        )

    second_splitter = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=RANDOM_STATE + 1
    )

    second_folds = list(
        second_splitter.split(
            temp_df["path_key"],
            temp_df["label"],
            temp_df["group_id"]
        )
    )

    # Select the fold closest to 50:50 class distribution.
    best_score = float("inf")
    best_val_indices = None

    for _, val_indices in second_folds:

        candidate = temp_df.iloc[val_indices]

        score = 0.0

        for label, total in temp_df["label"].value_counts().items():

            actual = candidate["label"].eq(label).sum()

            score += abs(actual - total * 0.50) / max(total, 1)

        if score < best_score:

            best_score = score
            best_val_indices = val_indices

    val_df = temp_df.iloc[best_val_indices].copy()

    test_indices = sorted(
        set(range(len(temp_df))) - set(best_val_indices)
    )

    test_df = temp_df.iloc[test_indices].copy()

    train_df["new_split"] = "train"
    val_df["new_split"] = "val"
    test_df["new_split"] = "test"

    result = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    # Verify that no group crosses splits.
    group_split_counts = result.groupby(
        "group_id"
    )["new_split"].nunique()

    if (group_split_counts > 1).any():

        raise RuntimeError(
            "Group leakage detected in the new split."
        )

    return result


# ==========================================
# 7. COPY CORRECTED DATASET
# ==========================================

def copy_corrected_dataset(df):

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename or back up it before running again."
        )

    output_records = []

    for _, row in df.iterrows():

        source = ROOT / Path(row["processed_path"])

        if not source.exists():

            raise FileNotFoundError(
                f"Processed image missing: {source}"
            )

        destination_dir = (
            OUTPUT_DIR /
            row["new_split"] /
            row["label"]
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            destination_dir /
            row["image_id"]
        )

        shutil.copy2(source, destination)

        record = row.to_dict()

        record["corrected_path"] = str(
            destination.relative_to(ROOT)
        ).replace("\\", "/")

        output_records.append(record)

    return pd.DataFrame(output_records)


# ==========================================
# 8. GENERATE REPORTS
# ==========================================

def generate_reports(df):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in ["train", "val", "test"]:

        split_df = df[
            df["new_split"] == split
        ]

        split_df.to_csv(
            REPORT_DIR / f"{split}.csv",
            index=False
        )

    counts = pd.crosstab(
        df["label"],
        df["new_split"]
    ).reindex(
        columns=["train", "val", "test"],
        fill_value=0
    )

    counts["total"] = counts.sum(axis=1)

    counts.to_csv(
        REPORT_DIR / "class_counts.csv"
    )

    df.to_csv(
        REPORT_DIR / "corrected_dataset_manifest.csv",
        index=False
    )

    print("\nCORRECTED DATASET DISTRIBUTION")
    print(counts)

    print("\nTotal images:", len(df))

    print(
        "Unique groups:",
        df["group_id"].nunique()
    )

    print(
        "\nCorrected dataset:",
        OUTPUT_DIR
    )


# ==========================================
# 9. MAIN
# ==========================================

def main():

    print("=" * 55)
    print("AUTOMATIC GROUP-BASED DATASET CORRECTION")
    print("=" * 55)

    if not DUPLICATE_FILE.exists():

        raise FileNotFoundError(DUPLICATE_FILE)

    df = load_dataset()

    (
        df,
        grouped_edges,
        cross_class_edges,
        uncertain_edges,
        unmatched_edges
    ) = create_groups(df)

    # Save audit information before splitting.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(grouped_edges).to_csv(
        REPORT_DIR / "automatically_grouped_pairs.csv",
        index=False
    )

    pd.DataFrame(cross_class_edges).to_csv(
        REPORT_DIR / "cross_class_review_required.csv",
        index=False
    )

    pd.DataFrame(uncertain_edges).to_csv(
        REPORT_DIR / "uncertain_pairs.csv",
        index=False
    )

    pd.DataFrame(unmatched_edges).to_csv(
        REPORT_DIR / "unmatched_pairs.csv",
        index=False
    )

    result = split_groups(df)

    corrected_df = copy_corrected_dataset(result)

    generate_reports(corrected_df)

    print("\nDATASET CORRECTION COMPLETED!")
    print("Original dataset remains untouched.")


if __name__ == "__main__":
    main()
