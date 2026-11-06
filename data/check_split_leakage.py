import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

pairs = pd.read_csv(ROOT / "reports" / "exact_hash_matches.csv")

split_files = {
    "train": ROOT / "datasets" / "splits" / "train.csv",
    "val": ROOT / "datasets" / "splits" / "val.csv",
    "test": ROOT / "datasets" / "splits" / "test.csv",
}

split_map = {}

for split, file_path in split_files.items():
    df = pd.read_csv(file_path)

    for col in df.columns:
        if "path" in col.lower() or "image" in col.lower():
            for path in df[col].dropna():
                split_map[str(path).replace("/", "\\").lower()] = split
            break

results = []

for _, row in pairs.iterrows():
    img1 = str(row["image_1"]).replace("/", "\\").lower()
    img2 = str(row["image_2"]).replace("/", "\\").lower()

    split1 = split_map.get(img1, "Not Found")
    split2 = split_map.get(img2, "Not Found")

    results.append({
        "image_1": row["image_1"],
        "image_2": row["image_2"],
        "split_1": split1,
        "split_2": split2,
        "leakage_risk": (
            "YES" if split1 != split2
            and split1 != "Not Found"
            and split2 != "Not Found"
            else "NO"
        )
    })

output = ROOT / "reports" / "split_leakage_audit.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("Total pairs checked:", len(results))
print("Pairs across different splits:",
      sum(r["leakage_risk"] == "YES" for r in results))
print("Report saved:", output)