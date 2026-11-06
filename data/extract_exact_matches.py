
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

source = ROOT / "reports" / "perceptual_duplicates.csv"
output = ROOT / "reports" / "exact_hash_matches.csv"

df = pd.read_csv(source)

exact = df[df["hamming_distance"] == 0].copy()

exact.to_csv(output, index=False)

print("Exact dHash matches:", len(exact))
print("Report saved:", output)
print("No images deleted.")
