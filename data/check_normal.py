
from pathlib import Path
from PIL import Image
from collections import Counter

folder = Path("datasets/raw/external_eye/Normal")

dimensions = Counter()

for path in folder.rglob("*"):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        continue

    try:
        with Image.open(path) as img:
            dimensions[img.size] += 1
    except Exception:
        pass

print("Normal image dimensions:")
for size, count in dimensions.most_common(20):
    print(f"{size}: {count} images")
