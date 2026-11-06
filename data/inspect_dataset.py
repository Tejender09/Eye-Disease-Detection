
from pathlib import Path
from PIL import Image

DATASET = Path("datasets/raw/external_eye")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_folder in sorted(DATASET.iterdir()):
    if not class_folder.is_dir():
        continue

    total = 0
    valid = 0
    corrupted = 0

    for path in class_folder.rglob("*"):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            corrupted += 1

    print(f"\nClass: {class_folder.name}")
    print(f"Total: {total}")
    print(f"Valid: {valid}")
    print(f"Corrupted: {corrupted}")

print("\nInspection completed!")
