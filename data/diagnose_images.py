
from pathlib import Path
from PIL import Image

ROOT = Path("datasets/raw/external_eye")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    total = 0
    small = 0
    unreadable = 0
    valid = 0

    for path in folder.rglob("*"):
        if path.suffix.lower() not in EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()

            if width < 64 or height < 64:
                small += 1
            else:
                valid += 1

        except Exception:
            unreadable += 1

    print(f"\n{folder.name}")
    print("Total:", total)
    print("Valid dimensions:", valid)
    print("Too small:", small)
    print("Unreadable:", unreadable)
