
from pathlib import Path
import hashlib
from collections import defaultdict

DATASET = Path("datasets/raw/external_eye")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = defaultdict(list)

for path in DATASET.rglob("*"):
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(str(path))

duplicates = {
    digest: paths
    for digest, paths in hashes.items()
    if len(paths) > 1
}

print("Total unique images:", len(hashes))
print("Duplicate groups:", len(duplicates))
print("Extra duplicate images:", sum(len(v)-1 for v in duplicates.values()))

for paths in duplicates.values():
    print("\nDuplicate:")
    for path in paths:
        print(path)
