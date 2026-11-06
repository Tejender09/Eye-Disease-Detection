
from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

DATASET = Path("datasets/raw/external_eye")
OUTPUT = Path("reports/dataset_samples.jpg")

CLASSES = [
    "Cataract",
    "Conjunctivitis",
    "Eyelid",
    "Normal",
    "Uveitis"
]

SAMPLES_PER_CLASS = 5
random.seed(42)

images = []

for class_name in CLASSES:

    folder = DATASET / class_name

    files = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    selected = random.sample(
        files,
        min(SAMPLES_PER_CLASS, len(files))
    )

    for path in selected:
        try:
            img = Image.open(path).convert("RGB")
            images.append((class_name, path.name, img))
        except Exception:
            continue

cell_width = 220
cell_height = 200

columns = 5
rows = math.ceil(len(images) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, (label, filename, img) in enumerate(images):

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    img.thumbnail((190, 145))

    sheet.paste(img, (x + 10, y + 10))

    draw.text(
        (x + 10, y + 160),
        label,
        fill="black"
    )

    draw.text(
        (x + 10, y + 178),
        filename[:28],
        fill="black"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT)

print("Sample visualization saved:", OUTPUT)
print("Total samples:", len(images))
