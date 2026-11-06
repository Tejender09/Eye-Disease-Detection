
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

df = pd.read_csv("reports/perceptual_duplicates.csv")

cross = df[df["class_1"] != df["class_2"]]

cell_w, cell_h = 260, 210

sheet = Image.new(
    "RGB",
    (cell_w * 2, cell_h * len(cross)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, row in enumerate(cross.itertuples()):

    for j, path in enumerate([row.image_1, row.image_2]):

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((240, 160))

        x = j * cell_w
        y = i * cell_h

        sheet.paste(img, (x + 10, y + 10))

        label = row.class_1 if j == 0 else row.class_2

        draw.text(
            (x + 10, y + 175),
            f"{label}: {Path(path).name}",
            fill="black"
        )

sheet.save("reports/cross_class_review.jpg")

print("Cross-class pairs:", len(cross))
print("Saved: reports/cross_class_review.jpg")
