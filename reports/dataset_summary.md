# Eye Disease Detection – Dataset Summary

## 1. Dataset Information

- **Dataset:** Eye Diseases Classification Dataset
- **Source:** Mendeley Data
- **Image type:** External-eye images
- **Original images:** 2,298
- **Unique images before cleaning:** 2,297
- **Final usable images:** 1,725
- **Image dimensions:** 224 × 224
- **Image format:** RGB JPEG

## 2. Dataset Classes

| Class | Original Images | Usable Images |
|---|---:|---:|
| Cataract | 543 | 541 |
| Conjunctivitis | 357 | 357 |
| Eyelid | 525 | 518 |
| Normal | 649 | 90 |
| Uveitis | 223 | 219 |
| **Total** | **2,297** | **1,725** |

## 3. Data Cleaning

- Checked image readability.
- Excluded images smaller than 64 × 64 pixels.
- Removed one exact duplicate image.
- No corrupted images were detected.

## 4. Preprocessing

- Converted images to RGB.
- Resized images to 224 × 224 pixels.
- Saved processed images in JPEG format.

## 5. Dataset Splitting

- Training: 70%
- Validation: 15%
- Testing: 15%
- Random seed: 42
- Stratified splitting was used.

## 6. Final Dataset Distribution

| Split | Images |
|---|---:|
| Training | 1,207 |
| Validation | 259 |
| Testing | 259 |
| **Total** | **1,725** |

## 7. Generated Files

- `datasets/splits/train.csv`
- `datasets/splits/val.csv`
- `datasets/splits/test.csv`
- `reports/class_counts.csv`
- `reports/cleaning_report.csv`
- `reports/split_summary.csv`

## 8. Important Notes

- The Normal class has fewer usable images because most images were smaller than the minimum accepted dimensions.
- Patient-level separation and medical label verification are pending.
- The dataset is intended for research and educational model development, not standalone medical diagnosis.

