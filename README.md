## Data Preparation

The Data Lead is responsible for dataset collection, cleaning, preprocessing and preparation for model training.

### Dataset
- Image type: External-eye images
- Classes: Cataract, Conjunctivitis, Eyelid, Normal, Uveitis
- Original unique images: 2,297
- Final usable images: 1,725

### Preprocessing
- Image validation
- Exact duplicate detection
- Minimum image size filtering (64 × 64)
- RGB conversion
- Image resizing (224 × 224)
- Stratified train/validation/test splitting

### Dataset Split

| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 1,207 |
| Validation | 15% | 259 |
| Testing | 15% | 259 |

Random seed: 42

### Data Scripts
- `data/prepare_data.py` – Main preprocessing pipeline
- `data/inspect_dataset.py` – Image validation
- `data/check_duplicates.py` – Duplicate detection
- `data/diagnose_images.py` – Image dimension analysis

### Reports
- `reports/dataset_summary.md`
- `reports/class_counts.csv`
- `reports/cleaning_report.csv`
- `reports/split_summary.csv`

**Note:** Raw and processed datasets are excluded from GitHub. Dataset access and licensing information should be maintained separately.
