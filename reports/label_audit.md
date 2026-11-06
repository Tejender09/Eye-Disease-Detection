# Dataset Label and Quality Audit

**Project:** Eye Disease Detection Using Deep Learning  
**Dataset:** External Eye Disease Image Dataset  
**Role:** Data Lead  
**Audit Status:** In Progress

---

## 1. Dataset Overview

The dataset contains external-eye images organized into five disease-related categories:

1. Cataract
2. Conjunctivitis
3. Eyelid
4. Normal
5. Uveitis

### Dataset Statistics

| Metric | Count |
|---|---:|
| Original images | 2,298 |
| Images after exact file duplicate removal | 2,297 |
| Corrupt images | 0 |
| Images excluded due to small dimensions | 572 |
| Final usable images | 1,725 |

### Final Class Distribution

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cataract | 379 | 81 | 81 | 541 |
| Conjunctivitis | 250 | 53 | 54 | 357 |
| Eyelid | 362 | 78 | 78 | 518 |
| Normal | 63 | 14 | 13 | 90 |
| Uveitis | 153 | 33 | 33 | 219 |
| **Total** | **1,207** | **259** | **259** | **1,725** |

The dataset was divided into approximately 70% training, 15% validation, and 15% testing using stratified image-level splitting.

---

## 2. Image Quality Assessment

The dataset was inspected for image dimensions, corruption, and suitability for preprocessing.

### Findings

- No corrupt images were identified.
- Images smaller than 64 × 64 pixels were excluded.
- A total of 572 images were excluded due to insufficient dimensions.
- The Normal class was most affected, with 559 images excluded.
- The remaining 1,725 images were prepared for model development.
- Processed images were converted to RGB and resized to 224 × 224 pixels.

### Observation

The Normal class contains only 90 usable images, creating substantial class imbalance.

**Recommendation:** Consider class weighting and training-time augmentation. Evaluate per-class recall and F1-score rather than relying on accuracy alone.

---

## 3. Visual Label Inspection

A random sample of 25 images was generated using `data/visualize_dataset.py`, with five images selected from each class.

The visualization was saved as:

`reports/dataset_samples.jpg`

### Class-Wise Observations

**Cataract**
- Some images show visible opacity consistent with cataract-related appearance.
- Other images are difficult to interpret using external-eye appearance alone.
- Clinical label verification is recommended.

**Conjunctivitis**
- Several images show visible redness and inflammation.
- The sampled images generally appear consistent with the intended category.
- Visual appearance alone cannot confirm every diagnosis.

**Eyelid**
- The class contains images showing different eyelid conditions.
- The category is broad and may include multiple underlying disorders.
- The original dataset's labeling criteria should be reviewed.

**Normal**
- Sampled images generally appear normal.
- A large number of original Normal images were below the minimum resolution threshold.
- The reduced number of usable images may affect model generalization.

**Uveitis**
- Some images show visible eye abnormalities.
- Uveitis cannot be reliably confirmed from external photographs alone.
- Clinical annotations or original source documentation should be verified.

### Visual Audit Conclusion

The preliminary sample inspection identified potential label ambiguity, particularly in the Cataract, Eyelid, and Uveitis categories.

No images were automatically relabeled or removed based on visual appearance.

---

## 4. Exact File Duplicate Investigation

The dataset was examined for exact file duplicates using `data/check_duplicates.py`.

### Findings

- One exact duplicate pair was initially identified in the Cataract class.
- The duplicate involved `184.JPG` and `534.JPG`.
- One duplicate file was removed during the initial cleaning process.
- A subsequent exact duplicate scan reported no remaining exact file duplicate groups.

**Status:** Exact file duplicate scan completed.

---

## 5. Perceptual Duplicate Investigation

The dataset was examined using a difference hash (dHash) algorithm.

**Script:** `data/perceptual_duplicates.py`  
**Report:** `reports/perceptual_duplicates.csv`

### Method

- Perceptual hashing algorithm: dHash
- Hash length: 64 bits
- Hamming distance threshold: 5
- Images scanned: 2,297

### Results

| Metric | Count |
|---|---:|
| Potentially similar image pairs | 243 |
| Same-class pairs | 239 |
| Cross-class pairs | 4 |
| Pairs with Hamming distance 0 | 46 |
| Pairs with Hamming distance 1–5 | 197 |

### Hamming Distance Distribution

| Distance | Pairs |
|---|---:|
| 0 | 46 |
| 1 | 35 |
| 2 | 38 |
| 3 | 32 |
| 4 | 35 |
| 5 | 57 |
| **Total** | **243** |

### Observations

- Several similar pairs contain filenames beginning with `aug_`, suggesting possible augmentation-derived images.
- Multiple pairs have identical dHash values.
- Identical dHash values do not necessarily mean files are pixel-identical.
- Perceptual similarity scores are hash-based measurements and should not be interpreted as verified clinical or visual similarity percentages.

**Status:** Automated perceptual similarity detection completed; manual verification remains pending.

---

## 6. Cross-Class Similarity Investigation

Four potentially similar image pairs were identified across different class labels.

| Image 1 | Image 2 | Hamming Distance |
|---|---|---:|
| Conjunctivitis/617.png | Uveitis/203.jpeg | 3 |
| Cataract/229.jpg | Normal/359.jpg | 5 |
| Conjunctivitis/680.png | Eyelid/204.jpeg | 5 |
| Conjunctivitis/aug_832.jpg | Uveitis/13.jpeg | 5 |

### Findings

These pairs require manual review to determine whether they represent:

- The same underlying image with different labels.
- Cropped or transformed versions of the same image.
- Different images with similar visual structures.
- False-positive perceptual hash matches.

No cross-class pair has been confirmed as mislabeled solely from its hash distance.

**Status:** Manual verification pending.

---

## 7. Exact Perceptual Hash Investigation

A separate report was generated to extract pairs with a Hamming distance of 0.

**Script:** `data/extract_exact_matches.py`  
**Report:** `reports/exact_hash_matches.csv`

### Findings

- Exact dHash matches identified: 46 pairs.
- All 46 pairs belong to the same class.
- Several pairs include filenames beginning with `aug_`.
- Some images appear in multiple pairwise comparisons.

A Hamming distance of 0 means the images produce identical dHash values. It does not establish byte-for-byte identity.

**Status:** Extraction completed; manual visual verification pending.

---

## 8. Exact Perceptual Hash Split Leakage Audit

**Script:** `data/check_split_leakage.py`  
**Report:** `reports/split_leakage_audit.csv`

### Results

| Check | Result |
|---|---:|
| Exact dHash pairs examined | 46 |
| Cross-split pairs reported | 0 |
| Leakage detected by the current audit | None |

### Interpretation

The automated script reported zero cross-split matches among the 46 pairs with identical dHash values.

This is a positive preliminary finding. However, the script's path-matching logic has not yet been independently verified against the split CSV column formats.

Consequently, the reported result should not be treated as definitive proof that all 46 pairs were assigned to the same split.

### Remaining Verification

- Confirm that image paths in the duplicate report match the paths in the split manifests.
- Confirm the number of matched and unmatched images.
- Check perceptually similar pairs with Hamming distances of 1–5.
- Review potential augmentation-derived image overlap.

**Status:** Automated audit completed; path-matching verification pending.

---

## 9. Patient-Level Data Leakage Investigation

### Findings

- Original filenames are primarily numbered image files.
- No patient identification metadata was found in the inspected dataset.
- No patient-level grouping information was available.
- Train, validation, and test sets were generated using stratified image-level splitting.

### Risk

Different photographs from the same patient could potentially appear in different splits.

Without patient identifiers, patient-level separation cannot be confirmed.

**Status:** Patient-level separation unverified.

---

## 10. Dataset Cleaning and Preprocessing

**Script:** `data/prepare_data.py`

### Completed Operations

1. Scanned the raw external-eye dataset.
2. Checked image readability and dimensions.
3. Excluded images smaller than 64 × 64 pixels.
4. Converted accepted images to RGB.
5. Resized images to 224 × 224 pixels.
6. Created stratified training, validation, and testing splits.
7. Generated split manifests and cleaning reports.

### Generated Files

- `datasets/splits/train.csv`
- `datasets/splits/val.csv`
- `datasets/splits/test.csv`
- `reports/class_counts.csv`
- `reports/cleaning_report.csv`
- `reports/split_summary.csv`
- `reports/dataset_summary.md`
- `reports/dataset_samples.jpg`
- `reports/perceptual_duplicates.csv`
- `reports/exact_hash_matches.csv`
- `reports/split_leakage_audit.csv`

---

## 11. Outstanding Issues and Recommendations

| Issue | Recommended Action | Status |
|---|---|---|
| Normal class imbalance | Use class weighting and training-time augmentation | Pending |
| Possible perceptual duplicates | Manually review flagged pairs | Pending |
| Cross-class similar images | Verify labels against original dataset documentation | Pending |
| Exact dHash split audit | Verify path matching and unmatched records | Pending |
| Near-duplicate split leakage | Audit pairs with distances 1–5 | Pending |
| Patient-level separation | Obtain patient metadata if available | Unverified |
| Clinical label validity | Review original annotations and diagnostic definitions | Pending |

---

## 12. Final Audit Summary

The Data Lead preprocessing pipeline has been implemented successfully.

### Completed Work

- Dataset folder inspection
- Corrupt image detection
- Image dimension validation
- Exact file duplicate scanning
- Image preprocessing
- Stratified train/validation/test splitting
- Class distribution reporting
- Preliminary visual label inspection
- Perceptual duplicate detection
- Extraction of 46 exact dHash pairs
- Initial automated split leakage audit
- Dataset documentation

### Remaining Work

- Verify split audit path matching.
- Audit all 197 additional perceptually similar pairs.
- Manually review the four cross-class pairs.
- Investigate possible augmentation-derived duplicates.
- Record unresolved label and patient-level limitations.

**Overall Status: Core Data Lead preprocessing completed. Final dataset quality and leakage validation remain in progress.**

No additional images should be deleted or relabeled until flagged cases have been reviewed and documented.