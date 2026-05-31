# AI-Powered Fashion Design Assistant
### Aarivya Labs Internship Program | H&P Projects
**Domain:** AI/ML | **Week:** 1 of 8 | **Focus:** Fashion Domain Research & Dataset Curation

---

## Project Overview

This project builds an end-to-end **Generative AI system for fashion design** using diffusion models, style transfer, and personalization pipelines. The goal is to create an AI assistant that can generate, recommend, and customize fashion designs.

**Tech Stack (Full Project):**
`Stable Diffusion XL` · `ControlNet` · `LoRA` · `CLIP` · `ChromaDB` · `Gradio` · `FastAPI` · `Celery` · `Docker` · `Kubernetes` · `Weights & Biases`

---

## Week 1 — Fashion Domain Research & Dataset Curation

### What Was Done

| Task | Status |
|------|--------|
| Studied fashion design workflow (11 stages: Inspiration → Final Presentation) | ✅ Complete |
| Mapped each design stage to its AI training equivalent | ✅ Complete |
| Explored DeepFashion dataset — all 8 modalities analyzed | ✅ Complete |
| Curated dataset: 12,694 samples with all 9 modalities per sample | ✅ Complete |
| Organized metadata — `metadata.json` + `metadata_summary.csv` | ✅ Complete |
| Built structured PyTorch data pipeline for model training | ✅ Complete |

---

## Dataset

**DeepFashion** was selected over FashionGen for its rich multi-modal annotations required for ControlNet and DensePose-conditioned generation.

### Curated Dataset Stats

| Metric | Value |
|--------|-------|
| Total raw images | ~44,096 |
| Valid curated samples | **12,694** |
| Skipped (incomplete modalities) | 31,402 |
| Files per sample | 9 |

### Per-Sample Folder Structure

```
curated_dataset/
└── MEN-Denim-id_00000080-01_7_additional/
    ├── image.jpg               # Garment photograph
    ├── caption.txt             # Text description
    ├── densepose.png           # IUV body surface map
    ├── segm.png                # Clothing region mask
    ├── shape.txt               # Silhouette label vector (12 values)
    ├── fabric.txt              # Fabric label vector (3 values)
    ├── pattern.txt             # Pattern label vector (3 values)
    ├── keypoints_loc.txt       # 21 joint XY coordinates (42 values)
    └── keypoints_vis.txt       # Joint visibility flags (21 values)
```

### Modality Breakdown

| Modality | Count | Training Use |
|----------|-------|-------------|
| Images | ~44,096 | Ground truth for diffusion model |
| DensePose | ~44,096 | Pose conditioning in ControlNet |
| Texture labels | ~44,096 | Fabric + pattern annotations |
| Captions | ~42,543 | Text conditioning via CLIP |
| Segmentation | ~12,701 | Garment region conditioning |
| Shape labels | ~12,701 | Silhouette-aware generation |
| Keypoints (loc) | ~12,701 | Skeleton pose conditioning |
| Keypoints (vis) | ~12,701 | Filters occluded joints |

---

## Repository Structure

```
week1/
├── curate_dataset.py       # Main curation pipeline
├── metadata_builder.py     # Builds metadata.json + metadata_summary.csv
├── fashion_pipeline.py     # PyTorch Dataset + DataLoader (Week 2 ready)
├── Week1_Progress_Report.docx
└── Week_1_notes.pdf
```

---

## Scripts

### `curate_dataset.py`
Filters DeepFashion and copies only complete samples (all 9 modalities present) into a clean `curated_dataset/` folder.

```bash
python curate_dataset.py
# Output: curated_dataset/ with 12,694 sample folders
```

### `metadata_builder.py`
Reads the curated dataset and builds structured metadata files with decoded label values.

```bash
python metadata_builder.py
# Output: metadata.json + metadata_summary.csv
```

### `fashion_pipeline.py`
PyTorch Dataset and DataLoader for model training. Ready for integration with SDXL in Week 2.

```python
from fashion_pipeline import get_dataloader
train_loader, val_loader = get_dataloader("curated_dataset/", batch_size=8)
```

> **Note:** Full pipeline testing will be completed in Week 2 once the SDXL training environment is set up.

---

## Fashion Workflow → AI Mapping

| Stage | Designer Does | AI Equivalent |
|-------|--------------|---------------|
| Inspiration | References from nature, movies, culture | Dataset collection |
| Research | Observes textures, shapes, moods | Dataset analysis |
| Moodboard | Visual collage of colors and vibes | CLIP embeddings |
| Color Story | Converts colors into mood | Color attribute labels |
| Textures | Selects materials — texture changes perception | Diffusion model feature learning |
| Fabric Selection | Different garments need different fabrics | Dataset diversity |
| Sketching | Rough sketches, silhouette definition | Stable Diffusion image generation |
| Sampling | Fabric testing, draping — drawing ≠ wearable | Multiple variation generation, CLIP + FID evaluation |
| Development | Constant redesigning — Generate → Evaluate → Fix | LoRA fine-tuning, hyperparameter tuning |
| Flats | Technical diagrams with exact proportions | ControlNet structural conditioning |
| Final Presentation | Polished renders, runway visualization | Deployment + Gradio showcase |

---

## Week 2 Preview

- Set up Stable Diffusion XL pipeline
- Experiment with prompt engineering
- Evaluate image quality using CLIP Score and FID
- Create reusable prompt template libraries
