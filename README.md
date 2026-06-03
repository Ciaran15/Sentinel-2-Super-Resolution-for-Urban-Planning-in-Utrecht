# Sentinel-2 Super Resolution for Urban Planning in Utrecht

![Status](https://img.shields.io/badge/status-work_in_progress-orange)

This project focuses on enhancing and evaluating the spatial resolution of publicly available Sentinel-2 satellite imagery (10 m) using Dutch PDOK orthomosaics (7.5–25 cm) as ground truth data within the Utrecht municipal area.

The project compares AI-based image super-resolution methods, specifically:
- **Satlas** (GAN-based)
- **SwinIR** (Transformer-based)

The goal is to investigate whether Sentinel-2 imagery can be improved for applications such as:
- Urban planning
- Infrastructure mapping
- Disaster monitoring
- Environmental analysis

---

# Visual Comparison

| Satlas (GAN) | Low Resolution (Sentinel-2) | SwinIR (Transformer) |
|:---:|:---:|:---:|
| ![Satlas](https://raw.githubusercontent.com/Ciaran15/Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht/main/SatlasSuperResolution/Summer/Figures/Satlas_Summer.png) | ![Low Resolution](https://raw.githubusercontent.com/Ciaran15/Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht/main/SatlasSuperResolution/Summer/Figures/LR_Summer.png) | *Coming soon* |


---

# Objectives

This project aims to:
- Improve the spatial quality of Sentinel-2 imagery
- Compare GAN-based and Transformer-based super-resolution approaches
- Analyze both quantitative metrics and visual quality differences
- Investigate the usefulness of AI-enhanced satellite imagery for urban analysis

---

# Methods

The workflow consists of:

1. Sentinel-2 image collection
2. Image preprocessing and alignment
3. Super-resolution inference
4. Metric evaluation
5. Visual comparison analysis

### Models
- Satlas (GAN-based architecture)
- SwinIR (Transformer-based architecture)

### Processing Steps
- Sentinel-2 preprocessing
- Image alignment with PDOK orthomosaics
- Super-resolution inference using Satlas and SwinIR
- Quantitative evaluation using CLIP-Score and PSNR
- Visual comparison between outputs and ground truth imagery

### TODO
- Add preprocessing workflow diagram
- Add model architecture explanations
- Add inference settings and hyperparameters
- Add dataset preprocessing scripts

---

# Dataset

The dataset consists of:
- Sentinel-2 imagery
- PDOK orthomosaics as high-resolution reference data

### Seasonal Splits
- Summer Dataset
- Winter Dataset

### TODO
- Add dataset statistics
- Add image resolution examples
- Add train/validation/test split information
- Add download instructions

---

# Results & Evaluation

This repository evaluates image restoration and super-resolution results using multiple quality metrics.

## Quantitative Metrics

### CLIP-Score

CLIP-Score measures semantic similarity between images using the OpenAI CLIP model. Higher scores generally indicate that restored images preserve more meaningful visual content and structural information.

### PSNR

PSNR (Peak Signal-to-Noise Ratio) is a traditional pixel-based metric that compares reconstructed images to ground truth references. Higher PSNR values generally indicate lower reconstruction error.

---

# Why Metrics Are Not Everything

Although these metrics are useful for benchmarking and comparing models, they should not be treated as the final measure of quality.

In practice, visual perception matters far more than a single number. An image can achieve strong PSNR or CLIP scores while still appearing blurry, over-smoothed, or visually unnatural.

Ultimately, the most important question is whether the output is visually convincing and useful for real-world applications.

---

# Example Comparison

| Model | CLIP-Score ↑ | PSNR ↑ |
|-------|---------------|---------|
| SwinIR | X | X |
| Satlas | X | X |

In this example, one model may achieve better pixel accuracy while another produces more visually convincing or semantically meaningful results. The visually preferred output does not always correspond to the highest metric value.

---

# Visual Results

## Urban Areas
(Add urban comparison images here)

## Roads and Infrastructure
(Add zoomed-in road comparisons here)

## Vegetation and Water
(Add vegetation and water comparisons here)

## Failure Cases
(Add examples where models fail or produce artifacts)

### TODO
- Add side-by-side image comparisons
- Add animated GIF comparisons
- Add perceptual quality analysis
- Add detailed benchmark tables

---

# Key Findings

- GAN-based approaches produce sharper visual textures
- Transformer-based approaches preserve structural consistency
- Higher metric scores do not always correspond to visually preferred outputs
- Seasonal variation impacts reconstruction quality

### TODO
- Add quantitative findings
- Add seasonal analysis
- Add visual quality observations

---

# Limitations

Current limitations include:
- Cloud interference in Sentinel-2 imagery
- Temporal mismatch between datasets
- Seasonal lighting variations
- Limited availability of perfectly aligned ground truth data

### TODO
- Investigate domain adaptation techniques
- Add cloud removal preprocessing
- Improve temporal alignment

---

# Future Improvements

Potential future improvements include:
- Cloud and shadow detection
- Temporal image stacking
- Multi-spectral feature integration
- Additional evaluation metrics
- Fine-tuning on local Dutch satellite imagery
- Real-time inference optimization

---

# Project Structure

```bash
data/
models/
results/
notebooks/
scripts/
README.md
```

### TODO
- Organize preprocessing scripts
- Add evaluation pipeline
- Add automated inference scripts

---

# Installation

```bash
git clone https://github.com/Ciaran15/Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht/

cd Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht

pip install -r requirements.txt
```

### TODO
- Add CUDA requirements
- Add model download instructions
- Add environment setup instructions

---

# References

## Papers
- SwinIR: Image Restoration Using Swin Transformer
- Satlas Super-Resolution

## Data Sources
- Sentinel-2
- PDOK Orthomosaics

### TODO
- Add citation links
- Add BibTeX references
- Add related research papers
