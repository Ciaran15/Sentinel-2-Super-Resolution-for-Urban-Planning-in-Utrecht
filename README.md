# Sentinel-2 Super Resolution for Urban Planning in Utrecht

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

# Objectives

This project aims to:
- Improve the spatial quality of Sentinel-2 imagery
- Compare GAN-based and Transformer-based super-resolution approaches
- Evaluate the usefulness of AI-enhanced satellite imagery for urban analysis
- Analyze both quantitative metrics and visual quality differences

---

# Methods

The following methods were used:
- Sentinel-2 preprocessing
- Image alignment with PDOK orthomosaics
- Super-resolution inference using Satlas and SwinIR
- Quantitative evaluation using CLIP-Score and PSNR
- Visual comparison between outputs and ground truth imagery

---

# Dataset

The dataset consists of:
- Sentinel-2 imagery
- PDOK orthomosaics as high-resolution reference data

### Seasonal Splits
- Summer Dataset
- Winter Dataset

---

# Metrics Evaluation

This repository evaluates image restoration and super-resolution results using multiple quality metrics.

## CLIP-Score

CLIP-Score measures semantic similarity between images using the OpenAI CLIP model. Higher scores generally indicate that restored images preserve more meaningful visual content and structural information.

## PSNR

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

# Results

- GAN vs. Transformer-based model comparison
- Visual output comparisons
- Quantitative metric analysis

(Add result images and metric tables here)

---

# Future Improvements

Potential future improvements include:
- Cloud and shadow detection
- Temporal image stacking
- Multi-spectral feature integration
- Additional evaluation metrics
- Fine-tuning on local Dutch satellite imagery

---

# How to Run

```bash
git clone https://github.com/Ciaran15/Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht/
cd Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht
