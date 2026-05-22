# Sentinel-2 Super Resolution for Urban Planning in Utrecht
This project aims to enhance and compare the spatial resolution of publicly available Sentinel-2 satellite imagery (10 m) with Dutch PDOK orthomosaics (7.5-25 cm) as ground truth in the Utrecht municipal area. Using AI-based image super-resolution methods, more specifically Satlas super-resolution (GAN-based) and SwinIR (Transformer-based). Using Sentinel-2 data to support urban planning, infrastructure mapping, disaster monitoring, and environmental analysis.

# Objectives
What problem did we solve and why?

# Methodes
What methodes have we used to obtain these results?

# Dataset
What data did we use?
Summer Dataset
Winter Dataset

# Metrics Evaluation
This repository evaluates image restoration and super-resolution results using multiple quality metrics. Two commonly used metrics are:

CLIP-Score = Measures semantic similarity between images using the OpenAI CLIP model. Higher scores generally indicate that restored images preserve more meaningful visual content and structure.
PSNR (Peak Signal-to-Noise Ratio) = A traditional pixel-based metric that compares reconstructed images to ground truth references. Higher PSNR values usually mean lower reconstruction error.

While these metrics are useful for benchmarking and comparing models, they should not be treated as the final measure of quality. In practice, visual perception matters far more than a single number. An image can achieve strong PSNR or CLIP scores while still looking blurry, over-smoothed, or visually unnatural. Ultimately, the most important question is whether the output is actually useful and visually convincing to humans.


Example Comparison
Model	CLIP-Score ↑	PSNR ↑
SwinIR	X	            X
Satlas	X             X

In this example, ADD INFORMATION LATER!!

# Results
GAN vs. Transformer based models
Visuals

# Future Improvements
Cloud and shadow detection
stacked images

# How to run
git clone https://github.com/Ciaran15/Sentinel-2-Super-Resolution-for-Urban-Planning-in-Utrecht/
