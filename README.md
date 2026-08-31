# VibraNet

Official implementation of **VibraNet**, a deep feature matching method for marker-free vibration measurement of low-texture structures.

> **Paper:** *A Deep Feature Matching Method for Marker-Free Vibration Measurement of Low-Texture Structures*

VibraNet is developed from the **LoFTR** coarse-to-fine matching framework and introduces vibration-oriented feature processing and displacement reconstruction components for low-texture structural surfaces.

## Overview

The proposed framework contains four main stages:

1. **CNN-based feature processing and extraction**
   - Information-aware Feature Shunting (IFS)
   - Improved CNN (Imp-CNN)
2. **Transformer-based feature augmentation**
3. **Coarse-to-fine feature matching**
4. **Sub-pixel vibration displacement reconstruction**

The method is designed for marker-free vibration measurement under challenging imaging conditions, including weak texture, motion blur, illumination variation, and dynamically changing image quality.

## Demo

A simple matching demo can be run with:

```bash
python demo.py
```

The visualization shows feature correspondences between two images, with matched points connected by lines.

## Installation

This repository uses a Python environment that differs slightly from the original LoFTR environment.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` specifies the following main dependencies:

- OpenCV
- Albumentations
- Ray
- Einops
- Kornia
- Loguru
- YACS
- tqdm
- Matplotlib
- h5py
- PyTorch Lightning
- TorchMetrics
- joblib

Please install a compatible **PyTorch/CUDA** version for your hardware before running the model. The provided requirements file does not pin a PyTorch version, so the exact version should be selected according to your GPU/CUDA environment. 

> **Note:** Large checkpoint files should preferably be distributed through GitHub Releases, Git LFS, Zenodo, or another dedicated model/data host rather than committed directly to the normal Git history.


## Relationship to LoFTR

VibraNet is built upon the LoFTR framework:

> J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detector-Free Local Feature Matching With Transformers,” *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8922–8931.

The original LoFTR code provides the basis for the coarse-to-fine feature matching framework. VibraNet modifies and extends this framework for marker-free vibration measurement of low-texture structures.

The main modifications include:

- Information-aware Feature Shunting (IFS)
- Improved CNN (Imp-CNN)
- Vibration-oriented feature processing
- Transformer-based feature augmentation
- Vibration displacement reconstruction and evaluation

Portions of this repository may be derived from or adapted from the original LoFTR implementation. The original copyright notices and license terms should be retained for redistributed derivative code.

## License

This repository contains original VibraNet code as well as code derived from or adapted from LoFTR.

The original LoFTR implementation is distributed under the **Apache License 2.0**. Please retain the corresponding copyright and license notices for any redistributed LoFTR-derived files.

For the exact licensing terms of the original LoFTR implementation, see:

- https://github.com/zju3dv/LoFTR

If this repository is released with additional third-party components, their corresponding licenses should also be retained and acknowledged.


## Acknowledgements

This project is based in part on the LoFTR framework. We thank the authors of LoFTR for making their implementation available to the research community.

## Contact

For questions regarding the VibraNet implementation, please open a GitHub Issue or contact the corresponding authors listed in the paper.
