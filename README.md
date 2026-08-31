# VibraNet_
A Deep Feature Matching Method for Marker-Free Vibration Measurement of Low-Texture Structures

#Using from kornia
#VibraNet is integrated into kornia library since version 0.5.11.
pip install kornia
#Then you can import it as
from kornia.feature import LoFTR

Installation
# For full pytorch-lightning trainer features (recommended)
conda env create -f environment.yaml
conda activate loftr

# For the LoFTR matcher only
pip install torch einops yacs kornia
