"""
ICC (Image Complexity Classifier) module.

Keep inference lightweight: importing `icc` must not require PyTorch Lightning.
"""

from .inference import ConvNeXtICC, load_icc_checkpoint

__all__ = ["ConvNeXtICC", "load_icc_checkpoint"]

