# APGD-PyTorch-Implementation
This repository contains a PyTorch implementation of the Adversarial Projected Gradient Descent (APGD) attack.  It includes script along with a separate file for data management and model.
# APGD (PyTorch) – CIFAR-10 Example
(1,000 correctly-classified samples)

This repo provides an APGD implementation and a minimal example to evaluate robustness on CIFAR-10 using a ResNet-56 checkpoint.

## Requirements

-   Python 3.8+
-   `torch`
-   `torchvision`

### Local helper modules

This repository depends on the following files, which are included in this repo:
-   `DataManagerPytorch.py`
-   `ResNet.py` (with `resnet56(...)`)
-   `AttackWrappersAPGD.py`

### Installation

If needed, you can install the required Python packages with `pip`:
```bash
pip install torch torchvision
