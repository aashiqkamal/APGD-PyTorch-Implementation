# APGD-PyTorch-Implementation
This repository contains a PyTorch implementation of the Adversarial Projected Gradient Descent (APGD) attack.  It includes attack script along with a separate file for data management and model.
# APGD (PyTorch) – CIFAR-10 Example
(1,000 correctly-classified samples)

This repo provides an APGD implementation and a minimal example to evaluate robustness on CIFAR-10 using a ResNet-56 checkpoint.

## Requirements

-   Python 3.8+
-   `torch` (version 2.7.1)
-   `torchvision` (version 0.22.1)

### Local helper modules

This repository depends on the following files, which are included in this repo:
-   `DataManagerPytorch.py`
-   `ResNet.py` (with `resnet56(...)`)
-   `AttackWrappersAPGD.py`

### Installation

If needed, you can install the required Python packages with `pip`:
```bash
pip install torch torchvision

### Quickstart (evaluate APGD on 1,000 balanced clean samples):
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ResNet import resnet56
import DataManagerPytorch as DMP
from AttackWrappersAPGD import APGDNativePytorch as APGD_new

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# CIFAR-10 val loader (normalized to [-1,1])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
valLoader = DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=val_tf),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

# Load model & checkpoint
# Note: The checkpoint file `ModelResNet56-Run0.th` is not included in this repository.
# It can be provided upon request.
input_shape = [1, 3, 32, 32]
model = resnet56(inputShape=input_shape, dropOutRate=0, numClasses=10).to(device)
ckpt = torch.load("ModelResNet56-Run0.th", map_location=device)
state_dict = ckpt.get("state_dict", ckpt)
state_dict = {k.replace("module.",""): v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()
#clean accuracy
clean_acc = DMP.validateD(valLoader, model, device)
# Collect 1,000 correctly-classified samples balanced by class
cleanLoader1000 = DMP.GetCorrectlyIdentifiedSamplesBalanced(
    model,
    totalSamplesRequired=1000,
    dataLoader=valLoader,
    numClasses=10
)

# Convert to tensors, then wrap to batch size 32
x_clean, y_clean = DMP.DataLoaderToTensor(cleanLoader1000)
cleanLoader_32 = DMP.TensorToDataLoader(x_clean, y_clean, transforms=None, batchSize=32, randomizer=None)

# APGD attack params 
eps = 16/255
steps = 25
clip_min, clip_max = -1.0, 1.0       # matches Normalize((0.5,)*3, (0.5,)*3)

# Run APGD (bs=32)
advLoader_new = APGD_new(
    device=device,
    dataLoader=cleanLoader_32,
    model=model,
    eps_max=eps,
    num_steps=steps,
    clip_min=clip_min,
    clip_max=clip_max,
    random_start=False
)

# Export adversarial tensor if desired
x_adv_new, _ = DMP.DataLoaderToTensor(advLoader_new)
torch.save(x_adv_new, 'x_adv_bs32.pth')

# --- Robust accuracy ---
robust_acc_new = DMP.validateD(advLoader_new, model, device)
print(f"APGD (bs=32) robust acc: {robust_acc_new*100:.2f}%")
## Expected Output

When you run the `Quickstart` script, you should see output similar to the following:
Device: cuda
ResNet56 loaded.
Clean CIFAR-10 accuracy: 92.77%
1000 balanced clean samples ready.
x_adv_new generated and saved.
APGD attack (bs=32) robust acc: 0.00%
