#!/usr/bin/env python

import torch
import torchvision
import matplotlib.pyplot as plt
import random
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

# ==== CONFIG ====
MODEL_PATH = "checkpoints_2/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pick random test image
pic_number = random.randint(1, 3064)
IMAGE_PATH = f"brain_tumor_dataset/images/{pic_number}.png"
MASK_PATH = f"brain_tumor_dataset/masks/{pic_number}.png"

# ==== MODEL ====
# Create DeepLabV3 with ResNet50 backbone, 2 classes, with aux head
deeplab_v3 = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None, aux_loss=True, num_classes=2
)

# Replace first conv to handle grayscale input
deeplab_v3.backbone.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# === FREEZING (like notebook) ===
# Freeze backbone
for param in deeplab_v3.backbone.parameters():
    param.requires_grad = False

# Freeze classifier except final conv
for name, param in deeplab_v3.classifier.named_parameters():
    if "4" not in name:  # only keep last conv trainable
        param.requires_grad = False

# Freeze aux classifier except final conv
for name, param in deeplab_v3.aux_classifier.named_parameters():
    if "4" not in name:  # only keep last conv trainable
        param.requires_grad = False

deeplab_v3 = deeplab_v3.to(DEVICE)

# Load checkpoint
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
deeplab_v3.load_state_dict(state_dict)
deeplab_v3.eval()

# ==== TRANSFORMS ====
# Image transform (notebook version)
image_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(520, interpolation=InterpolationMode.BILINEAR),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# Mask transform (notebook version)
mask_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(520, interpolation=InterpolationMode.NEAREST),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
])

# ==== LOAD IMAGE & MASK ====
img = Image.open(IMAGE_PATH)
mask = Image.open(MASK_PATH)

orig_img = img.copy()
orig_mask = mask.copy()

img_t = image_transform(torch.tensor(np.array(img)))  # match notebook pipeline
img_t = img_t.unsqueeze(0).to(DEVICE)

mask_t = mask_transform(torch.tensor(np.array(mask)))

# ==== INFERENCE ====
with torch.no_grad():
    output = deeplab_v3(img_t)["out"]  # shape (1,2,h,w)
    prob = torch.softmax(output, dim=1)[0, 1].cpu()  # tumor channel
bin_mask = (prob > 0.5).float()

# ==== PLOT ====
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(orig_img, cmap="gray")
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(orig_img, cmap="gray")
ax[0, 1].imshow(bin_mask.numpy(), alpha=0.5, cmap="Reds")
ax[0, 1].set_title("Prediction Overlay")
ax[0, 1].axis("off")

ax[1, 0].imshow(bin_mask, cmap="gray")
ax[1, 0].set_title("Predicted Mask")
ax[1, 0].axis("off")

ax[1, 1].imshow(orig_mask, cmap="gray")
ax[1, 1].set_title("Ground Truth Mask")
ax[1, 1].axis("off")

plt.tight_layout()
plt.show()
