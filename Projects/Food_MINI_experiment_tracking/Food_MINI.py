import torch
from going_modular import GPU_check, utils
from pathlib import Path

device = GPU_check.GPU_check()

# Load model
model = utils.create_effnetb0(OUT_FEATURES= 3, device=device)
model.load_state_dict(torch.load("07_effnetb0_data_20_10_epochs.pth"))

# Get list of all .png files in the directory
IMG_PATH = Path("TEST_IMGS/")
IMG_LIST = list(IMG_PATH.glob("*.png"))

# Plot all the results of imgs in TEST_IMGS folder
for img in IMG_LIST:
    utils.pred_and_plot_image(model=model,
                            image_path= img)
    