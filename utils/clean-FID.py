!pip install clean-fid
import os
from cleanfid import fid
import torchvision.utils
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, source_dir, target_dir, style_idx, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.style_idx = style_idx
        self.transform = transform
        source_images = set(os.listdir(source_dir))
        target_images = set(os.listdir(target_dir))
        self.images = list(source_images.intersection(target_images))
        if len(self.images) == 0:
            raise ValueError("No matching images found.")
        print(f"Found {len(self.images)} matching images for style {style_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        source_path = os.path.join(self.source_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        return source_image, target_image, self.style_idx


def load_model(model_path, device):
    """Load the saved model with DataParallel handling."""
    checkpoint = torch.load(model_path)
    # Initialize the model with the correct number of styles (3 in this case)
    model = ImprovedGenerator(num_styles=3).to(device)
    
    # Remove the 'module.' prefix from state_dict keys
    state_dict = checkpoint['generator']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def test_model(model, input_image, style_idx, device):
    """Run model inference with a specified style index."""
    with torch.no_grad():
        input_tensor = input_image.to(device)
        style_idx_tensor = torch.tensor([style_idx], device=device)
        output = model(input_tensor, style_idx_tensor)
    return output


# Dataset and transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

source_dir = '/kaggle/input/test-multiprompt/original_images'

target_dir_list = [
    'albino', 'blonde', 'gogh', 'greenlantern', 'old', 'sculpture', 'whitewalker'
]
target_dir_list = [f'/kaggle/input/test-multiprompt/{style}' for style in target_dir_list]

    

# Create indexed datasets
datasets = []
for style_idx, target_dir in enumerate(target_dir_list):
    dataset = ImageDataset(source_dir, target_dir, style_idx, transform)
    datasets.append(dataset)


# Load the model (from working or from input)
model_path = '/kaggle/input/modelprova/pytorch/default/1/best_model.pth'
# model_path = '/kaggle/working/models/best_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(model_path, device)

# 0 = albino, 1 = blonde, 2 = van gogh, 3 = green lantern, 4 = old, 5 = sculpture, 6 = whitewalker
style_array = [0, 1, 2, 3, 4, 5, 6]

# the idea is to create this structure
# working
#  |_diffusion 
#     |_ albino -> contains 30 modified taken from diffusion from albino (diffusion_0_0, ... diffusion_0_30)
#     |_ blonde -> contains 30 modified taken from diffusion from blonde (diffusion_1_0, ... diffusion_1_30)
#     |_ gogh -> contains 30 modified taken from diffusion from gogh (diffusion_2_0, ... diffusion_2_30)
#     |...
#  |_created
#     |_0 -> contains 30 images generated with albino prompt (generated_0_0, ... generated_0_29)
#     |_1 -> contains 30 images generated with blonde prompt (generated_1_0, ... generated_1_29)
#     |_2 -> contains 30 images generated with gogh prompt (generated_2_0, ... generated_2_29)
#     |...

# Generate 30 images for each style
for style in style_array:
    created_dir = f"/kaggle/working/created/{style}/"
    diffusion_dir = f"/kaggle/working/diffusion/{style}/"
    os.makedirs(created_dir, exist_ok=True)  # create directory modified/0,1,2
    os.makedirs(diffusion_dir, exist_ok=True)  # create directory original/

    dataset = datasets[style]
    
    for i in range(30):
        original_image, modified_image, _ = dataset[i] # extract from testset 30 original and modified images
        modified_image = modified_image.unsqueeze(0)  # add a dimension to solve some problems related to batch
        original_image = original_image.unsqueeze(0)

        # generate image from original taken from testset
        created_image = test_model(model, original_image, style, device)
        #print(f"Input shape: {created_image.shape}, Output shape: {created_image.shape}")
        
        # Save generated image in correct dir
        save_created_path = os.path.join(created_dir, f"created_{style}_{i}.png")
        torchvision.utils.save_image(created_image, save_created_path, normalize=True, format="PNG")

        print(f"Saved created image in: {save_created_path}")

        # save image generate by Diffusion in dir
        save_diffusion_path = os.path.join(diffusion_dir, f"diffusion_{style}_{i}.png")
        torchvision.utils.save_image(modified_image, save_diffusion_path, normalize=True, format="PNG")

        print(f"Saved diffusion image in: {save_diffusion_path}")

# Compute FID scores
created_dirs = [f'/kaggle/working/created/{i}' for i in range(len(datasets))]
diffusion_dirs = [f'/kaggle/working/diffusion/{i}' for i in range(len(datasets))]

fid_scores = [fid.compute_fid(created_dirs[i], diffusion_dirs[i]) for i in range(len(datasets))]

for style, score in enumerate(fid_scores):
    print(f"FID for style {style}: {score}")

median = np.median(fid_scores)
print(f"The overall FID score is {median}")
