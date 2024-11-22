# Install Kaggle API if not already installed
!pip install -q kaggle

# Move kaggle.json to the correct location and set permissions
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset from Kaggle
!kaggle datasets download -d denislukovnikov/ffhq256-images-only -p /content

# Unzip the dataset
import zipfile
import os

dataset_zip_path = '/content/ffhq256-images-only.zip'  # Change this to the actual downloaded file name
dataset_extracted_dir = '/content/dataset_images'

if not os.path.exists(dataset_extracted_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_extracted_dir)
    print("Dataset extracted.")
else:
    print("Dataset already extracted.")

# Import libraries
import PIL
import torch
import matplotlib.pyplot as plt
import zipfile

!pip install diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Initialize the model pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Define output zip paths for storing original and modified images
original_zip_path = '/kaggle/working/original_images.zip'
modified_zip_path = '/kaggle/working/modified_images.zip'

# Function to load a local image
def load_image_from_local(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)  # Handle orientation based on EXIF data
    image = image.convert("RGB")                # Convert to RGB format if needed
    return image

# Define prompt for style transfer
prompt = "albino person"

# Number of images to process
max_images = 5000  # Replace with the number of images you want to process
processed_images = []

# Gather all filenames and sort them
all_filenames = []
for dirname, _, filenames in os.walk(dataset_extracted_dir):
    for filename in filenames:
        all_filenames.append(os.path.join(dirname, filename))

# Sort filenames to ensure consistent ordering and select the first n images
all_filenames = sorted(all_filenames)[:max_images]

# Open zip files to save images
with zipfile.ZipFile(original_zip_path, 'w') as original_zip, zipfile.ZipFile(modified_zip_path, 'w') as modified_zip:
    for count, image_path in enumerate(all_filenames):
        print(f"Number {count + 1}, Processing image: {image_path}")

        # Load the original image
        original_image = load_image_from_local(image_path)

        # Save the original image to the zip file
        with original_zip.open(f"{count + 1}.png", 'w') as img_file:
            original_image.save(img_file, format='PNG')

        # Apply style transfer to the image

        # van gogh
        # modified_images = pipe(prompt, image=orignal_image, num_inference_steps=10, image_guidance_scale=1).images

        # blonde
        # modified_images = pipe(prompt, image=original_image, num_inference_steps=10, image_guidance_scale=1.5).images
        
        # albino
        modified_images = pipe(prompt, image=original_image, num_inference_steps=10, image_guidance_scale=2).images

        # silver sculpture
        # modified_images = pipe(prompt, image=original_image, num_inference_steps=20).images

        modified_image = modified_images[0]

        # Save the modified image to the modified zip file
        with modified_zip.open(f"{count + 1}.png", 'w') as img_file:
            modified_image.save(img_file, format='PNG')

        # Store the modified image for display
        processed_images.append(modified_image)

print("Processing complete. Zipped files saved at:")
print(f"- Original images: {original_zip_path}")
print(f"- Modified images: {modified_zip_path}")
