import os
from PIL import Image
import shutil
from math import floor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch

# Utils functin to generate dataset for Pix2Pix
def combine_images(dir1, dir2, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List and sort files in each directory
    images1 = sorted([f for f in os.listdir(dir1) if f.endswith(('png'))])
    images2 = sorted([f for f in os.listdir(dir2) if f.endswith(('png'))])
    
    # Make sure we have the same number of images in both directories
    num_images = min(len(images1), len(images2))
    
    for i in range(num_images):
        # Open each image
        image1_path = os.path.join(dir1, images1[i])
        image2_path = os.path.join(dir2, images2[i])
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Ensure both images have the same height
        if image1.height != image2.height:
            new_height = max(image1.height, image2.height)
            image1 = image1.resize((image1.width, new_height), Image.ANTIALIAS)
            image2 = image2.resize((image2.width, new_height), Image.ANTIALIAS)

        # Create a new image with the combined width and same height
        combined_width = image1.width + image2.width
        combined_image = Image.new('RGB', (combined_width, image1.height))

        # Paste both images into the new image
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (image1.width, 0))

        # Save the combined image to the output directory
        output_path = os.path.join(output_dir, f'combined_{i + 1}.png')
        combined_image.save(output_path)
        print(f'Saved combined image to {output_path}')

def split_dataset(output_dir, split_dir):
    # Create the directories for train, test, and val if they don't exist
    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')
    val_dir = os.path.join(split_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all images in the output directory and sort them
    images = sorted([f for f in os.listdir(output_dir) if f.endswith(('png'))])
    num_images = len(images)

    # Calculate the number of images for each split
    train_count = floor(num_images * 0.8)
    test_count = floor(num_images * 0.1)
    val_count = num_images - train_count - test_count  # Remaining images go to val

    # Copy images to the respective directories
    for i, image in enumerate(images):
        src_path = os.path.join(output_dir, image)

        if i < train_count:
            dest_path = os.path.join(train_dir, image)
        elif i < train_count + test_count:
            dest_path = os.path.join(test_dir, image)
        else:
            dest_path = os.path.join(val_dir, image)

        shutil.copy(src_path, dest_path)
        print(f'Copied {image} to {dest_path}')


class ImagePairsDataset(Dataset):
    def __init__(self, originals_dir, filtered_dir, transform=None):
        self.originals_dir = originals_dir
        self.filtered_dir = filtered_dir
        self.transform = transform
        self.image_names = os.listdir(originals_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Percorsi delle immagini
        image_name = self.image_names[idx]
        original_path = os.path.join(self.originals_dir, image_name)
        filtered_path = os.path.join(self.filtered_dir, image_name)

        # Caricamento delle immagini
        original_image = Image.open(original_path)
        filtered_image = Image.open(filtered_path)
        
        # Applichiamo la trasformazione
        if self.transform:
            original_image = self.transform(original_image)
            filtered_image = self.transform(filtered_image)

        return original_image, filtered_image


if __name__ == '__main__':
    # Test Dataset entries
    torch.manual_seed(42)

    # Trasformazioni da applicare
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Creazione del dataset e DataLoader
    current_dir = os.path.dirname(os.path.abspath(__file__))
    originals_dir = os.path.join(current_dir, 'Images', 'original_images')
    filtered_dir = os.path.join(current_dir, 'Images', 'modified_images')
    
    dataset = ImagePairsDataset(originals_dir, filtered_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Trovate {len(dataset)} coppie di immagini.")

    # Iterazione e caricamento di un batch di tensori
    for original, target in data_loader:
        print("Originale:", original.shape)
        print("Target:", target.shape)
        break

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    batch_size = 32

    trainloader = DataLoader(train_set, batch_size=batch_size)
    validloader = DataLoader(val_set, batch_size=len(val_set))
    testloader = DataLoader(test_set, batch_size=len(test_set))

    print(len(train_set), len(val_set), len(test_set))


    # Affinca immagine originale e immagine modificata
    # Splitta in train, val e test
    combined = os.path.join(current_dir, 'Pix2Pix', 'data','modified_images')
    splitted = os.path.join(current_dir, 'Pix2Pix', 'data','potraits')
    combine_images(originals_dir, filtered_dir, combined)
    split_dataset(combined, splitted)