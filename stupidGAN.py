import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch



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
    torch.manual_seed(42)

    # Trasformazioni da applicare
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Creazione del dataset e DataLoader
    originals_dir = './original_images'
    filtered_dir = './modified_images'
    
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
