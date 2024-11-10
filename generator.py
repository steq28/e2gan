import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim


class ImagePairsDataset(Dataset):
    def __init__(self, originals_dir, filtered_dir, transform=None):
        self.originals_dir = originals_dir
        self.filtered_dir = filtered_dir
        self.transform = transform
        self.image_names = os.listdir(originals_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        original_path = os.path.join(self.originals_dir, image_name)
        filtered_path = os.path.join(self.filtered_dir, image_name)

        original_image = Image.open(original_path)
        filtered_image = Image.open(filtered_path)
        
        if self.transform:
            original_image = self.transform(original_image)
            filtered_image = self.transform(filtered_image)

        return original_image, filtered_image

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64):
        super(Generator, self).__init__()
        
        # Initial downsampling
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True)
        )
        
        # ResNet blocks
        self.resblocks = nn.Sequential(
            ResNetBlock(base_channels*4),
            ResNetBlock(base_channels*4),
            ResNetBlock(base_channels*4)
        )
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.resblocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            # Primo layer senza BatchNorm
            nn.Conv2d(input_channels * 2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Secondo layer
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Terzo layer
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Quarto layer
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer finale per output patch
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class GANTrainer:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_pixelwise = nn.L1Loss()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.lambda_pixel = 100

    def train_epoch(self, dataloader, epoch):
        for i, (real_A, real_B) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            batch_size = real_A.size(0)
            
            # Generate a batch of images
            fake_B = self.generator(real_A)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()
            
            # Calcola la dimensione dell'output del discriminatore
            pred_fake_shape = self.discriminator(real_A, fake_B.detach()).shape
            
            # Crea target della dimensione corretta
            valid = torch.ones((batch_size, *pred_fake_shape[1:]), requires_grad=False).to(self.device)
            fake = torch.zeros((batch_size, *pred_fake_shape[1:]), requires_grad=False).to(self.device)
            
            # Real loss
            pred_real = self.discriminator(real_A, real_B)
            loss_real = self.criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = self.discriminator(real_A, fake_B.detach())
            loss_fake = self.criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            self.optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            self.optimizer_G.zero_grad()
            
            # GAN loss
            pred_fake = self.discriminator(real_A, fake_B)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_pixel * loss_pixel
            loss_G.backward()
            self.optimizer_G.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
                
        return loss_D.item(), loss_G.item()


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset e DataLoader
    originals_dir = 'e2gan/or_images'
    filtered_dir = 'e2gan/mod_images'
    
    dataset = ImagePairsDataset(originals_dir, filtered_dir, transform=transform)
    
    # Split del dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Inizializzazione modelli
    generator = Generator()
    discriminator = Discriminator()
    
    # Trainer
    trainer = GANTrainer(generator, discriminator, device)
    
    # Training
    epochs = 100
    for epoch in range(epochs):
        loss_D, loss_G = trainer.train_epoch(train_loader, epoch)
        
        # Salva i checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': trainer.optimizer_G.state_dict(),
                'optimizer_D_state_dict': trainer.optimizer_D.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pt')

    print("Training completato!")