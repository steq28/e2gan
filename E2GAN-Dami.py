# Import delle librerie necessarie
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm.notebook import tqdm

# Blocco ResNet - utilizzato per estrarre features
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x + residual

# Blocco Transformer - per l'elaborazione delle features con attention
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, text_embedding=None):
        # Reshape per attention
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)
        residual = x
        
        # Self attention
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x
        
        # Cross attention con il testo se fornito
        if text_embedding is not None:
            residual = x
            x = self.norm1(x)
            x, _ = self.attention(x, text_embedding, text_embedding)
            x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        # Reshape ritorno
        x = x.permute(1, 2, 0).reshape(x.size(1), x.size(2), 32, 32)
        return x

# Generator del GAN
class E2GANGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Convoluzione iniziale
        self.initial = nn.Conv2d(input_channels, 64, 7, padding=3)
        
        # Downsampling
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Blocchi ResNet
        self.resblocks = nn.ModuleList([
            ResNetBlock(256) for _ in range(3)
        ])
        
        # Blocco Transformer
        self.transformer = TransformerBlock(256)
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        
        # Convoluzione output
        self.output = nn.Conv2d(64, output_channels, 7, padding=3)
        
    def forward(self, x, text_embedding=None):
        x = self.initial(x)
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        
        for block in self.resblocks:
            x = block(x)
            
        x = self.transformer(x, text_embedding)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = torch.tanh(self.output(x))
        return x

# Discriminator del GAN
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)

# Dataset personalizzato per coppie di immagini
class ImagePairDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # Verifica directory
        if not os.path.exists(source_dir):
            raise ValueError(f"Directory sorgente non trovata: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory target non trovata: {target_dir}")
            
        # Trova le immagini che esistono in entrambe le directory
        source_images = set(os.listdir(source_dir))
        target_images = set(os.listdir(target_dir))
        self.images = list(source_images.intersection(target_images))
        
        if len(self.images) == 0:
            raise ValueError("Nessuna immagine corrispondente trovata")
            
        print(f"Trovate {len(self.images)} immagini corrispondenti")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]
            source_path = os.path.join(self.source_dir, img_name)
            target_path = os.path.join(self.target_dir, img_name)
            
            source_image = Image.open(source_path).convert('RGB')
            target_image = Image.open(target_path).convert('RGB')
            
            if self.transform:
                source_image = self.transform(source_image)
                target_image = self.transform(target_image)
                
            return source_image, target_image
        except Exception as e:
            print(f"Errore caricamento immagine {img_name}: {str(e)}")
            raise e

# Funzione per salvare le griglie di confronto
def create_comparison_grid(source_images, generated_images, target_images, epoch):
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Confronto Epoch {epoch}', fontsize=16, y=0.95)
    
    num_images = min(4, source_images.shape[0])
    titles = ['Immagine Input', 'Immagine Generata', 'Immagine Target']
    
    for i in range(num_images):
        for j in range(3):
            plt.subplot(num_images, 3, i*3 + j + 1)
            
            if j == 0:
                img = source_images[i]
            elif j == 1:
                img = generated_images[i]
            else:
                img = target_images[i]
                
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            
            plt.imshow(img_np)
            plt.title(f'{titles[j]} {i+1}', fontsize=10)
            plt.axis('off')
    
    plt.figtext(0.05, 0.02, 
                f'Dettagli Griglia:\n'
                f'- Prima riga: Immagini originali\n'
                f'- Seconda riga: Immagini generate\n'
                f'- Terza riga: Immagini target\n', 
                fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'images/confronto_epoch_{epoch}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Funzione di training
def train_e2gan(generator, discriminator, train_loader, num_epochs, device):
    # Funzioni di loss
    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    os.makedirs("newimages", exist_ok=True)
    os.makedirs("newmodels", exist_ok=True)
    
    for epoch in range(num_epochs):
        for i, (source, target) in enumerate(tqdm(train_loader)):
            batch_size = source.size(0)
            real = target.to(device)
            source = source.to(device)
            
            # Embedding di testo dummy
            text_embedding = torch.randn(16, batch_size, 256).to(device)
            
            # Training Discriminator
            optimizer_d.zero_grad()
            fake = generator(source, text_embedding)
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # Training Generator
            optimizer_g.zero_grad()
            pred_fake = discriminator(fake)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_pixel = criterion_pixel(fake, real) * 100
            loss_g = loss_g_gan + loss_g_pixel
            loss_g.backward()
            optimizer_g.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
        
        # Salva confronti solo agli epoch 100 e 200
        if epoch + 1 in [100, 200]:
            with torch.no_grad():
                fake = generator(source[:4], text_embedding[:, :4])
                create_comparison_grid(source[:4], fake.data, real[:4], epoch + 1)
                
                # Salva anche le metriche
                metrics = {
                    'Loss Discriminator': loss_d.item(),
                    'Loss Generator': loss_g.item(),
                    'Loss Pixel': loss_g_pixel.item(),
                }
                
                with open(f'images/metriche_epoch_{epoch + 1}.txt', 'w') as f:
                    f.write(f"Metriche Training Epoch {epoch + 1}:\n")
                    for nome_metrica, valore in metrics.items():
                        f.write(f"{nome_metrica}: {valore:.4f}\n")
        
        # Salva il modello finale
        if epoch + 1 == num_epochs:
            checkpoint = {
                'epoch': num_epochs,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }
            torch.save(checkpoint, 'models/e2gan_finale.pth')
            print("Modello salvato!")

# Funzione di test
def test_e2gan(generator, image_path, device):
    if not os.path.exists(image_path):
        raise ValueError(f"Immagine test non trovata: {image_path}")
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            text_embedding = torch.randn(16, 1, 256).to(device)
            fake = generator(image, text_embedding)
            
        save_image(torch.cat((image, fake), -2), "test_result.png", normalize=True)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(image.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(fake.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Generata')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Errore durante il test: {str(e)}")
        raise e

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizzo device: {device}")
        
        source_dir = "e2gan/Images/original_images"
        target_dir = "e2gan/Images/less_modified_images"
        
        if not os.path.exists(source_dir):
            raise ValueError(f"Directory sorgente non trovata: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory target non trovata: {target_dir}")
        
        # Setup trasformazioni immagini
        transform = transforms.Compose([
            transforms.Resize(128),          # Ridimensiona a 128x128
            transforms.CenterCrop(128),      # Crop centrale 128x128
            transforms.ToTensor(),           # Converte in tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizza
        ])
        
        # Inizializza i modelli
        generator = E2GANGenerator().to(device)
        discriminator = Discriminator().to(device)
        
        # Crea dataset e dataloader
        dataset = ImagePairDataset(source_dir, target_dir, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=16,        # Dimensione batch
            shuffle=True,         # Mischia i dati
            num_workers=2         # Numero di worker per caricamento dati
        )
        
        # Training del modello
        train_e2gan(generator, discriminator, dataloader, num_epochs=200, device=device)
        
        # Test del modello
        test_image_path = "e2gan/obama.jpg"
        if os.path.exists(test_image_path):
            test_e2gan(generator, test_image_path, device)
        else:
            print(f"Immagine test non trovata: {test_image_path}")
            
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
        raise e

if __name__ == "__main__":    
    main()