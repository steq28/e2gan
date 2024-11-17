# Import delle librerie necessarie
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm.notebook import tqdm

# Blocco ResNet
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

# Blocco Transformer
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
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)
        residual = x
        
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x
        
        if text_embedding is not None:
            residual = x
            x = self.norm1(x)
            x, _ = self.attention(x, text_embedding, text_embedding)
            x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        x = x.permute(1, 2, 0).reshape(x.size(1), x.size(2), 32, 32)
        return x

# Generator
class E2GANGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        self.initial = nn.Conv2d(input_channels, 64, 7, padding=3)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        self.resblocks = nn.ModuleList([
            ResNetBlock(256) for _ in range(3)
        ])
        
        self.transformer = TransformerBlock(256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
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

# Discriminator
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

# Dataset
class ImagePairDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        
        if not os.path.exists(source_dir):
            raise ValueError(f"Directory sorgente non trovata: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory target non trovata: {target_dir}")
            
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

# Funzione per creare griglie di confronto
def create_comparison_grid(source_images, generated_images, target_images, epoch, fid_score=None):
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Confronto Epoch {epoch}' + (f' - FID: {fid_score:.2f}' if fid_score is not None else ''), 
                fontsize=16, y=0.95)
    
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
    plt.savefig(f'newimages/confronto_epoch_{epoch}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Training function
def train_e2gan(generator, discriminator, train_loader, num_epochs, device):
    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Inizializza FID
    fid = FrechetInceptionDistance(feature=64).to(device)
    fid_scores = []
    
    os.makedirs("newimages", exist_ok=True)
    os.makedirs("newmodels", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_fid_scores = []
        
        for i, (source, target) in enumerate(tqdm(train_loader)):
            batch_size = source.size(0)
            real = target.to(device)
            source = source.to(device)
            
            text_embedding = torch.randn(16, batch_size, 256).to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            fake = generator(source, text_embedding)
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            pred_fake = discriminator(fake)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_pixel = criterion_pixel(fake, real) * 100
            loss_g = loss_g_gan + loss_g_pixel
            loss_g.backward()
            optimizer_g.step()
            
            # Calcola FID ogni 100 batch
            if i % 100 == 0:
                fid.reset()
                real_imgs = ((real + 1) * 0.5).clamp(0, 1)
                fake_imgs = ((fake + 1) * 0.5).clamp(0, 1)
                
                fid.update(real_imgs, real=True)
                fid.update(fake_imgs, real=False)
                
                fid_score = float(fid.compute())
                epoch_fid_scores.append(fid_score)
                fid_scores.append(fid_score)
                
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}] "
                      f"[FID: {fid_score:.2f}]")
                
                metrics_data = {
                    'epoch': epoch,
                    'batch': i,
                    'discriminator_loss': loss_d.item(),
                    'generator_loss': loss_g.item(),
                    'pixel_loss': loss_g_pixel.item(),
                    'fid_score': fid_score
                }
                
                with open(f'metrics/batch_metrics_e{epoch}_b{i}.txt', 'w') as f:
                    for key, value in metrics_data.items():
                        f.write(f"{key}: {value}\n")
        
        # FID medio dell'epoca
        epoch_fid_mean = np.mean(epoch_fid_scores) if epoch_fid_scores else 0
        
        # Plot FID progress
        if epoch % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(fid_scores, label='FID Score')
            plt.title(f'FID Score Progress - Epoch {epoch}')
            plt.xlabel('Steps (100 batches)')
            plt.ylabel('FID Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'metrics/fid_progress_e{epoch}.png')
            plt.close()
        
        # Salvataggio agli epoch 100 e 200
        if epoch + 1 in [100, 200]:
            with torch.no_grad():
                fake = generator(source[:4], text_embedding[:, :4])
                create_comparison_grid(source[:4], fake.data, real[:4], epoch + 1, epoch_fid_mean)
                
                metrics = {
                    'Loss Discriminator': loss_d.item(),
                    'Loss Generator': loss_g.item(),
                    'Loss Pixel': loss_g_pixel.item(),
                    'FID Score (Media Epoca)': epoch_fid_mean,
                    'FID Score (Ultimo Batch)': fid_scores[-1] if fid_scores else 0
                }
                
                with open(f'metrics/metriche_complete_e{epoch + 1}.txt', 'w') as f:
                    f.write(f"Metriche Training Epoch {epoch + 1}:\n")
                    for nome_metrica, valore in metrics.items():
                        f.write(f"{nome_metrica}: {valore:.4f}\n")
        
        # Salva modello finale
        if epoch + 1 == num_epochs:
            checkpoint = {
                'epoch': num_epochs,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'final_fid_score': epoch_fid_mean,
                'fid_history': fid_scores
            }
            torch.save(checkpoint, 'newmodels/e2gan_finale.pth')
            print(f"Modello salvato! FID Score finale (media): {epoch_fid_mean:.2f}")
            # Plot finale FID
            plt.figure(figsize=(12, 6))
            plt.plot(fid_scores)
            plt.title('FID Score Durante il Training')
            plt.xlabel('Steps (100 batches)')
            plt.ylabel('FID Score')
            plt.grid(True)
            plt.savefig('metrics/fid_finale.png')
            plt.close()

    return fid_scores

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
            
        # Salva risultato
        save_image(torch.cat((image, fake), -2), "newimages/test_result.png", normalize=True)
        
        # Visualizza risultato
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(image.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(fake.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Generata')
        plt.axis('off')
        
        plt.savefig('newimages/test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Errore durante il test: {str(e)}")
        raise e

# Funzione per caricare un modello salvato
def load_model(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        generator = E2GANGenerator().to(device)
        discriminator = Discriminator().to(device)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"Modello caricato dall'epoca {checkpoint['epoch']}")
        if 'final_fid_score' in checkpoint:
            print(f"FID Score finale: {checkpoint['final_fid_score']:.2f}")
        
        return generator, discriminator, checkpoint
        
    except Exception as e:
        print(f"Errore nel caricamento del modello: {str(e)}")
        raise e

def main():
    try:
        # Imposta device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizzo device: {device}")
        
        # Imposta percorsi
        source_dir = "e2gan/Images/original_images"
        target_dir = "e2gan/Images/less_modified_images"
        
        # Verifica directory
        if not os.path.exists(source_dir):
            raise ValueError(f"Directory sorgente non trovata: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory target non trovata: {target_dir}")
        
        # Crea directory per output
        os.makedirs("newimages", exist_ok=True)
        os.makedirs("newmodels", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)
        
        # Setup trasformazioni
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Inizializza modelli
        generator = E2GANGenerator().to(device)
        discriminator = Discriminator().to(device)
        
        # Crea dataset e dataloader
        dataset = ImagePairDataset(source_dir, target_dir, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=16,
            shuffle=True,
            num_workers=2
        )
        
        # Training
        print("Inizio training...")
        fid_scores = train_e2gan(generator, discriminator, dataloader, num_epochs=200, device=device)
        
        # Plot finale FID
        plt.figure(figsize=(12, 6))
        plt.plot(fid_scores)
        plt.title('FID Score - Training Completo')
        plt.xlabel('Steps (100 batches)')
        plt.ylabel('FID Score')
        plt.grid(True)
        plt.savefig('metrics/fid_storia_completa.png')
        plt.close()
        
        # Test
        test_image_path = "e2gan/obama.jpg"
        if os.path.exists(test_image_path):
            print("Testing del modello...")
            test_e2gan(generator, test_image_path, device)
        else:
            print(f"Immagine test non trovata: {test_image_path}")
            
        print("Training e test completati con successo!")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
        raise e

if __name__ == "__main__":    
    main()