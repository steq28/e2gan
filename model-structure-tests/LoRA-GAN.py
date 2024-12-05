#%%
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
from tqdm import tqdm

import torch_fidelity

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
    def __init__(self, dim, num_heads=8, rank=4, alpha=1.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)

        # LoRA for attention layers
        self.lora_q = LoRALayer(nn.Linear(dim, dim), rank, alpha)
        self.lora_k = LoRALayer(nn.Linear(dim, dim), rank, alpha)
        self.lora_v = LoRALayer(nn.Linear(dim, dim), rank, alpha)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MLP with LoRA
        self.mlp = nn.Sequential(
            LoRALayer(nn.Linear(dim, dim * 4), rank, alpha),
            nn.GELU(),
            LoRALayer(nn.Linear(dim * 4, dim), rank, alpha)
        )

    def forward(self, x, text_embedding=None):
        batch_size, channels, height, width = x.shape

        # Reshape for attention (sequence format)
        x = x.view(batch_size, channels, height * width).permute(2, 0, 1)  # [seq_len, batch, dim]
        residual = x

        # Self-attention with LoRA
        q = self.lora_q(x)
        k = self.lora_k(x)
        v = self.lora_v(x)

        x = self.norm1(x)
        attn_output, _ = self.attention(q, k, v)
        x = residual + attn_output

        # Cross-attention (optional) with text embeddings
        if text_embedding is not None:
            residual = x
            x = self.norm1(x)
            x, _ = self.attention(x, text_embedding, text_embedding)
            x = residual + x

        # MLP with LoRA
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        # Reshape back to 4D
        x = x.permute(1, 2, 0).view(batch_size, channels, height, width)  # Match input shape
        return x


# LoRA
# More Alpha = more strong is LoRA
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=14, alpha=1.0, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha  # Scaling factor for LoRA adjustments
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

        # LoRA decomposition for Conv2d layers
        if isinstance(original_layer, nn.Conv2d):
            self.lora_up = nn.Conv2d(
                original_layer.in_channels,
                rank,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.lora_down = nn.Conv2d(
                rank,
                original_layer.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

        # LoRA decomposition for ConvTranspose2d layers
        elif isinstance(original_layer, nn.ConvTranspose2d):
            self.lora_up = nn.ConvTranspose2d(
                original_layer.in_channels,
                rank,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.lora_down = nn.ConvTranspose2d(
                rank,
                original_layer.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

        # LoRA decomposition for Linear layers
        elif isinstance(original_layer, nn.Linear):
            self.lora_up = nn.Linear(original_layer.in_features, rank, bias=False)
            self.lora_down = nn.Linear(rank, original_layer.out_features, bias=False)

        else:
            raise ValueError("LoRALayer only supports Conv2d, ConvTranspose2d, and Linear layers")

        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Compute the original output
        original_output = self.original_layer(x)

        # Compute the LoRA output
        if isinstance(self.original_layer, (nn.Conv2d, nn.ConvTranspose2d)):
            lora_output = self.lora_down(self.lora_up(x))
            # Ensure LoRA output has the same spatial dimensions as the original output
            if lora_output.shape[2:] != original_output.shape[2:]:
                lora_output = F.interpolate(
                    lora_output, size=original_output.shape[2:], mode='bilinear', align_corners=False
                )
        elif isinstance(self.original_layer, nn.Linear):
            lora_output = self.lora_down(self.lora_up(x))

        # Apply dropout to the LoRA output
        lora_output = self.dropout(lora_output)

        # Scale and combine outputs
        return original_output + self.alpha * lora_output

    
# Generator
class E2GANGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Initial conv
        self.initial = nn.Conv2d(input_channels, 64, 7, padding=3)
        
        # Downsampling layers with LoRA
        self.down1 = LoRALayer(nn.Conv2d(64, 128, 3, stride=2, padding=1))  # Down to 128x128
        self.down2 = LoRALayer(nn.Conv2d(128, 256, 3, stride=2, padding=1))  # Down to 64x64
        
        # ResNet blocks
        self.resblock1 = ResNetBlock(256)
        self.resblock2 = ResNetBlock(256)
        
        # Downsample before transformer
        self.ds_transformer = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # Down to 32x32
        
        # Transformer block
        self.transformer = TransformerBlock(256, num_heads=8, rank=4, alpha=1.0)
        
        # Upsample after transformer
        self.us_transformer = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)  # Up to 64x64
        
        # Third ResNet block
        self.resblock3 = ResNetBlock(256)
        
        # Upsampling layers with LoRA
        self.up1 = LoRALayer(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))  # Up to 128x128
        self.up2 = LoRALayer(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))  # Up to 256x256
        
        # Output layer
        self.output = nn.Conv2d(64, output_channels, 7, padding=3)
        
    def forward(self, x, text_embedding=None):
        # Initial and downsample
        x = self.initial(x)
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        
        # ResNet blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        
        # Transformer section
        x = self.ds_transformer(x)
        x = self.transformer(x, text_embedding)
        x = self.us_transformer(x)
        
        # Third ResNet block
        x = self.resblock3(x)
        
        # Upsample and output
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
    images_dir = "/kaggle/working/images"  # Percorso corretto per Kaggle
    os.makedirs(images_dir, exist_ok=True)  # Crea la directory se non esiste

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
    plt.savefig(f'{images_dir}/confronto_epoch_{epoch}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Function to calculate FID score
def calculate_fid(generator, real_images_path, fake_images_dir, device):
    os.makedirs(fake_images_dir, exist_ok=True)  # Create directory for fake images if it doesn't exist
    
    # Generate fake images
    generator.eval()
    with torch.no_grad():
        for idx, real_image in enumerate(os.listdir(real_images_path)):
            real_image_path = os.path.join(real_images_path, real_image)
            image = Image.open(real_image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_image = transform(image).unsqueeze(0).to(device)
            
            # Generate fake image
            text_embedding = torch.randn(16, 1, 256).to(device)  # Example random text embedding
            fake_image = generator(input_image, text_embedding)
            
            # Save fake image
            fake_image_path = os.path.join(fake_images_dir, f"fake_{idx}.png")
            save_image(fake_image, fake_image_path, normalize=True)

    # Calculate FID using `torch_fidelity`
    metrics = torch_fidelity.calculate_metrics(
        input1=real_images_path,
        input2=fake_images_dir,
        cuda=torch.cuda.is_available(),
        isc=False,  # Inception Score Calculation
        fid=True,  # Fréchet Inception Distance
        kid=False  # Kernel Inception Distance
    )
    return metrics['frechet_inception_distance']


# Funzione di training
def train_e2gan(generator, discriminator, train_loader, num_epochs, device):
    # Loss functions
    criterion_gan = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    os.makedirs("newimages", exist_ok=True)
    os.makedirs("newmodels", exist_ok=True)
    
    print(f'Training for {num_epochs} epochs.')
    
    best_loss_g = float('inf')  # Initialize best generator loss
    best_loss_pixel = float('inf')
    best_model_path = "newmodels/e2gan_best_gen.pth"
    best_pixel_path = "newmodels/e2gan_best_pixel.pth"
    
    prev_loss_d = None
    prev_loss_g = None
    prev_loss_pixel = None
    
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress", unit="epoch"):
        epoch_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for i, (source, target) in epoch_bar:
            batch_size = source.size(0)
            real = target.to(device)
            source = source.to(device)
            
            # Dummy text embedding
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
            
            # Match shapes before pixel loss
            if fake.shape != real.shape:
                fake = F.interpolate(fake, size=real.shape[2:], mode='bilinear', align_corners=False)

            loss_g_pixel = criterion_pixel(fake, real) * 50
            loss_g = loss_g_gan + loss_g_pixel
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            # Update previous losses
            prev_loss_d = loss_d.item()
            prev_loss_g = loss_g.item()
            prev_loss_pixel = loss_g_pixel.item()
            
            # Update progress bar description
            epoch_bar.set_postfix_str(
                f"\nD_loss: {loss_d.item():.4f}, "
                f"G_loss: {loss_g.item():.4f}, "
                f"P_loss: {loss_g_pixel.item():.4f}"
            )
            
        # Calculate FID after each epoch
        #fake_images_dir = "/kaggle/working/fake_images"
        #fid_score = calculate_fid(generator, "/kaggle/input/e2gan-data/original_images", fake_images_dir, device)
        #print(f"FID Score for Epoch {epoch + 1}: {fid_score:.4f}")
        
        # Save comparisons and models at specific intervals
        if epoch + 1 in [100, 199]:
            with torch.no_grad():
                fake = generator(source[:4], text_embedding[:, :4])
                create_comparison_grid(source[:4], fake.data, real[:4], epoch + 1)
        
        # Save the best model if the generator loss improves
        if loss_g.item() < best_loss_g:
            best_loss_g = loss_g.item()
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss_g': best_loss_g
            }
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with generator loss {best_loss_g:.4f}!")
        
        # Save the best model if the pixel loss improves
        if loss_g_pixel.item() < best_loss_pixel:
            best_loss_pixel = loss_g_pixel.item()
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss_pixel': loss_g_pixel
            }
            torch.save(checkpoint, best_pixel_path)
            print(f"Best model saved at epoch {epoch + 1} with pixel loss {best_loss_pixel:.4f}!")
        
        # Save the final model at the last epoch
        if epoch + 1 == num_epochs:
            final_model_path = 'newmodels/e2gan_finale.pth'
            checkpoint = {
                'epoch': num_epochs,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }
            torch.save(checkpoint, final_model_path)
            print(f"Final model saved at {final_model_path}!")

# Funzione di test
def test_e2gan(generator, image_path, device):
    if not os.path.exists(image_path):
        raise ValueError(f"Immagine test non trovata: {image_path}")
    
    transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            text_embedding = torch.randn(16, 1, 256).to(device)
            fake = generator(image, text_embedding)
        
        print("Image shape:", image.shape)
        print("Fake shape:", fake.shape)

        #save_image(torch.cat((image, fake), -2), "test_result.png", normalize=True)
        
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
        
        source_dir = "/kaggle/input/e2gan-data/original_images"
        target_dir = "/kaggle/input/e2gan-data/modified_images"
        
        if not os.path.exists(source_dir):
            raise ValueError(f"Directory sorgente non trovata: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Directory target non trovata: {target_dir}")
        
        # Setup trasformazioni immagini
        transform = transforms.Compose([
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
            batch_size=32,        # Dimensione batch
            shuffle=True,         # Mischia i dati
            num_workers=4        # Numero di worker per caricamento dati
        )
        
        # Training del modello
        train_e2gan(generator, discriminator, dataloader, num_epochs=100, device=device)
        
        # Test del modello
        model_path = "/kaggle/working/newmodels/e2gan_finale.pth"
        tester = E2GANGenerator()
        checkpoint = torch.load('newmodels/e2gan_finale.pth')
        tester.load_state_dict(checkpoint['generator_state_dict'])
        tester = tester.to(device)
        test_image_path = "/kaggle/input/obamaset/obama.jpg"
        
        if os.path.exists(test_image_path):
            test_e2gan(tester, test_image_path, device)
        else:
            print(f"Immagine test non trovata: {test_image_path}")
            
        tester = E2GANGenerator()
        checkpoint = torch.load('newmodels/e2gan_best_gen.pth')
        tester.load_state_dict(checkpoint['generator_state_dict'])
        tester = tester.to(device) 
        test_e2gan(tester, test_image_path, device)
        
        tester = E2GANGenerator()
        checkpoint = torch.load('newmodels/e2gan_best_pixel.pth')
        tester.load_state_dict(checkpoint['generator_state_dict'])
        tester = tester.to(device) 
        test_e2gan(tester, test_image_path, device)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
        raise e

if __name__ == "__main__":    
    main()