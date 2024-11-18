import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# Transformer block for feature processing
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
        # Store original batch size and channels
        batch_size, channels, height, width = x.size()
        
        # Reshape to sequence format for transformer
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)
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
        
        # Reshape back to original spatial dimensions
        x = x.permute(1, 2, 0).reshape(batch_size, channels, height, width)
        return x

# Generator with pre-trained ResNet
class E2GANGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Load pre-trained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Use first few layers of ResNet
        self.initial_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        
        # Freeze pre-trained layers
        for param in self.initial_layers.parameters():
            param.requires_grad = False
            
        # Transformer section
        self.ds_transformer = nn.Conv2d(512, 256, 3, stride=2, padding=1)
        self.transformer = TransformerBlock(256)
        self.us_transformer = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        
        # Decoder section
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x, text_embedding=None):
        x = self.initial_layers(x)
        x = self.ds_transformer(x)
        x = self.transformer(x, text_embedding)
        x = self.us_transformer(x)
        x = self.decoder(x)
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
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)

# Dataset class for image pairs
class ImagePairDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory not found: {source_dir}")
        if not os.path.exists(target_dir):
            raise ValueError(f"Target directory not found: {target_dir}")
            
        source_images = set(os.listdir(source_dir))
        target_images = set(os.listdir(target_dir))
        self.images = list(source_images.intersection(target_images))
        
        if len(self.images) == 0:
            raise ValueError("No matching images found")
            
        print(f"Found {len(self.images)} matching images")
        
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
            print(f"Error loading image {img_name}: {str(e)}")
            raise e

# Function to create comparison grids
def create_comparison_grid(source_images, generated_images, target_images, epoch, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Comparison Epoch {epoch}', fontsize=16, y=0.95)
    
    num_images = min(4, source_images.shape[0])
    titles = ['Input Image', 'Generated Image', 'Target Image']
    
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_epoch_{epoch}.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Training function
def train_e2gan(generator, discriminator, train_loader, num_epochs, device, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss functions
    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.00025, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.00020, betas=(0.5, 0.999))
    
    # Training metrics
    metrics_history = {
        'g_loss': [],
        'd_loss': [],
        'pixel_loss': []
    }
    
    for epoch in range(num_epochs):
        for i, (source, target) in enumerate(tqdm(train_loader)):
            batch_size = source.size(0)
            real = target.to(device)
            source = source.to(device)
            
            # Text embedding (dummy for now)
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
            loss_g_pixel = criterion_pixel(fake, real) * 70
            loss_g = loss_g_gan + loss_g_pixel
            loss_g.backward()
            optimizer_g.step()
            
            # Save metrics
            metrics_history['g_loss'].append(loss_g.item())
            metrics_history['d_loss'].append(loss_d.item())
            metrics_history['pixel_loss'].append(loss_g_pixel.item())
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
        
        # Save samples and metrics at specific epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = generator(source[:4], text_embedding[:, :4])
                create_comparison_grid(source[:4], fake.data, real[:4], epoch + 1)
                
                # Save metrics
                metrics_file = os.path.join(save_dir, f'metrics_epoch_{epoch + 1}.txt')
                with open(metrics_file, 'w') as f:
                    f.write(f"Training Metrics Epoch {epoch + 1}:\n")
                    f.write(f"Generator Loss: {np.mean(metrics_history['g_loss'][-100:]):.4f}\n")
                    f.write(f"Discriminator Loss: {np.mean(metrics_history['d_loss'][-100:]):.4f}\n")
                    f.write(f"Pixel Loss: {np.mean(metrics_history['pixel_loss'][-100:]):.4f}\n")
        
        # Save model checkpoint
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'metrics_history': metrics_history
            }
            torch.save(checkpoint, os.path.join(save_dir, f'e2gan_checkpoint_epoch_{epoch + 1}.pth'))

# Test function
def test_e2gan(generator, image_path, device, output_dir="test_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        raise ValueError(f"Test image not found: {image_path}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            text_embedding = torch.randn(16, 1, 256).to(device)
            fake = generator(image, text_embedding)
            
        # Save results
        result_path = os.path.join(output_dir, 'test_result.png')
        save_image(torch.cat((image, fake), -2), result_path, normalize=True)
        
        # Display results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(image.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(fake.cpu().numpy()[0], (1, 2, 0)) * 0.5 + 0.5)
        plt.title('Generated')
        plt.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'test_comparison.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise e
    
if __name__ == '__main__':
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set directories
        source_dir='/kaggle/input/originalimages/original_images'
        target_dir='/kaggle/input/modifiedimages/modified_images'
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
            # Load dataset and dataloader
        dataset = ImagePairDataset(source_dir, target_dir, transform=transform)  
        train_loader = DataLoader(dataset, batch_size=18, shuffle=True, num_workers=4)  
    
        # Initialize models
        generator = E2GANGenerator().to(device)  
        discriminator = Discriminator().to(device)  
    
        # Train the model
        train_e2gan(generator, discriminator, train_loader, num_epochs=150, device=device)  
    
        print("Training completed successfully.")
        
        torch.save(generator.state_dict(), '/kaggle/working/e2gan_generator.pth')
        torch.save(discriminator.state_dict(), '/kaggle/working/e2gan_discriminator.pth')
        print("Models saved successfully!")
    except Exception as e:  
        print(f"Error in main execution: {str(e)}")  
        raise e
