import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import spectral_norm
from PIL import Image
import os
import numpy as np
from tqdm.notebook import tqdm
import torch_fidelity


# LoRA
class LoRALayer(nn.Module):
    # More alpha = stronger LoRA
    def __init__(self, original_layer, rank=14, alpha=1.0, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha  # Scaling factor for LoRA adjustments
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

        # LoRA decompositixon for Conv2d layers
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


def apply_lora_to_resnet_layer(layer, rank, alpha, dropout):
    """
    Wrap Conv2d, ConvTranspose2d, and Linear layers in LoRALayer within a given module.
    """
    for name, module in layer.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            setattr(layer, name, LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
            apply_lora_to_resnet_layer(module, rank, alpha, dropout)


# Generator
class ImprovedGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, dropout_rate=0.1, lora_rank=8):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Apply LoRA to resnet.layer1 and resnet.layer2
        self.resnet_layer1 = resnet.layer1
        apply_lora_to_resnet_layer(self.resnet_layer1, rank=lora_rank, alpha=1.0, dropout=dropout_rate)
        self.resnet_layer2 = resnet.layer2
        apply_lora_to_resnet_layer(self.resnet_layer2, rank=lora_rank, alpha=1.0, dropout=dropout_rate)

        # Downsample + Initial Layers with LoRA
        self.initial_layers = nn.Sequential(
            LoRALayer(resnet.conv1, rank=lora_rank),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            self.resnet_layer1,
            self.resnet_layer2
        )

        for param in self.initial_layers.parameters():
            param.requires_grad = False  # Freeze pretrained layers

        self.noise_layer = GaussianNoise(0.01)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Downsample Transformer
        self.ds_transformer = LoRALayer(nn.Conv2d(512, 256, 3, stride=2, padding=1), rank=lora_rank)

        # self.transformer = AntiOverfittingTransformerBlock(256, dropout_rate=dropout_rate)
        # Transformer
        self.transformer = nn.Transformer(
            d_model=256,
            nhead=8,  # Added number of attention heads
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )

        # Wrap in LoRA
        self.self_attention = LoRALayer(nn.Linear(256, 256), rank=lora_rank)
        self.cross_attention = LoRALayer(nn.Linear(256, 256), rank=lora_rank)

        # Upsample Transformer
        self.us_transformer = LoRALayer(
            spectral_norm(nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)), rank=lora_rank
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                LoRALayer(spectral_norm(nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)),
                          rank=lora_rank),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                LoRALayer(spectral_norm(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
                          rank=lora_rank),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                LoRALayer(spectral_norm(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                          rank=lora_rank),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_rate)
            )
        ])

        # Final Conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.noise_layer(x)
        features = self.initial_layers(x)
        x = self.ds_transformer(features)

        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # Transformer processing with LoRA-based attention
        x = self.self_attention(x)
        x = self.cross_attention(x)

        # Reshape back to conv format
        x = x.permute(0, 2, 1).view(b, c, h, w)

        x = self.dropout(x)
        x = self.us_transformer(x)
        for decoder_block in self.decoder_blocks:
            identity = x
            x = decoder_block(x)
            if x.size() == identity.size():
                x = x + identity
        return self.final_conv(x)


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


# Gaussian Noise Layer
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


# Image Pair Dataset
class ImagePairDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        source_images = set(os.listdir(source_dir))
        target_images = set(os.listdir(target_dir))
        self.images = list(source_images.intersection(target_images))
        if len(self.images) == 0:
            raise ValueError("No matching images found.")
        print(f"Found {len(self.images)} matching images")

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
        return source_image, target_image


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
            fake_image = generator(input_image)

            # Save fake image
            fake_image_path = os.path.join(fake_images_dir, f"fake_{idx}.png")
            save_image(fake_image.squeeze(0), fake_image_path, normalize=True)

    # Calculate FID using `torch_fidelity`
    metrics = torch_fidelity.calculate_metrics(
        input1=real_images_path,
        input2=fake_images_dir,
        cuda=torch.cuda.is_available(),
        isc=False,  # Inception Score Calculation
        fid=True,  # Fréchet Inception Distance
        kid=False,  # Kernel Inception Distance
        verbose=False
    )
    return metrics['frechet_inception_distance']


# Training Loop
def train_e2gan_with_regularization(generator, discriminator, train_loader, val_loader, num_epochs, device,
                                    save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    criterion_gan = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=num_epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=num_epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    best_fid_score = float('-inf')
    patience = 10
    early_stop_counter = 0
    train_metrics = {'g_loss': [], 'd_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        total_train_loss = 0
        for source, target in tqdm(train_loader):
            batch_size = source.size(0)
            real = target.to(device)
            source = source.to(device)
            optimizer_d.zero_grad()
            fake = generator(source)
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            real_labels = torch.ones_like(pred_real) * 0.9
            fake_labels = torch.zeros_like(pred_fake) * 0.1
            loss_d_real = criterion_gan(pred_real, real_labels)
            loss_d_fake = criterion_gan(pred_fake, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()
            pred_fake = discriminator(fake)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_pixel = criterion_pixel(fake, real) * 30
            loss_g = loss_g_gan + loss_g_pixel
            loss_g.backward()
            optimizer_g.step()
            train_metrics['g_loss'].append(loss_g.item())
            train_metrics['d_loss'].append(loss_d.item())
        generator.eval()
        total_val_loss = 0
        with torch.no_grad():
            for source, target in val_loader:
                source = source.to(device)
                target = target.to(device)
                fake = generator(source)
                val_loss = criterion_pixel(fake, target).item()
                total_val_loss += val_loss
        avg_val_loss = total_val_loss / len(val_loader)
        train_metrics['val_loss'].append(avg_val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs}: G_loss={np.mean(train_metrics['g_loss'][-len(train_loader):])}, D_loss={np.mean(train_metrics['d_loss'][-len(train_loader):])}, Val_loss={avg_val_loss}")

        # Calculate FID
        fake_images_dir = "/kaggle/working/fake_images"
        fid_score = calculate_fid(generator, "/kaggle/input/e2gan-data/modified_images", fake_images_dir, device)
        print(f"FID Score for Epoch {epoch + 1}: {fid_score:.4f}")

        if fid_score < best_fid_score:
            best_fid_score = fid_score
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'fid_score': fid_score
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_fid.pth"))
            print(f"Best model saved at epoch {epoch + 1} with FID {fid_score:.4f}!")

        scheduler_g.step()
        scheduler_d.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(
                {'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'epoch': epoch},
                os.path.join(save_dir, "best_model.pth"))
            print("Saved best model")
    return train_metrics


# Modify dataset creation to include validation split
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset
    full_dataset = ImagePairDataset(
        source_dir='/kaggle/input/e2gan-data/original_images',
        target_dir='/kaggle/input/e2gan-data/modified_images',
        transform=transform
    )

    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize models
    generator = torch.nn.DataParallel(ImprovedGenerator().to(device))
    discriminator = torch.nn.DataParallel(Discriminator().to(device))

    # Train with anti-overfitting techniques
    train_metrics = train_e2gan_with_regularization(
        generator,
        discriminator,
        train_loader,
        val_loader,
        num_epochs=100,
        device=device
    )
