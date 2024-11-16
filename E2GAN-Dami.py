import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, reduction_factor=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Downsampling
        self.down = nn.Conv2d(channels, channels, kernel_size=3, 
                             stride=reduction_factor, padding=1)
        
        # Upsampling
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=4,
                                    stride=reduction_factor, padding=1)
        
        # Transformer components
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.self_attention = nn.MultiheadAttention(channels, num_heads)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        
        # Downsampling
        x = self.down(x)
        
        # Normalization
        x = self.norm1(x)
        
        # Reshape per attention
        _, _, h_down, w_down = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)  # (h*w, batch, channels)
        
        # Self attention
        attn_output, _ = self.self_attention(x_flat, x_flat, x_flat)
        
        # Reshape back e prima skip connection
        x = attn_output.permute(1, 2, 0).view(b, c, h_down, w_down) + x
        
        # MLP
        x = self.norm2(x)
        x_mlp = x.view(b, c, -1).permute(0, 2, 1)
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp.permute(0, 2, 1).view(b, c, h_down, w_down)
        
        # Upsampling e seconda skip connection
        x = self.up(x)
        return x + identity

class E2GANGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        
        # Transformer block
        self.transformer = TransformerBlock(256)
        
        # Third residual block
        self.res3 = ResidualBlock(256)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.transformer(x)
        x = self.res3(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

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

        original_image = Image.open(original_path).convert("RGB")
        filtered_image = Image.open(filtered_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)
            filtered_image = self.transform(filtered_image)
        
        return original_image, filtered_image

class DatasetManager:
    def __init__(self, n_clusters=400):
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.eval()
        self.n_clusters = n_clusters
        
    def extract_features(self, dataset, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = self.feature_extractor.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        features = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feat = self.feature_extractor(images)
                features.append(feat.cpu().numpy())
                
        return np.concatenate(features)
    
    def create_clustered_dataset(self, original_dataset):
        # Estrai features
        features = self.extract_features(original_dataset)
        features = features.reshape(features.shape[0], -1)  # Appiattisci le features
        
        # Applica K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Seleziona le immagini più vicine ai centroidi
        selected_indices = []
        for i in range(self.n_clusters):
            cluster_points = features[clusters == i]
            cluster_indices = np.where(clusters == i)[0]
            
            # Trova il punto più vicino al centroide
            if len(cluster_points) > 0:  # Verifica che il cluster non sia vuoto
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
            
        return torch.utils.data.Subset(original_dataset, selected_indices)

def calculate_fid(real_features, fake_features):
    # Calcola media e covarianza per le features reali
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    # Calcola media e covarianza per le features generate
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    # Calcola la distanza tra le distribuzioni
    diff = mu1 - mu2
    
    # Calcola la matrice covarianza
    covmean = np.sqrt(sigma1.dot(sigma2))
    
    # Calcola FID
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid

def evaluate_model(generator, dataloader, device):
    generator.eval()
    feature_extractor = resnet50(pretrained=True).to(device)
    feature_extractor.eval()
    
    real_features = []
    fake_features = []
    
    with torch.no_grad():
        for real_A, real_B in dataloader:
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Genera immagini fake
            fake_B = generator(real_A)
            
            # Estrai features
            real_feat = feature_extractor(real_B).cpu().numpy()
            fake_feat = feature_extractor(fake_B).cpu().numpy()
            
            real_features.append(real_feat)
            fake_features.append(fake_feat)
    
    real_features = np.concatenate(real_features)
    fake_features = np.concatenate(fake_features)
    
    return calculate_fid(real_features, fake_features)

def train_e2gan(generator, discriminator, train_loader, val_loader, num_epochs, device, save_path):
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()
    lambda_pixel = 100

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LinearLR(optimizer_G, 
                                                      start_factor=1.0,
                                                      end_factor=0.5,
                                                      total_iters=num_epochs//2)
    lr_scheduler_D = torch.optim.lr_scheduler.LinearLR(optimizer_D,
                                                      start_factor=1.0,
                                                      end_factor=0.5,
                                                      total_iters=num_epochs//2)

    best_fid = float('inf')
    metrics = {
        'g_losses': [],
        'd_losses': [],
        'pixel_losses': [],
        'fid_scores': []
    }

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_pixel_losses = []
        
        for i, (real_A, real_B) in enumerate(train_loader):
            batch_size = real_A.size(0)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            valid = torch.ones((batch_size, 1, 16, 16), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1, 16, 16), requires_grad=False).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixel(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Save losses
            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append(loss_D.item())
            epoch_pixel_losses.append(loss_pixel.item())

            if i % 100 == 0:
                print(f"\rEpoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                      f"d_loss: {loss_D.item():.4f}, g_loss: {loss_G.item():.4f}", end="")

        # Save epoch metrics
        metrics['g_losses'].append(np.mean(epoch_g_losses))
        metrics['d_losses'].append(np.mean(epoch_d_losses))
        metrics['pixel_losses'].append(np.mean(epoch_pixel_losses))

        # Validation phase
        generator.eval()
        val_fid = evaluate_model(generator, val_loader, device)
        metrics['fid_scores'].append(val_fid)
        
        if val_fid < best_fid:
            best_fid = val_fid
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'metrics': metrics
            }, f'{save_path}/best_model.pt')

        # Step the schedulers
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        print(f"\nEpoch [{epoch}/{num_epochs}] "
              f"d_loss: {metrics['d_losses'][-1]:.4f}, "
              f"g_loss: {metrics['g_losses'][-1]:.4f}, "
              f"FID: {metrics['fid_scores'][-1]:.4f}")

        # Salva checkpoint periodico
        if (epoch + 1) % 10 == 0:
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'metrics': metrics
            }, f'{save_path}/checkpoint_epoch_{epoch+1}.pt')

    return generator, discriminator, metrics

def main():
    # Hyperparameters
    num_epochs = 200
    batch_size = 16
    image_size = 256
    n_clusters = 400  # Numero di cluster come nel paper
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transformations
    transforms_ = transforms.Compose([
        transforms.Resize((image_size, image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset completo
    full_dataset = ImagePairsDataset(
        originals_dir='./original_images',  # Modifica questi percorsi
        filtered_dir='./modified_images',   # con i tuoi
        transform=transforms_
    )
    
    # Split del dataset
    full_size = len(full_dataset)
    train_size = int(0.8 * full_size)
    val_size = int(0.1 * full_size)
    test_size = full_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Applica clustering solo al training set
    dataset_manager = DatasetManager(n_clusters=n_clusters)
    train_clustered = dataset_manager.create_clustered_dataset(train_dataset)
    print(f"Clustered training set size: {len(train_clustered)}")

    # DataLoaders
    train_loader = DataLoader(
        train_clustered,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = nn.DataParallel(E2GANGenerator()).to(device)
    discriminator = nn.DataParallel(Discriminator()).to(device)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Train
    save_path = 'checkpoints'
    generator, discriminator, metrics = train_e2gan(
        generator,
        discriminator,
        train_loader,
        val_loader,
        num_epochs,
        device,
        save_path
    )
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_fid = evaluate_model(generator, test_loader, device)
    print(f"Test FID score: {test_fid:.4f}")

if __name__ == "__main__":
    main()