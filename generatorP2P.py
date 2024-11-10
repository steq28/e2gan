import torch
import torch.nn as nn


# Funzione per il downsampling (simile alla funzione downsample di TensorFlow di P2P)
class Downsample(nn.Module):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        layers = [
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=size, stride=2, padding=1, bias=False)
        ]
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(filters))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Funzione per l'upsampling (simile alla funzione upsample di TensorFlow di P2P)
class Upsample(nn.Module):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=filters, out_channels=filters // 2, kernel_size=size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters // 2)
        ]
        
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Classe del generatore
class Generator(nn.Module):
    def __init__(self, output_channels):
        super(Generator, self).__init__()
        
        # Stack del downsampling
        self.down_stack = nn.ModuleList([
            Downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            Downsample(128, 4),  # (batch_size, 64, 64, 128)
            Downsample(256, 4),  # (batch_size, 32, 32, 256)
            Downsample(512, 4),  # (batch_size, 16, 16, 512)
            Downsample(512, 4),  # (batch_size, 8, 8, 512)
            Downsample(512, 4),  # (batch_size, 4, 4, 512)
            Downsample(512, 4),  # (batch_size, 2, 2, 512)
            Downsample(512, 4)   # (batch_size, 1, 1, 512)
        ])

        # ResNet block
        # ResNet block

        # Downsample

        # Transformer block

        # Updample

        # ResNet block
        
        # Stack dell'upsampling
        self.up_stack = nn.ModuleList([
            Upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            Upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            Upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            Upsample(512, 4),  # (batch_size, 16, 16, 1024)
            Upsample(256, 4),  # (batch_size, 32, 32, 512)
            Upsample(128, 4),  # (batch_size, 64, 64, 256)
            Upsample(64, 4)    # (batch_size, 128, 128, 128)
        ])
        
        # Ultimo layer di upsampling
        self.last = nn.ConvTranspose2d(in_channels=64, out_channels=output_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        skips = []
        
        # Downsampling
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        
        skips = skips[::-1][1:]  # Rimuove l'ultimo layer per le connessioni skip
        
        # Upsampling e connessioni skip
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
        
        x = self.last(x)
        x = torch.tanh(x)  # Simile all'activation='tanh' in TensorFlow
        
        return x
