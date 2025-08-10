import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFakeAutoEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(DeepFakeAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 112 x 112
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 56 x 56
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 28 x 28
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 14 x 14
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 7 x 7
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.unbottleneck = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 512 x 7 x 7
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 14 x 14
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 28 x 28
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 56 x 56
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 112 x 112
            
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 3 x 224 x 224
        )
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        
        # Decode
        unflattened = self.unbottleneck(latent)
        unflattened = unflattened.view(-1, 512, 7, 7)
        decoded = self.decoder(unflattened)
        
        return decoded, latent

class DeepFakeDetector(nn.Module):
    def __init__(self, autoencoder, threshold=0.1):
        super(DeepFakeDetector, self).__init__()
        self.autoencoder = autoencoder
        self.threshold = threshold
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        reconstructed, latent = self.autoencoder(x)
        reconstruction_error = F.mse_loss(x, reconstructed, reduction='none')
        reconstruction_error = reconstruction_error.view(reconstruction_error.size(0), -1).mean(dim=1)
        
        # Classification based on latent features
        classification_score = self.classifier(latent).squeeze()
        
        return {
            'reconstructed': reconstructed,
            'reconstruction_error': reconstruction_error,
            'classification_score': classification_score,
            'is_fake': (reconstruction_error > self.threshold) | (classification_score > 0.5)
        }
