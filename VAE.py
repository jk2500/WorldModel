import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Convolution stack
        # 64x64 -> 31x31
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        # 31x31 -> 14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        # 14x14 -> 6x6
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        # 6x6 -> 2x2
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # After conv4: output shape is (B, 256, 2, 2) => flatten to 1024
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        # Pass through the 4 conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # shape (B, 256, 2, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 1024)
        
        # Compute mu and logvar
        mu = self.fc_mu(x)         # (B, latent_dim)
        logvar = self.fc_logvar(x) # (B, latent_dim)
        return mu, logvar


def reparameterize(mu, logvar):
    """
    Given mu and logvar, sample z = mu + sigma * epsilon, where
    epsilon ~ N(0, 1) and sigma = exp(0.5 * logvar).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Dense up to 1x1x1024
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True)
        )
        
        # "Unflatten" from (B, 1024) to (B, 1024, 1, 1)
        # Then do transposed convolutions
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )  # => 5x5
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )  # => 13x13

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )  # => 30x30

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0),
            nn.Sigmoid()
        )  # => 64x64

    def forward(self, z):
        # Dense + reshape to (B, 1024, 1, 1)
        x = self.fc(z)                # (B, 1024)
        x = x.view(x.size(0), 1024, 1, 1)
        
        # Transposed convolutions to reconstruct to 64x64
        x = self.deconv1(x)  # => (B, 128, 5, 5)
        x = self.deconv2(x)  # => (B, 64, 13, 13)
        x = self.deconv3(x)  # => (B, 32, 30, 30)
        x = self.deconv4(x)  # => (B, 3, 64, 64)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    """
    1) Reconstruction loss: binary cross-entropy if input is in [0,1].
    2) KL Divergence: D_KL( q(z|x) || p(z) ).
    """
    bce = nn.functional.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )  # sum over all pixels & batch
    
    # KL Divergence = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    
    return (bce + kl) / x.size(0)  # average over batch




# 1. Set up device for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 2. Dataset & DataLoader
transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()  # yields [0,1] for pixel intensities
])

# dataset = ImageFolder(root="data", transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 3. Initialize Model, Optimizer
# model = VAE(latent_dim=1024).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # 4. Training Loop
# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
    
#     for images, _ in dataloader:
#         images = images.to(device)  # move data to MPS (Apple GPU)
        
#         # Forward pass
#         x_recon, mu, logvar = model(images)
#         loss = vae_loss(x_recon, images, mu, logvar)
        
#         # Backprop
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item() * images.size(0)
    
#     avg_loss = total_loss / len(dataset)
#     print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")