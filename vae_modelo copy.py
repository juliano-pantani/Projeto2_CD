import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 10 * 10, latent_dim)
        self.fc_logvar = nn.Linear(64 * 10 * 10, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 10 * 10)
        self.net = nn.Sequential(
            nn.Unflatten(1, (64, 10, 10)),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Upsample(size=(50, 50), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        return self.net(self.fc(z))

class VAEUpsampler(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrizar(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * torch.randn_like(sigma)

    def forward(self, x_pequena):
        mu, logvar = self.encoder(x_pequena)
        z = self.reparametrizar(mu, logvar)
        delta = self.decoder(z)
        base = torch.nn.functional.interpolate(x_pequena, size=delta.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(base + delta, 0.0, 1.0), mu, logvar