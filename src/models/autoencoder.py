# src/models/autoencoder.py
import torch
import torch.nn as nn
import numpy as np

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

def get_recon_errors(ae_model, X_np, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae_model.to(device)
    ae_model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, _ = ae_model(X_t)
    errs = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
    return errs
