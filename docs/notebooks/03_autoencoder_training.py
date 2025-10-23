# %% [markdown]
# 03 - Autoencoder Training (split from original)
# Contains: AE definition, training loop, save model, compute recon errors
# %%
import torch
import torch.nn as nn
import joblib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

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
    def forward(self,x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

input_dim = X_train_s.shape[1]
ae = AE(input_dim, AE_LATENT).to(device)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
train_ds = torch.utils.data.TensorDataset(X_train_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=AE_BATCH, shuffle=True, drop_last=False)

ae.train()
for epoch in range(AE_EPOCHS):
    running = 0.0
    for (batch,) in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        recon, _ = ae(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        opt.step()
        running += loss.item() * batch.size(0)
    running /= len(train_loader.dataset)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"AE Epoch {epoch+1}/{AE_EPOCHS} — loss: {running:.6f}")

torch.save(ae.state_dict(), "ae_model.pt")
print("Saved AE state: ae_model.pt")

# %%
ae.eval()
@torch.no_grad()
def get_recon_errors(ae_model, X_np):
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    recon, _ = ae_model(X_t)
    recon_np = recon.cpu().numpy()
    errs = np.mean((recon_np - X_np)**2, axis=1)
    return errs

errs_train = get_recon_errors(ae, X_train_s)
errs_val_seen = get_recon_errors(ae, X_val_seen_s)
errs_test_seen = get_recon_errors(ae, X_test_seen_s)
errs_unseen = get_recon_errors(ae, X_unseen_s)

print("AE recon error stats — train mean/std:", errs_train.mean(), errs_train.std())
print("val_seen mean:", errs_val_seen.mean(), "unseen mean:", errs_unseen.mean())
