import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from pipeline.config import SEED, DEVICE, AE_LATENT_DIM, AE_EPOCHS, AE_BATCH_SIZE, LGBM_N_ESTIMATORS, LGBM_NUM_LEAVES

# ------------------------
# Autoencoder
# ------------------------
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=AE_LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

# ------------------------
# Train LightGBM
# ------------------------
def train_lightgbm(X_train, y_train, X_val, y_val):
    clf = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        num_leaves=LGBM_NUM_LEAVES,
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    return clf

# ------------------------
# Train AE
# ------------------------
def train_autoencoder(X_train, latent_dim=AE_LATENT_DIM, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, device=DEVICE):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae = AE(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[AE] Epoch {epoch+1}/{epochs} Loss={total_loss/len(loader.dataset):.6f}")
    return ae
