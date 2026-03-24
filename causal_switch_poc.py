import torch
import torch.nn as nn
import math
from torch.utils.data import TensorDataset, DataLoader

# ── 1. Causal Switch (Dirac Delta Approximation) ───────────────
class CausalSwitch:
    def __init__(self, eps_0=1.0, eps_min=1e-5, total_steps=10000):
        self.eps_0       = eps_0
        self.eps_min     = eps_min
        self.gamma       = math.exp(math.log(eps_min / eps_0) / total_steps)
        self.active      = False
        self.step        = 0

    @property
    def epsilon(self):
        return max(self.eps_min, self.eps_0 * (self.gamma ** self.step))

    def anneal(self): 
        self.step += 1

    def inject(self, x_cf_emb: torch.Tensor) -> torch.Tensor:
        """Titik A: Timpa embedding X dengan aproksimasi Dirac Delta N(x_cf, εI)"""
        eps   = torch.tensor(self.epsilon, dtype=x_cf_emb.dtype, device=x_cf_emb.device)
        noise = torch.randn_like(x_cf_emb) * torch.sqrt(eps)
        # stop_gradient sangat krusial agar Z tidak terpengaruh saat backprop
        return (x_cf_emb + noise).detach()   


# ── 2. Dynamic Attention Mask (Graph Surgery) ───────────────────
def make_mask(intervene: bool, device) -> torch.Tensor:
    """
    Sequence: [Z(0), X(1), Y(2)]
    Titik B: Saat intervensi aktif, M[1,0] = -inf sehingga X tidak melihat Z.
    """
    NEG = float('-inf')
    M = torch.tensor([
        [0.,  NEG, NEG],
        [0.,  0.,  NEG],
        [0.,  0.,  0. ]
    ], device=device)
    
    if intervene:
        M[1, 0] = NEG   # GRAPH SURGERY: Memutus panah Z -> X
        
    return M


# ── 3. Arsitektur Causal Transformer ────────────────────────────
class CausalTransformer(nn.Module):
    def __init__(self, d_model=16, n_heads=4, total_steps=10000):
        super().__init__()
        self.d_model = d_model
        self.switch  = CausalSwitch(total_steps=total_steps)

        # Proyeksi skalar ke dimensi laten
        self.emb_z   = nn.Linear(1, d_model)
        self.emb_x   = nn.Linear(1, d_model)
        
        # y_query berfungsi sebagai kanvas kosong agar model menoleh ke Z dan X
        self.y_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        
        # Dual Prediction Heads (memastikan dimensi loss seimbang)
        self.head_x  = nn.Linear(d_model, 1)  # Prediksi X dari observasi Z
        self.head_y  = nn.Linear(d_model, 1)  # Prediksi Y dari observasi Z dan X

    def _forward_seq(self, z_emb, x_emb, intervene: bool):
        """Memproses sekuens token melewati blok Transformer"""
        B     = z_emb.shape[0]
        y_emb = self.y_query.expand(B, 1, self.d_model).squeeze(1)
        seq   = torch.stack([z_emb, x_emb, y_emb], dim=1)   # Shape: (B, 3, d_model)

        mask       = make_mask(intervene, seq.device)
        attn_out, _= self.attn(seq, seq, seq, attn_mask=mask)
        seq        = self.norm1(seq + attn_out)
        seq        = self.norm2(seq + self.ff(seq))
        return seq

    def forward(self, z, x, x_cf=None):
        z_emb = self.emb_z(z)
        x_emb = self.emb_x(x_cf if x_cf is not None else x)

        # Injeksi intervensi jika do(X) dipanggil
        if x_cf is not None:
            x_emb = self.switch.inject(x_emb)
            self.switch.anneal()

        seq    = self._forward_seq(z_emb, x_emb, intervene=(x_cf is not None))
        x_pred = self.head_x(seq[:, 0, :])
        y_pred = self.head_y(seq[:, 2, :])
        return x_pred, y_pred

    # ── Level 3 AGI: Abduksi → Intervensi → Prediksi ────────────
    def counterfactual_inference(self, x_obs, y_obs, x_cf, steps=500, lr=0.01):
        """Menghitung realita alternatif (Counterfactuals)"""
        B, device = x_obs.shape[0], x_obs.device

        # Langkah 1: ABDUKSI (Menebak confounder masa lalu z_hat)
        z_hat = torch.randn(B, self.d_model, device=device)
        z_hat.requires_grad_(True) # Leaf tensor untuk optimizer
        opt_z = torch.optim.Adam([z_hat], lr=lr)

        # Bekukan parameter model agar hanya z_hat yang di-update
        for p in self.parameters():
            p.requires_grad_(False)

        mse = nn.MSELoss()
        for _ in range(steps):
            opt_z.zero_grad()
            x_emb = self.emb_x(x_obs)
            seq   = self._forward_seq(z_hat, x_emb, intervene=False)
            
            x_rec = self.head_x(seq[:, 0, :])
            y_rec = self.head_y(seq[:, 2, :])
            
            # Loss untuk mencocokkan tebakan masa lalu dengan fakta observasi
            loss  = mse(x_rec, x_obs) + mse(y_rec, y_obs)
            loss.backward()
            opt_z.step()

        z_past = z_hat.detach()

        # Buka kembali kunci parameter
        for p in self.parameters():
            p.requires_grad_(True)

        # Langkah 2 & 3: INTERVENSI + PREDIKSI
        x_cf_emb = self.switch.inject(self.emb_x(x_cf))
        seq      = self._forward_seq(z_past, x_cf_emb, intervene=True)
        return self.head_y(seq[:, 2, :])


# ── 4. Data Generating Process (DGP) Sintetik ───────────────────
def generate_data(n=5000, alpha=2.0, beta=1.5, gamma=1.0):
    Z = torch.randn(n, 1)
    X = alpha * Z + torch.randn(n, 1) * 0.1
    Y = beta  * X + gamma * Z + torch.randn(n, 1) * 0.1
    return Z, X, Y

def generate_do_data(n=5000, beta=1.5):
    X = torch.randn(n, 1)
    Y = beta * X + torch.randn(n, 1) * 0.1
    return X, Y

def generate_cf_data(n=500, alpha=2.0, beta=1.5, gamma=1.0):
    Z    = torch.randn(n, 1)
    X    = alpha * Z + torch.randn(n, 1) * 0.1
    Y    = beta  * X + gamma * Z + torch.randn(n, 1) * 0.1
    X_cf = torch.randn(n, 1)
    Y_cf = beta * X_cf + gamma * Z + torch.randn(n, 1) * 0.1
    return Z, X, Y, X_cf, Y_cf

def make_loaders(n=5000, batch=64):
    Z, X, Y = generate_data(n)
    Xd, Yd  = generate_do_data(n)
    return (DataLoader(TensorDataset(Z, X, Y),  batch_size=batch, shuffle=True),
            DataLoader(TensorDataset(Xd, Yd),   batch_size=batch, shuffle=True))


# ── 5. Training Loop ────────────────────────────────────────────
def train(model, optimizer, obs_loader, do_loader, alpha=10.0):
    mse = nn.MSELoss()
    for (z, x, y), (x_do, y_do) in zip(obs_loader, do_loader):
        optimizer.zero_grad()

        # Observational Loss
        x_pred_obs, y_pred_obs = model(z, x)
        loss_obs = mse(x_pred_obs, x) + mse(y_pred_obs, y)

        # Interventional Loss (Causal Switch Aktif)
        z_null = torch.zeros(x_do.shape[0], 1)
        _, y_pred_do = model(z_null, x_do, x_cf=x_do)
        loss_do = mse(y_pred_do, y_do)

        # Joint Objective Optimization (alpha = Causal Penalty)
        (loss_obs + alpha * loss_do).backward()
        optimizer.step()


# ── Entrypoint ──────────────────────────────────────────────────
if __name__ == "__main__":
    STEPS = 5000 * 100 // 64
    obs_loader, do_loader = make_loaders()

    results = {}
    for d in [12, 14]:
        model     = CausalTransformer(d_model=d, n_heads=2, total_steps=STEPS)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(100):
            train(model, optimizer, obs_loader, do_loader, alpha=10.0)
        _, X, Y, X_cf, Y_cf_true = generate_cf_data(n=500)
        Y_cf_pred = model.counterfactual_inference(X, Y, X_cf, steps=500, lr=0.01)
        mse_val   = nn.MSELoss()(Y_cf_pred, Y_cf_true).item()
        results[d] = mse_val
        print(f"d_model={d:>3} | CF MSE = {mse_val:.4f}")

    best = min(results, key=results.get)
    print(f"\nSweet spot: d_model={best} | MSE={results[best]:.4f}")