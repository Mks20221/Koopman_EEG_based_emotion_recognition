"""Test: does validation evaluation cause the backward error?"""
import torch, torch.nn as nn, numpy as np, sys, copy
sys.path.insert(0, 'src')
from snntorch import Leaky, surrogate
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
data = build_segments('seed', 1, 1, preproc_cfg, use_cache=False)
feats = de_features(data['segs'], 200)
mu, std = feats.mean(0, keepdims=True), feats.std(0, keepdims=True) + 1e-8
X = (feats - mu) / std
X_t = torch.FloatTensor(X[:256]).to(DEVICE)
y_t = torch.LongTensor(data['y'][:256]).to(DEVICE)

n = len(X_t)
perm_base = np.random.RandomState(42).permutation(n)
val_k = int(n * 0.2)
val_idx = perm_base[:val_k]
tr_idx = perm_base[val_k:]
X_va, y_va = X_t[val_idx], y_t[val_idx]
X_s, y_s = X_t[tr_idx], y_t[tr_idx]

class TestSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(255, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.lif1 = Leaky(beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.lif2 = Leaky(beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.readout = nn.Linear(32, 3)

    def forward(self, x):
        T = 8; B = x.shape[0]
        h = self.proj(x).unsqueeze(1).expand(B, T, -1)
        spk_out = []
        m1, m2 = None, None
        for t in range(T):
            s1, m1 = self.lif1(h[:, t, :], m1)
            s2, m2 = self.lif2(s1, m2)
            spk_out.append(s2)
        return self.readout(torch.stack(spk_out, dim=1).mean(1))

model = TestSNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

# Test: run 20 epochs WITHOUT any eval in the loop
for ep in range(20):
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(X_s[idx])
        loss = ce(logits, y_s[idx])
        loss.backward()
        opt.step()

    # NO model.eval(), NO evaluation at all
    print(f"ep {ep}: done (no eval)")

print("20 epochs WITHOUT eval: OK")

# Now test WITH eval (like train_model)
best_va = -1
for ep in range(20, 25):
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(X_s[idx])
        loss = ce(logits, y_s[idx])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    model.train()
    if va_acc > best_va:
        best_va = va_acc
    print(f"ep {ep}: va={va_acc:.3f}")

print("5 epochs WITH eval: OK")
print("DONE")
