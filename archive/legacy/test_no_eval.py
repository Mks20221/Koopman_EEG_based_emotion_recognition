"""Isolate: does eval() cause the backward issue?"""
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

    def forward(self, x, mem1=None, mem2=None):
        T = 8; B = x.shape[0]
        h = self.proj(x).unsqueeze(1).expand(B, T, -1)
        spk_out = []; m1, m2 = mem1, mem2
        for t in range(T):
            s1, m1 = self.lif1(h[:, t, :], m1)
            s2, m2 = self.lif2(s1, m2)
            spk_out.append(s2)
        return self.readout(torch.stack(spk_out, dim=1).mean(1))

model = TestSNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

for ep in range(5):
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(X_s[idx])
        loss = ce(logits, y_s[idx])
        loss.backward()
        opt.step()

    # Test A: evaluate WITH model.eval() and no_grad
    model.eval()
    with torch.no_grad():
        va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    print(f"ep {ep}: va={va_acc:.3f}")

print("With eval(): OK")

# Now test B: WITHOUT model.eval()
for ep in range(5, 8):
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(X_s[idx])
        loss = ce(logits, y_s[idx])
        loss.backward()
        opt.step()
    # No model.eval(), just use train mode
    with torch.no_grad():
        va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    print(f"ep {ep}: va={va_acc:.3f} (no eval)")

print("Without eval(): OK")
print("DONE")
