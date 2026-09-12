"""Minimal test: does snntorch Leaky work with init_hidden=False?"""
import torch, torch.nn as nn, numpy as np, sys
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
X_t = torch.FloatTensor(X[:64]).to(DEVICE)
y_t = torch.LongTensor(data['y'][:64]).to(DEVICE)

# Simple model with init_hidden=False (default)
class MinimalSNN2(nn.Module):
    def __init__(self, in_dim=255, hidden=64, out_dim=3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lif1 = Leaky(beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.lif2 = Leaky(beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        T = 8
        B = x.shape[0]
        h = torch.relu(self.proj(x))  # (B, hidden)
        h = h.unsqueeze(1).expand(B, T, -1)  # (B, T, hidden)
        # Initialize mem states
        mem1 = torch.zeros_like(h[:, 0, :])
        mem2 = torch.zeros_like(h[:, 0, :])
        spk_out = []
        for t in range(T):
            spk1, mem1 = self.lif1(h[:, t, :], mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spk_out.append(spk2)
        spikes = torch.stack(spk_out, dim=1)  # (B, T, hidden)
        return self.readout(spikes.mean(1))

model = MinimalSNN2().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

for ep in range(10):
    opt.zero_grad()
    logits = model(X_t)
    loss = ce(logits, y_t)
    loss.backward()
    opt.step()
    acc = (logits.argmax(1) == y_t).float().mean().item()
    print(f"ep {ep}: loss={loss.item():.4f} acc={acc:.3f}")
print("Done - 10 epochs successful")
