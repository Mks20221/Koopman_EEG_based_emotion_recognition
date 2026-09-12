"""Debug SNN: inspect forward values step by step."""
import torch, torch.nn as nn, numpy as np, sys
sys.path.insert(0, 'src')
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features, CNNEncoder, LIFNeurons, MeanReadout

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
data = build_segments('seed', 1, 1, preproc_cfg, use_cache=False)
feats = de_features(data['segs'], 200)
mu, std = feats.mean(0, keepdims=True), feats.std(0, keepdims=True) + 1e-8
X = (feats - mu) / std

# First 64 samples
X_t = torch.FloatTensor(X[:64]).to(DEVICE)
y_t = torch.LongTensor(data['y'][:64]).to(DEVICE)

enc = CNNEncoder().to(DEVICE)
lif1 = LIFNeurons(32, threshold=1.0, tau=5.0).to(DEVICE)
lif2 = LIFNeurons(32, threshold=1.0, tau=5.0).to(DEVICE)
readout = MeanReadout(32, 3).to(DEVICE)

# Forward
h = enc(X_t)  # (64, 32)
print(f"Encoder out: mean={h.mean().item():.3f} std={h.std().item():.3f}")

T = 8
B = 64
h = h.unsqueeze(1).expand(B, T, -1)
m1, m2 = None, None
spike_sums = []
mem1_vals, mem2_vals = [], []
for t in range(T):
    x_t = h[:, t, :]
    sp1, m1 = lif1(x_t, m1)
    sp2, m2 = lif2(sp1, m2)
    spike_sums.append((sp1.sum().item(), sp2.sum().item()))
    mem1_vals.append(m1.detach().cpu())
    mem2_vals.append(m2.detach().cpu())

print("Spike sums (lif1, lif2) per step:", spike_sums)
spikes = torch.stack([s[1] for s in [spike_sums[t] for t in range(T)]], dim=1)
spike_tensor = torch.stack([sp2 for _, sp2 in [spike_sums[t] for t in range(T)]], dim=1)
print("spike_tensor shape:", spike_tensor.shape if hasattr(spike_tensor, 'shape') else 'N/A')

# Build full spike tensor
h_expanded = h
m1, m2 = None, None
spikes_out = []
for t in range(T):
    sp1, m1 = lif1(h_expanded[:, t, :], m1)
    sp2, m2 = lif2(sp1, m2)
    spikes_out.append(sp2)
spikes = torch.stack(spikes_out, dim=1)  # (B, T, z_dim)
print(f"Spikes: mean={spikes.mean().item():.4f}, sum per step={[spikes[:,t].sum().item() for t in range(T)]}")

logits = readout(spikes)
print(f"Logits: mean={logits.mean().item():.3f} std={logits.std().item():.3f}")
print(f"Logits per class: {logits[0].detach().cpu().numpy()}")
acc = (logits.argmax(1) == y_t).float().mean().item()
print(f"Acc on first 64: {acc:.3f}")
