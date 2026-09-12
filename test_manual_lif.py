"""Test: manual LIF with straight-through estimator (STE) surrogate gradient."""
import torch, torch.nn as nn, numpy as np, sys, copy
sys.path.insert(0, 'src')
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features

DEVICE = torch.device('cuda')

class ManualLIF(nn.Module):
    """Manual LIF with STE surrogate gradient - no snntorch."""
    def __init__(self, dim, threshold=1.0, tau=5.0, learn_thr=True, learn_tau=True):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=learn_thr)
        self.log_tau = nn.Parameter(torch.tensor(tau).log(), requires_grad=learn_tau)

    @property
    def beta(self):
        return torch.exp(-1.0 / torch.clamp(torch.exp(self.log_tau), min=0.1))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        # Membrane update
        mem = self.beta * mem + (1 - self.beta) * x
        # STE surrogate: spike gradient = 1 in forward, normal BP backward
        spike = (mem - self.threshold).clamp(min=0) / (self.threshold + 1e-6)
        spike = spike.clamp(0, 1)
        # Hard reset: subtract threshold when spike
        mem = mem - spike.detach() * self.threshold
        return spike, mem


class TestSNN(nn.Module):
    def __init__(self, in_dim=255, hidden=64, z_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim), nn.ReLU(),
        )
        self.lif1 = ManualLIF(z_dim, threshold=1.0, tau=5.0)
        self.lif2 = ManualLIF(z_dim, threshold=1.0, tau=5.0)
        self.readout = nn.Linear(z_dim, 3)

    def forward(self, x, mem1=None, mem2=None):
        T = 8; B = x.shape[0]
        h = self.proj(x).unsqueeze(1).expand(B, T, -1)
        spk_out = []; m1, m2 = mem1, mem2
        for t in range(T):
            s1, m1 = self.lif1(h[:, t, :], m1)
            s2, m2 = self.lif2(s1, m2)
            spk_out.append(s2)
        return self.readout(torch.stack(spk_out, dim=1).mean(1))


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

model = TestSNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

best_va, best_state = -1, None
for ep in range(80):
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        opt.zero_grad()
        logits = model(X_s[idx])
        loss = ce(logits, y_s[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

    model.eval()
    with torch.no_grad():
        va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        tr_acc = (model(X_s[:512]).argmax(1) == y_s[:512]).float().mean().item()
    if va_acc > best_va:
        best_va = va_acc
        best_state = copy.deepcopy(model.state_dict())
    if ep % 20 == 0:
        print(f"ep {ep}: tr={tr_acc:.3f} va={va_acc:.3f} best={best_va:.3f}")

print(f"Best va: {best_va:.3f}")
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_acc = (model(X_t).argmax(1) == y_t).float().mean().item()
print(f"Test acc: {te_acc:.3f}")
