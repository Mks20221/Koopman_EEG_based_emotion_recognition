"""Test snntorch Leaky inside train_model pattern."""
import torch, torch.nn as nn, numpy as np, sys, copy
sys.path.insert(0, 'src')
from snntorch import Leaky, surrogate
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
X_all, y_all = [], []
for subj in [1, 2]:
    data = build_segments('seed', subj, 1, preproc_cfg, use_cache=False)
    feats = de_features(data['segs'], 200)
    X_all.append(feats); y_all.append(data['y'])
X = np.concatenate(X_all, 0)
y = np.concatenate(y_all, 0)
mu, std = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
X_n = (X - mu) / std

# Same model pattern as EEGSNN
class TestSNN(nn.Module):
    def __init__(self, in_dim=255, hidden=64, z_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim), nn.ReLU(),
        )
        self.lif1 = Leaky(beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.lif2 = Leaky(beta=0.8, threshold=1.0, learn_beta=True, learn_threshold=True,
                          spike_grad=surrogate.atan(), init_hidden=False)
        self.readout = nn.Linear(z_dim, 3)

    def forward(self, x, mem1=None, mem2=None):
        T = 8
        B = x.shape[0]
        h = self.proj(x)  # (B, z_dim)
        h = h.unsqueeze(1).expand(B, T, -1)
        spk_out = []
        m1, m2 = mem1, mem2
        for t in range(T):
            s1, m1 = self.lif1(h[:, t, :], m1)
            s2, m2 = self.lif2(s1, m2)
            spk_out.append(s2)
        spikes = torch.stack(spk_out, dim=1)
        return self.readout(spikes.mean(1))

model = TestSNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

X_t = torch.FloatTensor(X_n).to(DEVICE)
y_t = torch.LongTensor(y).to(DEVICE)

n = len(X_t)
perm_base = np.random.RandomState(42).permutation(n)
val_k = int(n * 0.2)
val_idx = perm_base[:val_k]
tr_idx = perm_base[val_k:]
X_va, y_va = X_t[val_idx], y_t[val_idx]
X_s, y_s = X_t[tr_idx], y_t[tr_idx]

print(f"Train: {len(X_s)}, Val: {len(X_va)}")
best_va, best_state = -1, None

for ep in range(10):  # only 10 epochs for test
    model.train()
    perm = torch.randperm(len(X_s), device=DEVICE)
    epoch_loss = 0.0
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        xb, yb = X_s[idx], y_s[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()

    sched.step()
    model.eval()
    with torch.no_grad():
        va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        tr_acc = (model(X_s[:512]).argmax(1) == y_s[:512]).float().mean().item()

    if va_acc > best_va:
        best_va = va_acc
        best_state = copy.deepcopy(model.state_dict())

    if ep % 3 == 0:
        print(f"ep {ep}: loss={epoch_loss/len(range(0,len(perm),32)):.4f} tr={tr_acc:.3f} va={va_acc:.3f} best={best_va:.3f}")

print(f"Done. Best va: {best_va:.3f}")
