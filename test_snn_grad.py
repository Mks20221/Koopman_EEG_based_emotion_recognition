"""Debug SNN gradient flow."""
import torch, numpy as np, sys
sys.path.insert(0, 'src')
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features, EEGSNN, EEGCNN

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
X_all, y_all, subj_all = [], [], []
for subj in [1, 2, 3]:
    data = build_segments('seed', subj, 1, preproc_cfg, use_cache=False)
    feats = de_features(data['segs'], 200)
    X_all.append(feats); y_all.append(data['y'])
    subj_all.append(np.full(len(data['y']), subj))
X_all = np.concatenate(X_all, 0)
y_all = np.concatenate(y_all, 0)
subj_all = np.concatenate(subj_all, 0)

# Use subj 1 as test, 2&3 as train
X_tr = X_all[subj_all != 1]
y_tr = y_all[subj_all != 1]
mu, std = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-8
X_tr_n = (X_tr - mu) / std

# First 64 samples
X64 = torch.FloatTensor(X_tr_n[:64]).to(DEVICE)
y64 = torch.LongTensor(y_tr[:64]).to(DEVICE)

# Test CNN gradient
cnn = EEGCNN().to(DEVICE)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
logits = cnn(X64)
loss = torch.nn.CrossEntropyLoss()(logits, y64)
opt.zero_grad(); loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in cnn.named_parameters() if p.grad is not None}
print("CNN grad norms:", {k: f"{v:.6f}" for k, v in grad_norms.items()})
print(f"CNN loss={loss.item():.4f}, logits mean={logits.mean().item():.4f}")

# Test SNN gradient
snn = EEGSNN().to(DEVICE)
opt2 = torch.optim.Adam(snn.parameters(), lr=1e-3)
logits2 = snn(X64)
loss2 = torch.nn.CrossEntropyLoss()(logits2, y64)
opt2.zero_grad(); loss2.backward()
grad_norms2 = {n: p.grad.norm().item() for n, p in snn.named_parameters() if p.grad is not None}
print("\nSNN grad norms:", {k: f"{v:.6f}" for k, v in grad_norms2.items()})
print(f"SNN loss={loss2.item():.4f}, logits mean={logits2.mean().item():.4f}")

# Inspect SNN internal
print("\nSNN encoder gain:", snn.encoder.gain.data)
h = snn.encoder(X64)
print("SNN encoder out: mean={:.4f} std={:.4f}".format(h.mean().item(), h.std().item()))

# Step by step through LIF
lif1 = snn.lif1
lif2 = snn.lif2
print("lif1 threshold:", lif1.threshold.data, "tau:", lif1.tau.data)
print("lif2 threshold:", lif2.threshold.data, "tau:", lif2.tau.data)

# Check readout gradients
print("\nSNN readout classifier weight grad norm:", snn.readout.classifier.weight.grad.norm().item() if snn.readout.classifier.weight.grad is not None else None)
print("SNN readout classifier bias grad:", snn.readout.classifier.bias.grad.item() if snn.readout.classifier.bias.grad is not None else None)
