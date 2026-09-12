"""Test SNN vs CNN on training subjects (fit quality check)."""
import torch, numpy as np, sys
sys.path.insert(0, 'src')
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features, EEGSNN, EEGCNN, train_model, evaluate_model

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
X_all, y_all, subj_all = [], [], []
for subj in [1, 2, 3]:
    for ses in [1]:
        data = build_segments('seed', subj, ses, preproc_cfg, use_cache=False)
        feats = de_features(data['segs'], 200)
        X_all.append(feats); y_all.append(data['y'])
        subj_all.append(np.full(len(data['y']), subj))
X_all = np.concatenate(X_all, 0)
y_all = np.concatenate(y_all, 0)
subj_all = np.concatenate(subj_all, 0)

# Test on subj=1, train on 2&3: check if model can at least fit 2&3 well
X_tr, y_tr, X_te, y_te = [], [], [], []
for test_subj in [1, 2, 3]:
    mask = subj_all == test_subj
    if mask.sum() > 0:
        X_te.append(X_all[mask]); y_te.append(y_all[mask])
    else:
        X_tr.append(X_all); y_tr.append(y_all)

X_te = np.concatenate(X_te, 0)
y_te = np.concatenate(y_te, 0)
# For a single-subject test, just use all data split by subject
# Actually: use subj 1 as test, 2&3 as train
X_tr = X_all[subj_all != 1]
y_tr = y_all[subj_all != 1]
X_te = X_all[subj_all == 1]
y_te = y_all[subj_all == 1]

mu, std = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-8
X_tr_n = (X_tr - mu) / std
X_te_n = (X_te - mu) / std

# Split train into train/val
n = len(X_tr_n)
perm = np.random.RandomState(42).permutation(n)
val_k = int(n * 0.2)
val_idx, tr_idx = perm[:val_k], perm[val_k:]
X_va, y_va = X_tr_n[val_idx], y_tr[val_idx]
X_s, y_s = X_tr_n[tr_idx], y_tr[tr_idx]
print(f"Train: {X_s.shape}, Val: {X_va.shape}, Test: {X_te_n.shape}")

for name, Model in [("CNN", EEGCNN), ("SNN", EEGSNN)]:
    model = Model().to(DEVICE)
    best_va, best_st = train_model(model, X_s, y_s, X_va, y_va, epochs=80, lr=1e-3, batch_size=32, seed=42, verbose=False)
    model.load_state_dict(best_st)
    # Evaluate on train subset
    X_s_t = torch.FloatTensor(X_s[:512]).to(DEVICE)
    y_s_t = torch.LongTensor(y_s[:512]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        tr_acc = (model(X_s_t).argmax(1) == y_s_t).float().mean().item()
        # Evaluate on full training and test
        X_full_t = torch.FloatTensor(X_tr_n).to(DEVICE)
        y_full_t = torch.LongTensor(y_tr).to(DEVICE)
        X_te_t = torch.FloatTensor(X_te_n).to(DEVICE)
        y_te_t = torch.LongTensor(y_te).to(DEVICE)
        full_tr_acc = (model(X_full_t).argmax(1) == y_full_t).float().mean().item()
        te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
    print(f"{name}: va={best_va:.3f} tr_subset={tr_acc:.3f} tr_full={full_tr_acc:.3f} te={te_acc:.3f}")
