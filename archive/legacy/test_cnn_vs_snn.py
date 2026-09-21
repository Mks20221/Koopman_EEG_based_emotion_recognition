"""Compare CNN vs SNN LOSO on 3 subjects."""
import torch, torch.nn as nn, numpy as np, sys, copy
sys.path.insert(0, 'src')
from preprocess import build_segments, PreprocConfig
from snn_baseline import de_features, loso_split, EEGSNN, EEGCNN, train_model, evaluate_model
from config import SEED_FS

DEVICE = torch.device('cuda')
preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True, reject_abs_thr=15.0)
X_all, y_all, subj_all = [], [], []
for subj in [1, 2, 3]:
    for ses in [1]:
        data = build_segments('seed', subj, ses, preproc_cfg, use_cache=False)
        feats = de_features(data['segs'], SEED_FS)
        X_all.append(feats); y_all.append(data['y'])
        subj_all.append(np.full(len(data['y']), subj))
X_all = np.concatenate(X_all, 0)
y_all = np.concatenate(y_all, 0)
subj_all = np.concatenate(subj_all, 0)

results = []
for test_subj in [1, 2, 3]:
    X_tr, y_tr, X_te, y_te = loso_split(X_all, y_all, subj_all, test_subj)
    mu, std = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-8
    X_tr_n, X_te_n = (X_tr-mu)/std, (X_te-mu)/std

    n_tr = len(X_tr_n)
    perm = np.random.RandomState(42).permutation(n_tr)
    val_k = max(1, int(n_tr * 0.2))
    val_idx, tr_idx = perm[:val_k], perm[val_k:]
    X_va, y_va = X_tr_n[val_idx], y_tr[val_idx]
    X_s, y_s = X_tr_n[tr_idx], y_tr[tr_idx]

    # CNN
    cnn2 = EEGCNN().to(DEVICE)
    best_va_cnn2, best_st_cnn2 = train_model(cnn2, X_tr_n, y_tr, X_va, y_va, epochs=50, lr=1e-3, batch_size=32, seed=42, verbose=False)
    cnn2.load_state_dict(best_st_cnn2)
    _, preds_cnn = evaluate_model(cnn2, X_te_n, y_te)

    # SNN
    snn2 = EEGSNN().to(DEVICE)
    best_va_snn2, best_st_snn2 = train_model(snn2, X_tr_n, y_tr, X_va, y_va, epochs=50, lr=1e-3, batch_size=32, seed=42, verbose=False)
    snn2.load_state_dict(best_st_snn2)
    _, preds_snn = evaluate_model(snn2, X_te_n, y_te)

    cnn_te = float((preds_cnn == y_te).mean())
    snn_te = float((preds_snn == y_te).mean())
    results.append({'subj': test_subj, 'cnn_va': best_va_cnn2, 'cnn_te': cnn_te, 'snn_va': best_va_snn2, 'snn_te': snn_te})
    print(f"subj {test_subj}: CNN va={best_va_cnn2:.3f} te={cnn_te:.3f} | SNN va={best_va_snn2:.3f} te={snn_te:.3f}")

print(f"Mean CNN test: {np.mean([r['cnn_te'] for r in results]):.3f}")
print(f"Mean SNN test: {np.mean([r['snn_te'] for r in results]):.3f}")
