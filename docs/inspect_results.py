import json

for fname in [
    "results/koopman_gate/gate_rows_cf.json",
    "results/koopman_gate/gate_rows_fast.json",
    "results/koopman_gate/loso_rows_full.json",
]:
    try:
        d = json.load(open(fname))
        print(f"\n=== {fname} ===")
        print(f"  len={len(d)}")
        print(f"  keys: {sorted(d[0].keys())}")
        print(f"  variants: {sorted({r.get('variant', r.get('variant', '?')) for r in d})}")
        print(f"  framing: {sorted({r.get('framing', '?') for r in d})}")
        print(f"  subjects: {sorted({r.get('subject', r.get('test_subject', '?')) for r in d})}")
        # check if z_tr/z_te are saved
        has_z = any('z_' in k for k in d[0].keys())
        print(f"  has z_tr/z_te: {has_z}")
    except Exception as e:
        print(f"\n=== {fname} === ERROR: {e}")
