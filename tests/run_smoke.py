"""Run the complete synthetic protocol and resume checks in a new output directory."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results')/('snn_synthetic_smoke_'+time.strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.out.resolve()
    cmd=[sys.executable,'-u','-m','src.snn_baseline','--synthetic','--models','M0','M1','M1_plain','M2','M3',
         '--test-subjects','1','2','--epochs','2','--steps-per-epoch','2','--hidden','12','--latent','8',
         '--per-subject','4','--meta-epochs','2','--meta-episodes','2','--meta-inner-steps','2',
         '--adapt-steps','2','--adapt-lrs','0.1','--repeats','2','--device','cpu','--threads','1','--out',str(out)]
    subprocess.run(cmd,cwd=root,check=True)
    folds=json.loads((out/'rows.json').read_text())
    assert len(folds)==2
    assert all(len(f['rows'])==51 for f in folds)
    assert json.loads((out/'status.json').read_text())['complete_loso'] is False
    for fold in folds:
        assert not set(fold['train_subjects'])&set(fold['val_subjects'])
        assert fold['test_subject'] not in fold['train_subjects']+fold['val_subjects']
        qrows=[r for r in fold['rows'] if r['method']=='lowdim']
        assert all(r['target_trainable_parameters']==8 for r in qrows)
        assert all(r['q_norm']>0 for r in qrows)
        assert all(0<=r['event_fraction']<=1 for r in fold['rows'])
    files=list(out.glob('seed*/*/*.pt'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    resumed=subprocess.run(cmd+['--resume'],cwd=root,text=True,capture_output=True)
    assert resumed.returncode==0,resumed.stderr
    assert resumed.stdout.count('skip complete')==2
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert not (out/'RUNNING.lock').exists()
    preserved=(out/'status.json').read_bytes()
    mismatch=subprocess.run(cmd+['--resume','--lr','0.123'],cwd=root,text=True,capture_output=True)
    assert mismatch.returncode!=0
    assert 'Resume refused' in mismatch.stdout
    assert preserved==(out/'status.json').read_bytes()
    print(f'SMOKE PASS: 2 synthetic folds, 102 result rows, fitted q, cost accounting and resume; {out}',flush=True)
    print('Engineering verification only; these are not EEG research results.',flush=True)

if __name__=='__main__': main()
