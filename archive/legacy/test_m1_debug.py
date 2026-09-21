"""Debug M1 spike_ops=0."""
import torch, sys
sys.path.insert(0, 'src')
from snn_baseline import EEGSNNWithEncoding, EEGSNN

# Test M1
m1 = EEGSNNWithEncoding().cuda()
m0 = EEGSNN().cuda()

X = torch.randn(32, 255).cuda()

# Check M1 internal values
m1.eval()
with torch.no_grad():
    # Threshold encoder output
    x_enc = m1.threshold_encoder(X)
    print('x_enc: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}'.format(
        x_enc.mean().item(), x_enc.std().item(), x_enc.min().item(), x_enc.max().item()))

    h = m1.spike_proj(x_enc)
    print('spike_proj out: mean={:.3f} std={:.3f}'.format(h.mean().item(), h.std().item()))

    # Step through
    T = m1.time_steps
    B = 32
    h_exp = h.unsqueeze(1).expand(B, T, -1)
    m1_state, m2_state = None, None
    spk_counts = []
    for t in range(T):
        sp1, m1_state = m1.lif1(h_exp[:, t, :], m1_state)
        sp2, m2_state = m1.lif2(sp1, m2_state)
        spk_counts.append((sp1.sum().item(), sp2.sum().item()))
    print('M1 spike counts (lif1, lif2) per step:', spk_counts)

# Test M0
m0.eval()
with torch.no_grad():
    h0 = m0.encoder(X)
    print('\nM0 encoder out: mean={:.3f} std={:.3f}'.format(h0.mean().item(), h0.std().item()))
    T = m0.time_steps
    B = 32
    h0_exp = h0.unsqueeze(1).expand(B, T, -1)
    m1_s, m2_s = None, None
    sc0 = []
    for t in range(T):
        sp1, m1_s = m0.lif1(h0_exp[:, t, :], m1_s)
        sp2, m2_s = m0.lif2(sp1, m2_s)
        sc0.append((sp1.sum().item(), sp2.sum().item()))
    print('M0 spike counts (lif1, lif2) per step:', sc0)

print('\nM1 lif1 threshold:', m1.lif1.threshold.data)
print('M1 lif1.beta:', m1.lif1.beta)
print('M1 lif1.log_tau:', m1.lif1.log_tau.data)
print('M1 lif2 threshold:', m1.lif2.threshold.data)
print('M1 lif2.beta:', m1.lif2.beta)
