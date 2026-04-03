import torch, torch.nn as nn, torch.nn.functional as F, time
from mamba2_multi_train import Mamba2LM, DEVICE, SEQ_LEN, BATCH_SIZE

model = Mamba2LM().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')

x = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN+1), device=DEVICE)
inp, tgt = x[:, :-1], x[:, 1:]

# Warmup
for _ in range(5):
    opt.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(inp)
        loss = F.cross_entropy(out.view(-1,256), tgt.reshape(-1))
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
torch.cuda.synchronize()

N = 20

# Full step
torch.cuda.synchronize(); t0 = time.time()
for _ in range(N):
    opt.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(inp)
        loss = F.cross_entropy(out.view(-1,256), tgt.reshape(-1))
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
torch.cuda.synchronize()
t_full = (time.time() - t0) / N * 1000

# Forward only
torch.cuda.synchronize(); t0 = time.time()
for _ in range(N):
    with torch.amp.autocast('cuda'):
        out = model(inp)
torch.cuda.synchronize()
t_fwd = (time.time() - t0) / N * 1000

t_bwd = t_full - t_fwd

print(f"Forward:  {t_fwd:.1f} ms")
print(f"Backward: {t_bwd:.1f} ms")
print(f"TOTAL:    {t_full:.1f} ms")
print(f"Steps/s:  {1000/t_full:.1f}")
print(f"Эпоха (~2400 steps): {t_full * 2400 / 1000:.0f}s")