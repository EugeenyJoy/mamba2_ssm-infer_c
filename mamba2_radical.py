import torch
import torch.nn as nn
import torch.nn.functional as F
import time, glob, sys, math
from pathlib import Path

DEVICE = "cuda"
SEQ_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 8
LR = 3e-4

# ============================================================
# Parallel SSM Scan
# ============================================================
class ParallelSSMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, B_mat, C_mat, A):
        B_sz, T, DI = x.shape
        x_f=x.float(); dt_f=dt.float(); A_f=A.float()
        B_f=B_mat.float(); C_f=C_mat.float()
        log_a = (dt_f * A_f.view(1,1,DI)).clamp(-20,20)
        cum_log_a = torch.cumsum(log_a, dim=1)
        dtx = dt_f * x_f
        inp = dtx.unsqueeze(-1) * B_f.unsqueeze(2)
        scaled_inp = torch.exp(-cum_log_a).unsqueeze(-1) * inp
        prefix_sum = torch.cumsum(scaled_inp, dim=1)
        h_all = torch.exp(cum_log_a).unsqueeze(-1) * prefix_sum
        y = (h_all * C_f.unsqueeze(2)).sum(-1)
        ctx.save_for_backward(x_f, dt_f, B_f, C_f, A_f, h_all, cum_log_a, dtx)
        return y
    @staticmethod
    def backward(ctx, dy):
        x, dt, B, C, A, h_all, cum_log_a, dtx = ctx.saved_tensors
        dy = dy.float()
        dC = torch.einsum('btd,btds->bts', dy, h_all)
        dh = dy.unsqueeze(-1) * C.unsqueeze(2)
        d_prefix = dh * torch.exp(cum_log_a).unsqueeze(-1)
        d_si = torch.flip(torch.cumsum(torch.flip(d_prefix,[1]),dim=1),[1])
        d_inp = d_si * torch.exp(-cum_log_a).unsqueeze(-1)
        d_dtx = (d_inp * B.unsqueeze(2)).sum(-1)
        dB = torch.einsum('btds,btd->bts', d_inp, dtx)
        return d_dtx*dt, d_dtx*x, dB, dC, None

# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.w

# ============================================================
# Оригинальная архитектура с parallel scan
# ============================================================
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_inner = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.log_A = nn.Parameter(torch.log(torch.linspace(1, 16, d_model)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        x_ssm = self.conv(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_ssm = F.silu(x_ssm)
        
        dt = F.softplus(self.dt_proj(x_ssm))
        B_mat = self.B_proj(x_ssm)
        C_mat = self.C_proj(x_ssm)
        A = -torch.exp(self.log_A.float())
        
        y = ParallelSSMScan.apply(x_ssm, dt, B_mat, C_mat, A)
        
        y = y + x_ssm * self.D
        y = y * F.silu(z)
        y = self.out_proj(self.drop(y))
        return residual + y

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return x + self.net(self.norm(x))

class OrigMamba(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_layers=8, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(0.1)
        
        layers = []
        for _ in range(n_layers):
            layers.append(SelectiveSSM(d_model, d_state))
            layers.append(FFN(d_model))
        self.layers = nn.ModuleList(layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {n_params/1e6:.2f}M params, d={d_model}, "
              f"layers={n_layers}×2, d_state={d_state}")
    
    def forward(self, x):
        x = self.drop(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ============================================================
# Radical 1: Слитые SSM+FFN блоки
# ============================================================
class FusedSSMFFN(nn.Module):
    def __init__(self, d, ds=32, expand=2):
        super().__init__()
        self.norm = RMSNorm(d)
        
        # SSM ветка
        self.ssm_in = nn.Linear(d, d*2, bias=False)
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)
        self.dt_proj = nn.Linear(d, d)
        self.B_proj = nn.Linear(d, ds, bias=False)
        self.C_proj = nn.Linear(d, ds, bias=False)
        self.ssm_out = nn.Linear(d, d, bias=False)
        
        self.log_A = nn.Parameter(torch.log(torch.linspace(1, 16, d)))
        self.D = nn.Parameter(torch.ones(d))
        
        # FFN ветка (параллельно!)
        self.ffn_w1 = nn.Linear(d, d*expand, bias=False)
        self.ffn_w2 = nn.Linear(d*expand, d, bias=False)
        
        # Слияние
        self.merge = nn.Linear(d*2, d, bias=False)
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        r = x
        x = self.norm(x)
        B, T, D = x.shape
        
        # SSM ветка
        xz = self.ssm_in(x)
        xs, z = xz.chunk(2, dim=-1)
        xs = F.silu(self.conv(xs.transpose(1,2))[:,:,:T].transpose(1,2))
        
        y_ssm = ParallelSSMScan.apply(
            xs, F.softplus(self.dt_proj(xs)),
            self.B_proj(xs), self.C_proj(xs),
            -torch.exp(self.log_A.float())
        )
        y_ssm = (y_ssm + xs*self.D) * F.silu(z)
        y_ssm = self.ssm_out(y_ssm)
        
        # FFN ветка (параллельно!)
        y_ffn = self.ffn_w2(F.gelu(self.ffn_w1(x)))
        
        # Слияние
        y = self.merge(torch.cat([y_ssm, y_ffn], dim=-1))
        y = self.drop(y)
        
        return r + y

class RadicalMamba(nn.Module):
    def __init__(self, V=256, d=512, n_blocks=4, ds=32):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            FusedSSMFFN(d, ds=ds) for _ in range(n_blocks)
        ])
        
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.embed.weight
        
        params = sum(p.numel() for p in self.parameters())
        print(f"RadicalMamba: {params/1e6:.2f}M params, d={d}, blocks={n_blocks}")
    
    def forward(self, x):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

# ============================================================
# Radical 2: Один большой SSM + FFN
# ============================================================
class BigSSMBlock(nn.Module):
    def __init__(self, d, ds=64):
        super().__init__()
        self.norm = RMSNorm(d)
        
        self.in_proj = nn.Linear(d, d*2, bias=False)
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)
        self.dt_proj = nn.Linear(d, d)
        self.B_proj = nn.Linear(d, ds, bias=False)
        self.C_proj = nn.Linear(d, ds, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        
        self.log_A = nn.Parameter(torch.log(torch.linspace(1, 16, d)))
        self.D = nn.Parameter(torch.ones(d))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        r = x
        x = self.norm(x)
        B, T, D = x.shape
        
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        xs = F.silu(self.conv(xs.transpose(1,2))[:,:,:T].transpose(1,2))
        
        y = ParallelSSMScan.apply(
            xs, F.softplus(self.dt_proj(xs)),
            self.B_proj(xs), self.C_proj(xs),
            -torch.exp(self.log_A.float())
        )
        y = (y + xs*self.D) * F.silu(z)
        y = self.out_proj(self.drop(y))
        
        return r + y

class BigSSMMamba(nn.Module):
    def __init__(self, V=256, d=768, n_ssm=2, ds=64):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.drop = nn.Dropout(0.1)
        
        layers = []
        for _ in range(n_ssm):
            layers.append(BigSSMBlock(d, ds=ds))
            layers.append(nn.Sequential(
                RMSNorm(d),
                nn.Linear(d, d*2, bias=False),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d*2, d, bias=False),
                nn.Dropout(0.1),
            ))
        
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.embed.weight
        
        params = sum(p.numel() for p in self.parameters())
        print(f"BigSSMMamba: {params/1e6:.2f}M params, d={d}, ssm={n_ssm}")
    
    def forward(self, x):
        x = self.drop(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# ============================================================
# БЕНЧМАРК
# ============================================================
def bench(name, model, N=20):
    model = model.to(DEVICE)
    model = torch.compile(model, mode="reduce-overhead")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    x = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN+1), device=DEVICE)
    inp, tgt = x[:, :-1], x[:, 1:]
    
    # Warmup
    for _ in range(10):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(inp)
            loss = F.cross_entropy(out.view(-1,256), tgt.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(inp)
            loss = F.cross_entropy(out.view(-1,256), tgt.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / N * 1000
    toks = BATCH_SIZE * SEQ_LEN * 1000 / ms
    
    print(f"  {name:<40s} {ms:>7.1f} ms  {toks:>8.0f} tok/s  {262*8*256/ms/8/256:>6.1f}x")
    return ms, toks

print(f"\n{'='*75}")
print(f"  RADICAL ARCHITECTURES")
print(f"{'='*75}")
print(f"  {'Model':<40s} {'Time':>7s}  {'Tokens/s':>8s}  {'vs orig':>6s}")
print(f"  {'-'*70}")

bench("ORIG (8 SSM+8 FFN, d=256)", OrigMamba())
bench("RADICAL 1 (4 FusedSSMFFN, d=512)", RadicalMamba(d=512, n_blocks=4, ds=32))
bench("RADICAL 2 (2 BigSSM, d=768)", BigSSMMamba(d=768, n_ssm=2, ds=64))
bench("RADICAL 3 (1 BigSSM, d=1024)", BigSSMMamba(d=1024, n_ssm=1, ds=64))

print(f"{'='*75}")