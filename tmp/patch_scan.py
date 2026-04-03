import torch
import torch.nn.functional as F
import sys

# Загружаем модуль
sys.path.insert(0, '.')
import mamba2_multi_train as m

# Новый стабильный scan с autograd
class StableScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, B_mat, C_mat, A):
        B_sz, T, DI = x.shape
        DS = B_mat.shape[2]
        
        x_f = x.float()
        dt_f = dt.float()
        A_f = A.float()
        B_f = B_mat.float()
        C_f = C_mat.float()
        
        y = torch.zeros(B_sz, T, DI, device=x.device)
        
        # Сохраним все h для backward
        h_states = torch.zeros(B_sz, T, DI, DS, device=x.device)
        h = torch.zeros(B_sz, DI, DS, device=x.device)
        
        for t in range(T):
            dA = torch.exp(dt_f[:, t] * A_f)           # [B, DI]
            dBx = (dt_f[:, t] * x_f[:, t]).unsqueeze(-1) * B_f[:, t].unsqueeze(1)  # [B, DI, DS]
            h = h * dA.unsqueeze(-1) + dBx
            h_states[:, t] = h
            y[:, t] = (h * C_f[:, t].unsqueeze(1)).sum(-1)
        
        ctx.save_for_backward(x_f, dt_f, B_f, C_f, A_f, h_states)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, dt, B_mat, C_mat, A, h_states = ctx.saved_tensors
        B_sz, T, DI = x.shape
        DS = B_mat.shape[2]
        dy = dy.float()
        
        dx = torch.zeros_like(x)
        ddt = torch.zeros_like(dt)
        dB = torch.zeros_like(B_mat)
        dC = torch.zeros_like(C_mat)
        dA = torch.zeros_like(A)
        
        dh = torch.zeros(B_sz, DI, DS, device=x.device)
        
        for t in range(T - 1, -1, -1):
            # dy/dC: y_t = sum(h_t * C_t)
            dC[:, t] = (dy[:, t].unsqueeze(-1) * h_states[:, t]).sum(1)
            
            # dy/dh -> dh
            dh += dy[:, t].unsqueeze(-1) * C_mat[:, t].unsqueeze(1)
            
            # h_t = h_{t-1} * exp(dt*A) + dt*x*B
            # dh/d(dt*x*B) = dh itself
            dBx = dh  # [B, DI, DS]
            
            # d(dt*x) from dBx
            d_dtx = (dBx * B_mat[:, t].unsqueeze(1)).sum(-1)  # [B, DI]
            dB[:, t] = (dBx * (dt[:, t] * x[:, t]).unsqueeze(-1)).sum(1)
            
            dx[:, t] = d_dtx * dt[:, t]
            ddt[:, t] = d_dtx * x[:, t]
            
            # dh propagate: h_t depends on h_{t-1} via exp(dt*A)
            decay = torch.exp(dt[:, t] * A)  # [B, DI]
            
            # dA contribution
            if t > 0:
                h_prev = h_states[:, t-1]
            else:
                h_prev = torch.zeros_like(h_states[:, 0])
            
            ddt[:, t] += (dh * h_prev * decay.unsqueeze(-1) * A.view(1, -1, 1)).sum(-1)
            dA += (dh * h_prev * decay.unsqueeze(-1) * dt[:, t].unsqueeze(-1)).sum((0, -1))
            
            dh = dh * decay.unsqueeze(-1)
        
        return dx, ddt, dB, dC, dA

def stable_scan_fn(x, dt, B, C, A):
    return StableScan.apply(x, dt, B, C, A)

# Подменяем scan в модуле
m.parallel_ssm_scan_v2 = stable_scan_fn

# Тест обучения
print("=== Тест с исправленным scan ===")
torch.manual_seed(42)
model = m.Mamba2LM().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

text = ('Привет мир! Hello world! Как дела? '*50).encode('utf-8')
data = torch.tensor(list(text), dtype=torch.long, device='cuda')

for step in range(20):
    idx = torch.randint(0, len(data)-65, (4,), device='cuda')
    batch = torch.stack([data[i:i+65] for i in idx])
    inp, tgt = batch[:, :-1], batch[:, 1:]
    
    opt.zero_grad()
    out = model(inp)
    loss = F.cross_entropy(out.view(-1, 256), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    if step % 5 == 0:
        print(f"  step {step:2d}: loss={loss.item():.4f}")

print("DONE!")
