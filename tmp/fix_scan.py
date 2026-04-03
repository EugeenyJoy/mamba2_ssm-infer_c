import torch

def stable_selective_scan(x, dt, B_mat, C_mat, A):
    """Chunked scan - numerically stable"""
    B_sz, T, DI = x.shape
    DS = B_mat.shape[2]
    
    CHUNK = 32  # короткие чанки — нет overflow
    
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    
    y = torch.zeros(B_sz, T, DI, device=x.device)
    h = torch.zeros(B_sz, DI, DS, device=x.device)  # hidden state
    
    for t0 in range(0, T, CHUNK):
        t1 = min(t0 + CHUNK, T)
        
        dt_c = dt_f[:, t0:t1]       # [B, chunk, DI]
        x_c = x_f[:, t0:t1]         # [B, chunk, DI]  
        B_c = B_mat[:, t0:t1].float()  # [B, chunk, DS]
        C_c = C_mat[:, t0:t1].float()  # [B, chunk, DS]
        
        # Внутри чанка — sequential scan (стабильный)
        for t in range(t1 - t0):
            dA = torch.exp(dt_c[:, t] * A_f)  # [B, DI] — decay
            dB_x = (dt_c[:, t] * x_c[:, t]).unsqueeze(-1) * B_c[:, t].unsqueeze(1)  # [B, DI, DS]
            
            h = h * dA.unsqueeze(-1) + dB_x   # [B, DI, DS]
            
            y_t = (h * C_c[:, t].unsqueeze(1)).sum(-1)  # [B, DI]
            y[:, t0 + t] = y_t
    
    return y

# Тест
B_sz, T, DI, DS = 2, 256, 256, 16
x = torch.randn(B_sz, T, DI, device='cuda')
dt = torch.randn(B_sz, T, DI, device='cuda').abs() * 0.1 + 0.5
B_m = torch.randn(B_sz, T, DS, device='cuda')
C_m = torch.randn(B_sz, T, DS, device='cuda')
A = -torch.exp(torch.log(torch.linspace(1, 16, DI))).cuda()

y = stable_selective_scan(x, dt, B_m, C_m, A)
print(f"y: min={y.min():.3f} max={y.max():.3f} nan={y.isnan().any()} inf={y.isinf().any()}")
