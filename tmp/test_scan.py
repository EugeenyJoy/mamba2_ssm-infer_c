import torch, time

B, D, S, T = 8, 192, 32, 256
device = "cuda"

x = torch.randn(B, T, D, device=device)
dt = torch.randn(B, T, D, device=device).abs() * 0.1
B_mat = torch.randn(B, T, S, device=device)
C_mat = torch.randn(B, T, S, device=device)
A = -torch.ones(D, device=device)

def scan_old(x, dt, B_mat, C_mat, A):
    B_sz, T, D = x.shape
    S = B_mat.shape[-1]
    h = torch.zeros(B_sz, D, S, device=x.device)
    outs = []
    for t in range(T):
        decay = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))
        inp = dt[:, t].unsqueeze(-1) * B_mat[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
        h = h * decay + inp
        y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
        outs.append(y_t)
    return torch.stack(outs, dim=1)

def scan_fast(x, dt, B_mat, C_mat, A):
    B_sz, T, D = x.shape
    S = B_mat.shape[-1]
    CS = 32
    h = torch.zeros(B_sz, D, S, device=x.device)
    outs = []

    for cs in range(0, T, CS):
        ce = min(cs + CS, T)
        L = ce - cs

        dt_c = dt[:, cs:ce]
        B_c = B_mat[:, cs:ce]
        C_c = C_mat[:, cs:ce]
        x_c = x[:, cs:ce]

        # log-decay per step [B, L, D, S]
        log_d = dt_c.unsqueeze(-1) * A.view(1, 1, D, 1)
        decay_step = torch.exp(log_d)

        # input per step [B, L, D, S]
        inp = dt_c.unsqueeze(-1) * B_c.unsqueeze(2) * x_c.unsqueeze(-1)

        # Рекуррентность внутри чанка через cumsum в log-пространстве:
        # h[t] = decay[t]*h[t-1] + inp[t]
        # h[t] = sum_{s=0}^{t} prod_{k=s+1}^{t} decay[k] * inp[s]  +  prod_{k=0}^{t} decay[k] * h_prev
        #
        # prod_{k=s+1}^{t} decay[k] = exp(sum_{k=s+1}^{t} log_d[k])
        #   = exp(cumlog[t] - cumlog[s])
        #   where cumlog[t] = sum_{k=0}^{t} log_d[k]
        #
        # h[t] = exp(cumlog[t]) * [ sum_{s=0}^{t} inp[s] * exp(-cumlog[s]) ] + exp(cumlog[t]) * h_prev
        #
        # НО! decay[s] применяется к inp[s-1], не к inp[s].
        # h[0] = decay[0]*h_prev + inp[0]
        # h[1] = decay[1]*h[0] + inp[1] = decay[1]*decay[0]*h_prev + decay[1]*inp[0] + inp[1]
        #
        # Значит для inp[s]: множитель = prod_{k=s+1}^{t} decay[k]
        # = exp(cumlog[t] - cumlog[s])
        #
        # cumlog[t] = sum_{k=0}^{t} log_d[k]
        # prod_{k=s+1}^{t} = exp(cumlog[t] - cumlog[s])
        #
        # h[t] = sum_{s=0}^{t} exp(cumlog[t]-cumlog[s]) * inp[s] + exp(cumlog[t]) * h_prev
        #       (где h_prev уже домножен на decay[0] через cumlog)
        #
        # Нет! Проверим руками:
        # h[0] = decay[0]*h_prev + inp[0]
        # cumlog[0] = log_d[0]
        # exp(cumlog[0] - cumlog[0]) * inp[0] = inp[0] ✓
        # exp(cumlog[0]) * h_prev = decay[0] * h_prev ✓
        #
        # h[1] = decay[1]*decay[0]*h_prev + decay[1]*inp[0] + inp[1]
        # exp(cumlog[1]-cumlog[0])*inp[0] = decay[1]*inp[0] ✓
        # exp(cumlog[1]-cumlog[1])*inp[1] = inp[1] ✓
        # exp(cumlog[1])*h_prev = decay[0]*decay[1]*h_prev ✓
        #
        # Всё верно!

        cumlog = torch.cumsum(log_d, dim=1)  # [B, L, D, S]

        # h[t] = exp(cumlog[t]) * sum_{s=0}^{t} [inp[s] * exp(-cumlog[s])] + exp(cumlog[t]) * h_prev
        scaled_inp = inp * torch.exp(-cumlog)
        prefix_sum = torch.cumsum(scaled_inp, dim=1)  # [B, L, D, S]

        exp_cumlog = torch.exp(cumlog)

        chunk_h = exp_cumlog * prefix_sum + exp_cumlog * h.unsqueeze(1)

        # Update h
        h = chunk_h[:, -1]

        # Output
        y_chunk = (chunk_h * C_c.unsqueeze(2)).sum(-1)
        outs.append(y_chunk)

    return torch.cat(outs, dim=1)

# Warmup
for _ in range(3):
    scan_old(x, dt, B_mat, C_mat, A)
    scan_fast(x, dt, B_mat, C_mat, A)
torch.cuda.synchronize()

r1 = scan_old(x, dt, B_mat, C_mat, A)
r2 = scan_fast(x, dt, B_mat, C_mat, A)
diff = (r1 - r2).abs().max().item()
rdiff = ((r1 - r2).abs() / (r1.abs() + 1e-8)).mean().item()
print(f"Max abs diff:  {diff:.8f}")
print(f"Mean rel diff: {rdiff:.8f}")

if diff < 0.01:
    print("✅ КОРРЕКТНО!")

    t0 = time.time()
    for _ in range(30):
        scan_old(x, dt, B_mat, C_mat, A)
    torch.cuda.synchronize()
    old_t = time.time() - t0

    t0 = time.time()
    for _ in range(30):
        scan_fast(x, dt, B_mat, C_mat, A)
    torch.cuda.synchronize()
    new_t = time.time() - t0

    print(f"Старый: {old_t:.3f}s")
    print(f"Новый:  {new_t:.3f}s")
    print(f"Ускорение: {old_t/new_t:.1f}x")
else:
    print(f"❌ diff={diff:.4f}")