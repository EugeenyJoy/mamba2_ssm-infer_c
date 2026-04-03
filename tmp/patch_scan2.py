with open("mamba2_multi_train.py", "r") as f:
    code = f.read()

old = '''    def _scan_python(self, x, dt, B_mat, C_mat, A, B_batch, T):
        """Vectorized chunked scan (3.5x faster)"""
        CS = self.chunk_size
        DI = self.d_inner; DS = self.d_state
        h = torch.zeros(B_batch, DI, DS,
                        device=x.device, dtype=x.dtype)
        outs = []
        for cs in range(0, T, CS):
            ce = min(cs + CS, T)
            dt_c = dt[:, cs:ce]
            B_c = B_mat[:, cs:ce]
            C_c = C_mat[:, cs:ce]
            x_c = x[:, cs:ce]
            log_d = dt_c.unsqueeze(-1) * A.view(1, 1, DI, 1)
            cumlog = torch.cumsum(log_d, dim=1)
            exp_cumlog = torch.exp(cumlog)
            inp = dt_c.unsqueeze(-1) * B_c.unsqueeze(2) * x_c.unsqueeze(-1)
            scaled_inp = inp * torch.exp(-cumlog)
            prefix_sum = torch.cumsum(scaled_inp, dim=1)
            chunk_h = exp_cumlog * (prefix_sum + h.unsqueeze(1))
            h = chunk_h[:, -1]
            y_chunk = (chunk_h * C_c.unsqueeze(2)).sum(-1)
            outs.append(y_chunk)
        return torch.cat(outs, dim=1)'''

new = '''    def _scan_python(self, x, dt, B_mat, C_mat, A, B_batch, T):
        """Vectorized chunked scan (3.5x faster, fp32-safe)"""
        CS = self.chunk_size
        DI = self.d_inner; DS = self.d_state
        # Force float32 to avoid AMP overflow in exp/cumsum
        orig_dtype = x.dtype
        x = x.float(); dt = dt.float()
        B_mat = B_mat.float(); C_mat = C_mat.float()
        A = A.float()
        h = torch.zeros(B_batch, DI, DS, device=x.device, dtype=torch.float32)
        outs = []
        for cs in range(0, T, CS):
            ce = min(cs + CS, T)
            dt_c = dt[:, cs:ce]
            B_c = B_mat[:, cs:ce]
            C_c = C_mat[:, cs:ce]
            x_c = x[:, cs:ce]
            log_d = dt_c.unsqueeze(-1) * A.view(1, 1, DI, 1)
            log_d = torch.clamp(log_d, min=-20.0, max=20.0)
            cumlog = torch.cumsum(log_d, dim=1)
            cumlog = torch.clamp(cumlog, min=-20.0, max=20.0)
            exp_cumlog = torch.exp(cumlog)
            inp = dt_c.unsqueeze(-1) * B_c.unsqueeze(2) * x_c.unsqueeze(-1)
            scaled_inp = inp * torch.exp(-cumlog)
            prefix_sum = torch.cumsum(scaled_inp, dim=1)
            chunk_h = exp_cumlog * (prefix_sum + h.unsqueeze(1))
            h = chunk_h[:, -1]
            y_chunk = (chunk_h * C_c.unsqueeze(2)).sum(-1)
            outs.append(y_chunk)
        return torch.cat(outs, dim=1).to(orig_dtype)'''

if old in code:
    code = code.replace(old, new)
    with open("mamba2_multi_train.py", "w") as f:
        f.write(code)
    print("✅ scan стабилизирован (float32 + clamp)")
else:
    print("❌ Не нашёл scan")