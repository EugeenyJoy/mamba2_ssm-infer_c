with open("mamba2_multi_train.py", "r") as f:
    lines = f.readlines()

# Найти начало generate
start = None
end = None
for i, line in enumerate(lines):
    if line.startswith("def generate("):
        start = i
    elif start and not line.startswith(" ") and not line.strip() == "" and i > start + 2:
        end = i
        break

if start is None:
    print("❌ Не нашёл generate")
else:
    if end is None:
        end = len(lines)
    
    new_func = '''def generate(model, prompt_bytes, length=200, temp=0.8):
    model.eval()
    tokens = list(prompt_bytes)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(length):
            logits = model(x)
            logits = logits[0, -1]
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            if temp < 0.01:
                nt = logits.argmax().unsqueeze(0)
            else:
                probs = F.softmax(logits / temp, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum()
                nt = torch.multinomial(probs, 1)
            tokens.append(nt.item())
            x = torch.cat([x, nt.unsqueeze(0)], dim=1)
            if x.shape[1] > SEQ_LEN:
                x = x[:, -SEQ_LEN:]
    try:
        return bytes(tokens).decode("utf-8", errors="replace")
    except:
        return ""

'''
    lines[start:end] = [new_func]
    with open("mamba2_multi_train.py", "w") as f:
        f.writelines(lines)
    print(f"✅ generate заменён (строки {start+1}-{end})")