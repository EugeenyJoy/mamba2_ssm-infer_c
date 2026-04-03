from transformers import AutoTokenizer, MambaForCausalLM
import torch, sys

model_name = 'state-spaces/mamba-2.8b-hf'
print("Загрузка модели (30 сек)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/home/jqw/.hf_cache')
model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir='/home/jqw/.hf_cache')
print("Готово! Пиши (exit для выхода)\n")

while True:
    try:
        user = input("Ты: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user.strip().lower() in ('exit', 'quit', 'выход'):
        break
    if not user.strip():
        continue

    inputs = tokenizer(user, return_tensors='pt')
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2
        )
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Mamba: {response}\n")
