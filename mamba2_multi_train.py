"""
MAMBA-2 MULTILINGUAL TRAINER v3
Русский + Английский + Код
Автоматическая загрузка данных из data/
Постепенное обучение с чекпоинтами

python mamba2_multi_train.py train
python mamba2_multi_train.py finetune
python mamba2_multi_train.py interactive
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, json, os, struct, glob, re, random
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  MAMBA-2 MULTILINGUAL TRAINER v3")
print("=" * 60)

import triton
import triton
import triton.language as tl
import triton.language as tl

@triton.jit
def _ssm_scan_fwd_kernel(
    x_ptr, dt_ptr, B_ptr, C_ptr, A_ptr, y_ptr, h_ptr,
    B_sz, T: tl.constexpr, DI: tl.constexpr, DS: tl.constexpr,
    stride_xb, stride_xt, stride_xi,
    stride_bb, stride_bt, stride_bs,
):
    # Один блок = один (batch, di) pair
    pid = tl.program_id(0)
    b = pid // DI
    di = pid % DI

    # h state для этого (b, di) — DS элементов
    h = tl.zeros([DS], dtype=tl.float32)
    a_val = tl.load(A_ptr + di)

    for t in range(T):
        # load dt, x scalar
        dt_val = tl.load(dt_ptr + b * T * DI + t * DI + di).to(tl.float32)
        x_val = tl.load(x_ptr + b * T * DI + t * DI + di).to(tl.float32)

        # load B[b, t, :] vector of DS
        b_offs = tl.arange(0, DS)
        b_vec = tl.load(B_ptr + b * T * DS + t * DS + b_offs).to(tl.float32)

        # load C[b, t, :] vector of DS
        c_vec = tl.load(C_ptr + b * T * DS + t * DS + b_offs).to(tl.float32)

        # decay
        decay = tl.exp(tl.minimum(dt_val * a_val, 0.0))  # clamp to <=0

        # update h
        dBx = dt_val * x_val * b_vec  # [DS]
        h = h * decay + dBx

        # output
        y_val = tl.sum(h * c_vec, axis=0)
        tl.store(y_ptr + b * T * DI + t * DI + di, y_val)

        # store h for backward
        h_offs = tl.arange(0, DS)
        h_base = b * (T + 1) * DI * DS + (t + 1) * DI * DS + di * DS
        tl.store(h_ptr + h_base + h_offs, h)


@triton.jit
def _ssm_scan_bwd_kernel(
    dy_ptr, x_ptr, dt_ptr, B_ptr, C_ptr, A_ptr, h_ptr,
    dx_ptr, ddt_ptr, dB_ptr, dC_ptr, dA_ptr,
    B_sz, T: tl.constexpr, DI: tl.constexpr, DS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // DI
    di = pid % DI

    dh = tl.zeros([DS], dtype=tl.float32)
    a_val = tl.load(A_ptr + di)
    da_local = 0.0
    s_offs = tl.arange(0, DS)

    for t in range(T - 1, -1, -1):
        base_t = b * T * DI + t * DI + di
        base_s = b * T * DS + t * DS

        dy_val = tl.load(dy_ptr + base_t).to(tl.float32)
        x_val = tl.load(x_ptr + base_t).to(tl.float32)
        dt_val = tl.load(dt_ptr + base_t).to(tl.float32)

        b_vec = tl.load(B_ptr + base_s + s_offs).to(tl.float32)
        c_vec = tl.load(C_ptr + base_s + s_offs).to(tl.float32)

        # h at t and t-1
        h_base_t = b * (T + 1) * DI * DS + (t + 1) * DI * DS + di * DS
        h_t = tl.load(h_ptr + h_base_t + s_offs)

        h_base_prev = b * (T + 1) * DI * DS + t * DI * DS + di * DS
        h_prev = tl.load(h_ptr + h_base_prev + s_offs)

        # dC[b,t,s] += dy * h_t[s]  (but dC is [B,T,DS] shared across DI — need atomic)
        # Actually dC[b,t] = sum over DI of dy * h_t — need atomicAdd
        dc_contrib = dy_val * h_t
        tl.atomic_add(dC_ptr + base_s + s_offs, dc_contrib)

        dh = dh + dy_val * c_vec

        d_dtx = tl.sum(dh * b_vec, axis=0)
        db_contrib = dh * (dt_val * x_val)
        tl.atomic_add(dB_ptr + base_s + s_offs, db_contrib)

        tl.store(dx_ptr + base_t, d_dtx * dt_val)

        decay = tl.exp(tl.minimum(dt_val * a_val, 0.0))
        ddt_val = d_dtx * x_val + tl.sum(dh * h_prev * decay * a_val, axis=0)
        tl.store(ddt_ptr + base_t, ddt_val)

        da_local += tl.sum(dh * h_prev * decay * dt_val, axis=0)
        dh = dh * decay

    tl.atomic_add(dA_ptr + di, da_local)


# Load triton kernels

class StableSSMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, B_mat, C_mat, A):
        B_sz, T, DI = x.shape
        DS = B_mat.shape[2]
        x_f, dt_f = x.float(), dt.float()
        A_f, B_f, C_f = A.float(), B_mat.float(), C_mat.float()

        y = torch.zeros(B_sz, T, DI, device=x.device)
        h_states = torch.zeros(B_sz, T+1, DI, DS, device=x.device)

        grid = (B_sz * DI,)
        _ssm_scan_fwd_kernel[grid](
            x_f, dt_f, B_f, C_f, A_f, y, h_states,
            B_sz, T, DI, DS,
            x_f.stride(0), x_f.stride(1), x_f.stride(2),
            B_f.stride(0), B_f.stride(1), B_f.stride(2),
        )
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

        grid = (B_sz * DI,)
        _ssm_scan_bwd_kernel[grid](
            dy, x, dt, B_mat, C_mat, A, h_states,
            dx, ddt, dB, dC, dA,
            B_sz, T, DI, DS,
        )
        return dx, ddt, dB, dC, dA

def parallel_ssm_scan_v2(x, dt, B, C, A):
    return StableSSMScan.apply(x, dt, B, C, A)

# AUTOCONFIG
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM = torch.cuda.mem_get_info(0)[1] / 1e9
    print(f"GPU: {GPU_NAME} ({VRAM:.1f} GB)")
    if VRAM >= 14:
        BATCH = 16; MODEL_SIZE = "large"
    elif VRAM >= 3.5:
        BATCH = 8; MODEL_SIZE = "medium"
    else:
        BATCH = 4; MODEL_SIZE = "small"
else:
    DEVICE = "cpu"; VRAM = 0
    BATCH = 4; MODEL_SIZE = "small"

SEQ_LEN = 256
BATCH_SIZE = 8
print(f"Device: {DEVICE} | Size: {MODEL_SIZE} | Seq: {SEQ_LEN}")


# ============================================================
# DATA BUDGET — сколько данных брать
# ============================================================
DATA_BUDGET = {
    "small":  {"total_mb": 4,  "ru_frac": 0.40, "en_frac": 0.25,
               "dialog_frac": 0.20, "code_frac": 0.15},
    "medium": {"total_mb": 6,  "ru_frac": 0.35, "en_frac": 0.25,
               "dialog_frac": 0.25, "code_frac": 0.15},
    "large":  {"total_mb": 10, "ru_frac": 0.35, "en_frac": 0.25,
               "dialog_frac": 0.25, "code_frac": 0.15},
}


# ============================================================
# CODE SAMPLES (встроенные, если нет файлов)
# ============================================================
BUILTIN_CODE = '''# Python: сортировка пузырьком
def bubble_sort(arr):
    """Сортировка пузырьком - простой алгоритм."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Python: бинарный поиск
def binary_search(arr, target):
    """Бинарный поиск в отсортированном массиве."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Python: чтение файла и подсчет слов
def count_words(filename):
    """Считает количество слов в файле."""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    return len(words)

# Python: класс для работы со стеком
class Stack:
    """Реализация стека на основе списка."""
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Python: HTTP запрос
import requests

def fetch_json(url):
    """Загружает JSON по URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

# Python: работа с файлами
import os
import json

def save_config(config, path="config.json"):
    """Сохраняет конфигурацию в JSON файл."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(path="config.json"):
    """Загружает конфигурацию из JSON файла."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Python: декоратор для замера времени
import time
import functools

def timer(func):
    """Декоратор для замера времени выполнения."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} выполнена за {elapsed:.3f} сек")
        return result
    return wrapper

# Python: генератор Фибоначчи
def fibonacci(n):
    """Генерирует первые n чисел Фибоначчи."""
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

# Python: простая нейронная сеть
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """Простая нейронная сеть для классификации."""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# C: связный список
/*
 * Реализация связного списка на C
 */
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

Node* create_node(int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->next = NULL;
    return node;
}

void push(Node** head, int data) {
    Node* node = create_node(data);
    node->next = *head;
    *head = node;
}

void print_list(Node* head) {
    Node* current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\\n");
}

void free_list(Node* head) {
    Node* tmp;
    while (head != NULL) {
        tmp = head;
        head = head->next;
        free(tmp);
    }
}

/* C: матричное умножение */
void matmul(float* out, const float* a, const float* b,
            int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * K + j];
            }
            out[i * K + j] = sum;
        }
    }
}

/* C: quicksort */
void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// JavaScript: fetch с обработкой ошибок
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Fetch failed:", error.message);
        return null;
    }
}

// JavaScript: debounce функция
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

# Python: обработка текста
import re

def clean_text(text):
    """Очищает текст от лишних символов."""
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'[^\\w\\s.,!?;:\\-()]', '', text)
    return text.strip()

def tokenize(text):
    """Простая токенизация по пробелам и знакам."""
    tokens = re.findall(r'\\w+|[.,!?;:]', text)
    return tokens

def ngrams(tokens, n=2):
    """Создает n-граммы из списка токенов."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
'''

# Встроенный русский текст (если нет книг)
BUILTIN_RU = """Россия — крупнейшее государство мира по площади территории. Столица — Москва. Население составляет более ста сорока миллионов человек. Государственный язык — русский.

Москва является столицей Российской Федерации. Город расположен на реке Москве и был основан в тысяча сто сорок седьмом году. В Москве расположен Кремль и Красная площадь.

Русская литература является одной из величайших в мире. Пушкин считается основоположником современного русского языка. Толстой написал Войну и мир. Достоевский создал Преступление и наказание. Чехов известен своими пьесами.

Математика изучает количественные отношения. Физика исследует законы природы. Химия изучает вещества. Биология исследует живые организмы. Информатика занимается обработкой информации.

Искусственный интеллект создаёт интеллектуальные программы. Нейронные сети вдохновлены строением мозга. Глубокое обучение использует многослойные сети. Модели пространства состояний являются перспективной альтернативой трансформерам.

Программирование — процесс создания компьютерных программ. Python используется для машинного обучения. Язык C применяется в системном программировании. Каждый язык имеет свои преимущества и области применения.

Байкал — самое глубокое озеро мира. Волга — самая длинная река Европы. Сибирь богата природными ресурсами. Космонавтика связана с полётами в космос. Гагарин совершил первый полёт.

"""

BUILTIN_EN = """Science and technology continue to advance rapidly. Artificial intelligence is transforming many industries. Machine learning allows computers to learn from data. Neural networks process information in layers.

Programming is the process of creating software. Python is popular for data science and machine learning. The C language is used for system programming. JavaScript runs in web browsers. Rust provides memory safety without garbage collection.

Mathematics is the foundation of computer science. Algorithms are step by step procedures for solving problems. Data structures organize information efficiently. Complexity analysis measures how resources scale with input size.

The history of computing began with mechanical calculators. Charles Babbage designed the first general purpose computer. Alan Turing formalized the concept of computation. Modern computers can perform billions of operations per second.

"""


# ============================================================
# DATA LOADING
# ============================================================
def classify_file(filepath):
    """Определяет тип файла: ru, en, dialog, code"""
    name = os.path.basename(filepath).lower()

    # По имени
    if 'english' in name or 'eng_' in name:
        return 'en'
    if 'code' in name or name.endswith('.py') or name.endswith('.c') or name.endswith('.js'):
        return 'code'
    if 'dialog' in name or 'chat' in name or 'conversation' in name:
        return 'dialog'

    # По содержимому
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(5000)
    except:
        return None

    # Код?
    code_markers = ['def ', 'class ', 'import ', 'function ', '#include',
                    'void ', 'int ', 'return ', 'const ', 'var ', 'let ']
    code_hits = sum(1 for m in code_markers if m in sample)
    if code_hits >= 3:
        return 'code'

    # Русский или английский?
    cyrillic = sum(1 for c in sample if '\u0400' <= c <= '\u04ff')
    latin = sum(1 for c in sample if 'a' <= c.lower() <= 'z')

    if cyrillic > latin * 0.5 and cyrillic > 100:
        # Диалоги? (короткие строки, вопросы)
        lines = sample.split('\n')
        avg_len = sum(len(l) for l in lines if l.strip()) / max(len(lines), 1)
        questions = sum(1 for l in lines if '?' in l)
        if avg_len < 80 and questions > 5:
            return 'dialog'
        return 'ru'
    elif latin > 100:
        return 'en'

    return None


def load_text_chunk(filepath, max_bytes):
    """Загружает до max_bytes из файла, с случайного места."""
    try:
        size = os.path.getsize(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            if size <= max_bytes * 1.5:
                text = f.read()
            else:
                # Читаем с случайного места
                start = random.randint(0, max(0, size - max_bytes * 2))
                f.seek(start)
                f.readline()  # пропускаем обрезанную строку
                text = f.read(max_bytes * 2)
        # Обрезаем до нужного размера
        encoded = text.encode('utf-8')
        if len(encoded) > max_bytes:
            text = encoded[:max_bytes].decode('utf-8', errors='ignore')
            # Обрезаем до последнего полного предложения/строки
            last_nl = text.rfind('\n')
            if last_nl > len(text) * 0.8:
                text = text[:last_nl]
        return text
    except Exception as e:
        print(f"    ⚠️ Ошибка чтения {filepath}: {e}")
        return ""


def prepare_data(data_dir="data"):
    """Собирает данные из папки, классифицирует, нарезает."""
    budget = DATA_BUDGET[MODEL_SIZE]
    total_bytes = int(budget["total_mb"] * 1024 * 1024)

    print(f"\n📊 Подготовка данных (бюджет: {budget['total_mb']} МБ)")
    print(f"   Папка: {data_dir}/")

    # Находим файлы
    files = {"ru": [], "en": [], "dialog": [], "code": []}
    patterns = [os.path.join(data_dir, "*.txt"),
                os.path.join(data_dir, "*.py"),
                os.path.join(data_dir, "*.c"),
                os.path.join(data_dir, "*.js"),
                os.path.join(data_dir, "**", "*.txt")]

    found = set()
    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if fp not in found:
                found.add(fp)
                ftype = classify_file(fp)
                if ftype and ftype in files:
                    fsize = os.path.getsize(fp) / 1024
                    files[ftype].append(fp)
                    print(f"   [{ftype:>6}] {os.path.basename(fp)} ({fsize:.0f} КБ)")

    # Собираем текст по категориям
    chunks = {"ru": "", "en": "", "dialog": "", "code": ""}

    for cat in ["ru", "en", "dialog", "code"]:
        frac = budget.get(f"{cat}_frac", 0.15)
        cat_bytes = int(total_bytes * frac)

        if files[cat]:
            per_file = cat_bytes // len(files[cat]) + 1
            parts = []
            for fp in files[cat]:
                t = load_text_chunk(fp, per_file)
                if t:
                    parts.append(t)
            chunks[cat] = '\n\n'.join(parts)
        else:
            # Встроенные данные
            if cat == "ru":
                print(f"   📦 Нет русских текстов — встроенные данные")
                base = BUILTIN_RU
                while len(base.encode('utf-8')) < cat_bytes:
                    base = base + base
                chunks[cat] = base
            elif cat == "en":
                print(f"   📦 Нет английских текстов — встроенные данные")
                base = BUILTIN_EN
                while len(base.encode('utf-8')) < cat_bytes:
                    base = base + base
                chunks[cat] = base
            elif cat == "code":
                print(f"   📦 Нет кода — встроенные примеры")
                base = BUILTIN_CODE
                while len(base.encode('utf-8')) < cat_bytes:
                    base = base + base
                chunks[cat] = base
            elif cat == "dialog":
                print(f"   📦 Нет диалогов — пропускаем")

    # Собираем всё вместе (перемешиваем абзацы)
    all_paragraphs = []
    for cat, text in chunks.items():
        if text:
            paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
            for p in paras:
                all_paragraphs.append((cat, p))

    random.seed(42)
    random.shuffle(all_paragraphs)
    combined = '\n\n'.join(p for _, p in all_paragraphs)

    # Статистика
    total_size = len(combined.encode('utf-8'))
    cat_sizes = {}
    for cat, text in chunks.items():
        s = len(text.encode('utf-8'))
        cat_sizes[cat] = s

    print(f"\n   📈 Итого: {total_size/1024/1024:.1f} МБ")
    for cat in ["ru", "en", "dialog", "code"]:
        s = cat_sizes.get(cat, 0)
        pct = s / max(total_size, 1) * 100
        bar = "█" * int(pct / 3)
        print(f"   {cat:>8}: {s/1024:.0f} КБ ({pct:.0f}%) {bar}")

    # Train/val split
    split = int(len(combined) * 0.9)
    train_text = combined[:split]
    val_text = combined[split:]

    def to_bytes(t):
        return torch.tensor(list(t.encode('utf-8')),
                           dtype=torch.long).clamp(0, 255)

    tb = to_bytes(train_text)
    vb = to_bytes(val_text)
    print(f"\n   Train: {len(tb):,} bytes | Val: {len(vb):,} bytes")
    print(f"   Пример: {train_text[:80]}...")
    return tb, vb


class ByteDS(Dataset):
    def __init__(self, data, sl):
        self.data = data; self.sl = sl
    def __len__(self):
        return max(0, (len(self.data) - self.sl - 1) // self.sl)
    def __getitem__(self, i):
        s = i * self.sl
        return self.data[s:s+self.sl], self.data[s+1:s+self.sl+1]

# ============================================================
# MODEL
# ============================================================
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4,
                 chunk_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner,
                              kernel_size=d_conv, padding=d_conv-1,
                              groups=self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.log_A = nn.Parameter(
            torch.log(torch.linspace(1, 16, self.d_inner)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

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

        y = parallel_ssm_scan_v2(x_ssm, dt, B_mat, C_mat, A)
        y = y.to(x.dtype)  # ← Приведи к типу входа (float16)

        y = y + x_ssm * self.D
        y = y * F.silu(z)
        y = self.out_proj(self.drop(y))
        return residual + y



class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x):
        return x + self.net(self.norm(x))


class Mamba2LM(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_layers=8,
                 d_state=16, d_ff=512, chunk_size=64, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model,
                           n_layers=n_layers, d_state=d_state, d_ff=d_ff,
                           chunk_size=chunk_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SelectiveSSM(d_model, d_state,
                                            chunk_size=chunk_size, dropout=dropout))
            self.layers.append(FFN(d_model, d_ff, dropout))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        x = self.drop(self.embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_config(size):
    return {
        "small":  dict(d_model=256, n_layers=8,  d_state=16, d_ff=512,  chunk_size=64),
        "medium": dict(d_model=256, n_layers=8,  d_state=16, d_ff=512,  chunk_size=64),
        "large":  dict(d_model=320, n_layers=10, d_state=16, d_ff=640,  chunk_size=64),
    }[size]


# ============================================================
# EXPORT
# ============================================================
def export_weights(model, path="mamba2_multi_weights.bin"):
    params = {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu().float().numpy()
    cfg_json = json.dumps(model.config).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(cfg_json))); f.write(cfg_json)
        f.write(struct.pack("I", len(params)))
        for name, arr in params.items():
            nb = name.encode("utf-8")
            f.write(struct.pack("I", len(nb))); f.write(nb)
            f.write(struct.pack("I", arr.ndim))
            for s in arr.shape: f.write(struct.pack("I", s))
            f.write(arr.tobytes())
    sz = os.path.getsize(path)
    print(f"  💾 {path} ({sz/1e6:.1f} МБ)")


def generate(model, prompt_bytes, length=200, temp=0.8):
    model.eval()
    tokens = list(prompt_bytes)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(length):
            # Без AMP для генерации — используем float32
            logits = model(x)
            logits = logits.float()  # Гарантируем float32
            
            if temp < 0.01:
                nt = logits[0, -1].argmax().unsqueeze(0)
            else:
                probs = F.softmax(logits[0, -1] / temp, dim=-1)
                probs = probs.clamp(min=1e-7)  # Избегаем NaN
                nt = torch.multinomial(probs, 1)
            
            tokens.append(nt.item())
            x = torch.cat([x, nt.unsqueeze(0)], dim=1)
            if x.shape[1] > SEQ_LEN:
                x = x[:, -SEQ_LEN:]
    try:
        return bytes(tokens).decode('utf-8', errors='replace')
    except:
        return ""


# ============================================================
# TRAINING
# ============================================================
def train(epochs=8, lr=5e-5, resume=None, **kwargs):
    if epochs is None:
        epochs = 8
    if lr is None:
        lr = 5e-5
    cfg = get_config(MODEL_SIZE)

    data = prepare_data()
    if not data:
        return
    train_bytes, val_bytes = data

    train_ds = ByteDS(train_bytes, SEQ_LEN)
    val_ds = ByteDS(val_bytes, SEQ_LEN)
    BS = 8

    tld = DataLoader(train_ds, batch_size=BS, shuffle=True,
                     num_workers=0, pin_memory=True)
    vld = DataLoader(val_ds, batch_size=BS, shuffle=False,
                     num_workers=0, pin_memory=True)

    model = Mamba2LM(**cfg)

    # Resume
    if resume and os.path.exists(resume):
        sd = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(sd)
        print(f"  ✅ Загружен чекпоинт: {resume}")

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  🧠 Mamba-2 | {total_params:,} params ({total_params*4/1e6:.1f} МБ)")
    print(f"  📊 {len(tld)} train | {len(vld)} val батчей")
    print(f"  ⚙️  lr={lr} epochs={epochs} batch={BS}")

    # === AMP (float16) ===
    use_amp = (DEVICE == "cuda")
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"  ⚡ AMP float16: ДА")
    else:
        scaler = None
        print(f"  ⚡ AMP float16: НЕТ (CPU)")

    print(f"  ⚡ torch.compile: ВЫКЛ (экономия RAM)")

    # Тест генерации ДО обучения
    # print(f"\n  📝 ДО дообучения:")
    # model.eval()
    # with torch.no_grad():
    #     for label, prompt in [("ru", "Россия — это "),
    #                           ("en", "The history of computing "),
    #                           ("code", "def sort(")]:
    #         text = generate(model, prompt.encode('utf-8'), length=50, temp=0.7)
    #         print(f"    [{label}] {text[:70]}")
    # model.train()

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(tld) * epochs
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n  🚀 Старт...\n")

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tt = 0.0, 0
        t0 = time.time()

        for bi, (x, y) in enumerate(tld):
            x, y = x.to(DEVICE), y.to(DEVICE)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            sched.step()
            tl += loss.item() * y.numel()
            tt += y.numel()

            # Прогресс каждые 20 батчей
            if (bi + 1) % 20 == 0:
                elapsed = time.time() - t0
                speed = (bi + 1) / elapsed
                eta = (len(tld) - bi - 1) / speed
                cur_loss = tl / tt
                mem = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
                print(f"    Эп {epoch}/{epochs} "
                      f"[{bi+1:4d}/{len(tld)}] "
                      f"loss={cur_loss:.4f} "
                      f"lr={sched.get_last_lr()[0]:.6f} "
                      f"{speed:.1f} b/s "
                      f"GPU:{mem:.1f}GB "
                      f"ETA:{eta:.0f}s   ", end="\r")

        train_time = time.time() - t0
        train_loss = tl / tt

        # Validation
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in vld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        lg = model(x)
                        vl += F.cross_entropy(
                            lg.view(-1, 256), y.view(-1)
                        ).item() * y.numel()
                else:
                    lg = model(x)
                    vl += F.cross_entropy(
                        lg.view(-1, 256), y.view(-1)
                    ).item() * y.numel()
                vc += (lg.argmax(-1) == y).sum().item()
                vt += y.numel()

        val_loss = vl / vt
        val_acc = vc / vt * 100
        val_ppl = math.exp(min(val_loss, 10))

        print(f"\n  Эп {epoch}/{epochs} "
              f"| T:{train_loss:.4f} V:{val_loss:.4f} "
              f"| Acc:{val_acc:.1f}% PPL:{val_ppl:.2f} "
              f"| {train_time:.0f}s")

        # Генерация примеров
        with torch.no_grad():
            for label, prompt in [("ru", "Россия — это "),
                                  ("en", "The "),
                                  ("code", "def ")]:
                text = generate(model, prompt.encode('utf-8'), length=50, temp=0.7)
                print(f"    [{label}] {text[:75]}")

        # Сохранение
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'mamba2_multi.pt')
            print(f"    💾 Сохранено (best val={val_loss:.4f})")
        else:
            torch.save(model.state_dict(), 'mamba2_multi.pt')
            print(f"    💾 Сохранено")

    # Экспорт для C
    print(f"\n  📦 Экспорт весов для C...")
    export_weights(model, 'mamba2_multi_weights.bin')
    print(f"  ✅ Готово!")

    # ============================================================
    # RESULTS
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  РЕЗУЛЬТАТЫ ({mode_str})")
    print(f"{'='*60}")
    print(f"  Параметры: {np_:,} ({np_*4/1e6:.1f} МБ)")
    print(f"  Лучший val loss: {best:.4f}")
    print(f"  Финал: acc={history[-1]['va']:.4f} ppl={history[-1]['vp']:.1f}")
    print(f"  Общее время: {sum(h['t'] for h in history):.0f}с")

    # Speed test
    model.eval()
    x = torch.randint(0, 256, (1, 50)).to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            lg = model(x)
            nt = lg[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            x = torch.cat([x, nt], dim=1)
            if x.shape[1] > SEQ_LEN:
                x = x[:, -SEQ_LEN:]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        tms = (time.time() - t0) / 100 * 1000
    print(f"  Python: {tms:.1f} мс/ток = {1000/tms:.0f} ток/с")

    # Learning curve
    print(f"\n  📈 Кривая обучения:")
    for h in history:
        bar = "█" * max(1, int(50 * h["va"]))
        print(f"    Эп{h['ep']:>2} V:{h['vl']:.4f} Acc:{h['va']:.4f} {bar}")

    # Final generation
    print(f"\n  📝 Генерация:")
    all_prompts = [
        ("Россия — это ", 0.7),
        ("Привет, как ", 0.7),
        ("The history of ", 0.7),
        ("def fibonacci(", 0.5),
        ("int main(", 0.5),
        ("Жил-был ", 0.8),
    ]
    for prompt, temp in all_prompts:
        s = generate(model, prompt.encode('utf-8'), length=150, temp=temp)
        print(f"    [{prompt.strip()}]:")
        print(f"    {s[:120]}")
        print()

    # Export
    print(f"{'='*60}")
    print(f"  ЭКСПОРТ")
    print(f"{'='*60}")
    export_weights(model, f"{save_name}_weights.bin")
    torch.save(model.state_dict(), f"{save_name}.pt")

    info = dict(config=cfg, params=np_, best_loss=best,
                final_acc=history[-1]["va"],
                final_ppl=history[-1]["vp"],
                tok_ms=tms, history=history,
                model_size=MODEL_SIZE,
                languages=["ru", "en", "code"])
    with open(f"{save_name}_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  📋 {save_name}_info.json")
    print(f"\n  Следующий шаг:")
    print(f"    ./mamba2 {save_name}_weights.bin \"Привет\" 200 0.7")

    return model, history

# ============================================================
# FINE-TUNE — дообучение на диалогах
# ============================================================
def finetune(base_model="mamba2_multi.pt", data_dir="data",
             epochs=5, lr=5e-5):
    """
    Дообучение на диалогах.
    Автоматически берёт больше диалогов, меньше книг.
    """
    print(f"\n{'='*60}")
    print(f"  ДООБУЧЕНИЕ НА ДИАЛОГАХ")
    print(f"{'='*60}")

    if not os.path.exists(base_model):
        print(f"  ❌ Нет базовой модели: {base_model}")
        print(f"     Сначала: python mamba2_multi_train.py train")
        return None

    # Меняем бюджет: больше диалогов
    global DATA_BUDGET
    old_budget = DATA_BUDGET[MODEL_SIZE].copy()
    DATA_BUDGET[MODEL_SIZE] = {
        "total_mb": old_budget["total_mb"] * 0.5,  # меньше данных
        "ru_frac": 0.10,       # чуть книг для памяти
        "en_frac": 0.05,       # чуть английского
        "dialog_frac": 0.75,   # МНОГО диалогов
        "code_frac": 0.10,     # чуть кода
    }

    model, history = train(
        epochs=epochs,
        lr=lr,
        data_dir=data_dir,
        resume=base_model,
        save_name="mamba2_multi_ft"
    )

    # Восстанавливаем бюджет
    DATA_BUDGET[MODEL_SIZE] = old_budget

    return model


# ============================================================
# INTERACTIVE
# ============================================================
def interactive(model_path="mamba2_multi.pt"):
    """Интерактивный режим."""
    print(f"\n{'='*60}")
    print(f"  ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print(f"{'='*60}")

    cfg = get_config(MODEL_SIZE)
    model = Mamba2LM(**cfg)

    # Пробуем найти модель
    candidates = [model_path, "mamba2_multi_ft.pt", "mamba2_multi.pt"]
    loaded = False
    for cp in candidates:
        if os.path.exists(cp):
            model.load_state_dict(torch.load(cp, map_location="cpu",
                                              weights_only=True))
            print(f"  ✅ Модель: {cp} ({model.count_params():,} params)")
            loaded = True
            break

    if not loaded:
        print(f"  ❌ Модель не найдена!")
        print(f"     Сначала обучи: python mamba2_multi_train.py train")
        return

    model = model.to(DEVICE)
    model.eval()

    print(f"\n  Введите текст — модель продолжит.")
    print(f"  Команды:")
    print(f"    /temp 0.5    — изменить температуру")
    print(f"    /len 200     — изменить длину")
    print(f"    /quit        — выход")
    print(f"{'='*60}\n")

    temp = 0.7
    gen_len = 200

    while True:
        try:
            prompt = input("▶ ")
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not prompt:
            continue
        if prompt.strip() == "/quit":
            break
        if prompt.startswith("/temp"):
            try:
                temp = float(prompt.split()[1])
                print(f"  Температура: {temp}")
            except:
                print(f"  Текущая: {temp}")
            continue
        if prompt.startswith("/len"):
            try:
                gen_len = int(prompt.split()[1])
                print(f"  Длина: {gen_len}")
            except:
                print(f"  Текущая: {gen_len}")
            continue

        t0 = time.time()
        text = generate(model, prompt.encode('utf-8'),
                       length=gen_len, temp=temp)
        elapsed = time.time() - t0

        # Показываем продолжение
        continuation = text[len(prompt):]
        print(f"{continuation}")
        tps = gen_len / max(elapsed, 0.001)
        print(f"  [{elapsed:.1f}с | {tps:.0f} ток/с | temp={temp}]\n")


# ============================================================
# MAIN
# ============================================================
def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False


def main():
    if is_notebook():
        # Colab / Jupyter
        print("🔬 Режим Colab/Jupyter\n")
        model, history = train()
        return model, history
    else:
        # Терминал
        import argparse
        parser = argparse.ArgumentParser(
            description="Mamba-2 Multilingual Trainer")
        parser.add_argument("mode", nargs="?", default="train",
                           choices=["train", "finetune", "interactive",
                                    "generate", "export"])
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--data", type=str, default="data")
        parser.add_argument("--model", type=str, default="mamba2_multi.pt")
        parser.add_argument("--prompt", type=str, default="Привет ")
        parser.add_argument("--length", type=int, default=200)
        parser.add_argument("--temp", type=float, default=0.7)
        args = parser.parse_args()

        if args.mode == "train":
            train(epochs=args.epochs, lr=args.lr,
                  data_dir=args.data, resume=args.resume)

        elif args.mode == "finetune":
            finetune(base_model=args.model, data_dir=args.data,
                    epochs=args.epochs or 5,
                    lr=args.lr or 5e-5)

        elif args.mode == "interactive":
            interactive(model_path=args.model)

        elif args.mode == "generate":
            cfg = get_config(MODEL_SIZE)
            model = Mamba2LM(**cfg)
            if os.path.exists(args.model):
                model.load_state_dict(
                    torch.load(args.model, map_location="cpu",
                               weights_only=True))
            model = model.to(DEVICE)
            text = generate(model, args.prompt.encode('utf-8'),
                           length=args.length, temp=args.temp)
            print(text)

        elif args.mode == "export":
            cfg = get_config(MODEL_SIZE)
            model = Mamba2LM(**cfg)
            if os.path.exists(args.model):
                model.load_state_dict(
                    torch.load(args.model, map_location="cpu",
                               weights_only=True))
            export_weights(model)


if __name__ == "__main__":
    main()