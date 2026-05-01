# SSM_LLM — Mamba-2 sketch

Реализация архитектуры Mamba-2 (State Space Model) — обучение, инференс на C, эксперименты с предобученными моделями.

## Что здесь

| Файл                    | Описание                                                                         |
| ----------------------- | -------------------------------------------------------------------------------- |
| `mamba2_multi_train.py` | Обучение Mamba-2 с нуля на русском + английском + коде. Triton-ядра для SSM scan |
| `mamba2.c`              | C-инференс для нашей обученной модели (byte-level, 256 vocab)                    |
| `mamba2_radical.py`     | Упрощённая версия обучалки для экспериментов                                     |
| `mamba2_27b.c`          | C-инференс для предобученной mamba2-2.7b (WIP)                                   |
| `chat_mamba28b.py`      | Чат через HuggingFace transformers + state-spaces/mamba-2.8b                     |
| `profiler.py`           | Профилирование скорости обучения                                                 |
| `SECURITY.md`           | ⚠️ Руководство по безопасности — не коммитьте секреты!                          |
| `.env.example`          | Шаблон конфигурации (скопируйте в `.env` и отредактируйте)                      |
| `data/`                 | Обучающие тексты (Толстой, диалоги, английские тексты)                           |

## Быстрый старт

### Требования

- Python 3.10+, PyTorch 2.0+, CUDA GPU (GTX 1650 Ti хватает)
- Triton (`pip install triton`)
- GCC для C-инференса
- python-dotenv (`pip install python-dotenv`)

`pip install torch triton python-dotenv`

### Конфигурация

**⚠️ ВАЖНО**: Скопируйте `.env.example` в `.env` и настройте параметры:

```bash
cp .env.example .env
```

Отредактируйте `.env` под вашу систему. **Никогда не коммитьте `.env` в репозиторий!**

Подробнее см. [SECURITY.md](SECURITY.md) — руководство по безопасности.

### 1. Обучение с нуля

#### Положить тексты в data/ (.txt файлы, UTF-8)

#### Обучение ~30-40 мин на GTX 1650 Ti

`python mamba2_multi_train.py train`

#### Дообучение на диалогах (~10-15 мин)

`python mamba2_multi_train.py finetune`

#### Продолжить с чекпоинта

`python mamba2_multi_train.py train --resume mamba2_multi.pt`

### 2. Генерация в Python

`python mamba2_multi_train.py interactive`

### 3. C-инференс (быстрая генерация без Python)

#### Экспорт весов

`python mamba2_multi_train.py export --model mamba2_multi.pt`

#### Компиляция

`gcc -O3 -march=native -o mamba2 mamba2.c -lm`

#### Генерация

`./mamba2 mamba2_multi_weights.bin "Россия" 200 0.7`
`./mamba2 mamba2_multi_weights.bin "def sort(" 150 0.5`
`./mamba2 mamba2_multi_weights.bin "The " 200 0.7`

#### Интерактивный чат

`./mamba2 mamba2_multi_weights.bin --chat`
Команды в чате: `/temp 0.5`, `/len 200`, `/reset`, `/quit`

### 4. Готовая большая модель (HuggingFace)

`pip install transformers`
`python chat_mamba28b.py`
Загружает state-spaces/mamba-2.8b-hf, ~5 GB VRAM.  

#### Архитектура  
Наша модель — Mamba-2 с byte-level токенизацией:  

Параметр Значение  
vocab_size 256 (побайтовый)  
d_model 256  
n_layers 8 (4×SSM + 4×FFN)  
d_state 16  
d_inner 512  
Параметров ~3M

#### Ключевые особенности:  

SSM scan через Triton-ядра (параллельный prefix-sum на GPU)  
Byte-level — никакого токенизатора, работает с любым языком  
C-инференс — fp32, ~200 tok/s на CPU для нашей модели  
Чередование SSM + FFN слоёв  

#### Структура проекта  

SSM_LLM/  
├── mamba2_multi_train.py # Основная обучалка  
├── mamba2.c # C инференс (наша модель)  
├── mamba2_radical.py # Эксперименты  
├── mamba2_27b.c # C инференс (2.7B, WIP)  
├── chat_mamba28b.py # HF чат с большой моделью  
├── profiler.py # Бенчмарки  
├── SECURITY.md # 🔐 Руководство по безопасности  
├── .env.example # Шаблон конфигурации  
├── .gitignore # Исключения из git  
├── README.md # Этот файл  
├── F.A.Q. # Шпаргалка по командам  
├── data/ # Обучающие тексты  
│ ├── WAP1-4.txt # Война и мир  
│ ├── english.txt # Английские тексты  
│ ├── dialogs.txt # Диалоги для файнтюна  
│ └── RAL.txt # Дополнительные тексты  
└── BC/ # Бэкапы и тесты  

#### TODO  
C-инференс для mamba2-2.7b с SIMD-оптимизацией  
BPE токенизатор для больших моделей  
Квантизация весов (int8) README  
