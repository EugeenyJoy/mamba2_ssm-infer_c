# Безопасность

## ⚠️ Критичное: Не коммитьте

- **API ключи** и токены доступа
- **Пути до домашних папок** (например: `/home/username/`, `/Users/name/`)
- **SSH приватные ключи** (`.key`, `.pem` файлы)
- **Переменные окружения с секретами** (`.env` файлы)
- **Логины и пароли**
- **Конфиденциальные данные обучения** (если содержат реальные данные)

## Правильная практика

### 1. Используйте переменные окружения

**Вместо этого (❌ ОПАСНО):**

```python
cache_dir = '/home/jqw/.hf_cache'
api_key = 'sk-1234567890abcdef'
```

**Сделайте так (✅ БЕЗОПАСНО):**

```python
import os
from dotenv import load_dotenv

load_dotenv()
cache_dir = os.path.expanduser('~/.cache/huggingface')
api_key = os.getenv('API_KEY')
```

### 2. Используйте `.env.example` вместо `.env`

- Коммитьте только **`.env.example`** с описаниями переменных
- **Никогда** не коммитьте `.env` (содержит реальные значения)
- Добавьте в `.gitignore`: `.env`, `.env.local`, `secrets/`

### 3. Структурируйте пути правильно

```python
import os

# ✅ ХОРОШО: Расширяется до полного пути пользователя
model_cache = os.path.expanduser('~/.cache/huggingface')
project_data = os.path.expanduser('~/projects/data')

# ✅ ХОРОШО: Использует переменные окружения
cache_dir = os.getenv('HF_CACHE_DIR', os.path.expanduser('~/.cache/huggingface'))

# ❌ ПЛОХО: Жесткий путь露выляет имя пользователя
model_cache = '/home/jqw/.hf_cache'
```

## Если данные уже скомичены

Если вы случайно закоммитили секреты:

### 1. Немедленно удалите из истории Git

```bash
# Просмотрите историю коммитов файла
git log --all --full-history -- "path/to/file"

# Удалите файл из всей истории
git filter-branch --tree-filter 'rm -f path/to/file' -- --all

# Принудительно отправьте обновленную историю
git push --force-with-lease origin main
```

### 2. Немедленно ротируйте секреты

- Перегенерируйте API ключи, токены, пароли
- Проверьте логи доступа на предмет несанкционированного доступа
- Уведомите администраторов безопасности, если требуется

### 3. Найдите другие случаи

```bash
# Найдите другие вхождения чувствительных данных
grep -r "/home/" . --include="*.py"
grep -r "token\|key\|secret\|password" . --include="*.py"
grep -r "\.hf_cache\|\.env\|config_local" . --include="*.py"
```

## Проверка перед коммитом

Перед коммитом выполните:

```bash
# Посмотрите, какие файлы вы добавляете
git diff --cached

# Проверьте чувствительные строки
git diff --cached | grep -i "token\|key\|secret\|password\|/home/"
```

## Инструменты для помощи

- **pre-commit** - автоматическая проверка перед коммитом
- **git-secrets** - обнаружение случайных коммитов секретов
- **detect-secrets** - поиск потенциальных секретов в коде

Пример `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

## Мониторинг в GitHub

Если используете GitHub:

- ✅ Включите "Secret scanning" в Settings → Security
- ✅ Включите "Push protection" для блокирования коммитов с секретами
- ✅ Регулярно проверяйте "Security" tab
