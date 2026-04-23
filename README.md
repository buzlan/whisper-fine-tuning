# Whisper small fine-tuning (готовый каркас)

В этом проекте уже подготовлены скрипты, чтобы тебе осталось только:
- собрать **локальный датасет** (аудио + транскрипт),
- запустить **подготовку манифестов**,
- запустить **fine-tuning Whisper small** (по умолчанию через **LoRA**).

Ниже — пошагово.

## 0) Что здесь уже есть

- `scripts/download_whisper_model.py`: скачивание/прогрев кэша модели.
- `scripts/prepare_local_dataset.py`: сбор локального датасета в `train.csv/eval.csv`.
- `scripts/train_whisper_small_lora.py`: обучение `openai/whisper-small` на своих данных.
- `scripts/train_whisper_small_full.py`: полный fine-tune `openai/whisper-small` (обновляются все веса, требует больше VRAM).

## 1) Установить зависимости

Рекомендуется отдельное окружение (можно использовать уже существующее, если оно ок).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

## 2) Скачать small-модель (один раз)

```bash
python scripts/download_whisper_model.py --model openai/whisper-small
```

## 3) Подготовить датасет (только ты заполняешь)

Самый простой формат входа:

```
datasets/my_asr/raw/
  sample_0001.wav
  sample_0001.txt
  sample_0002.wav
  sample_0002.txt
  ...
```

Требования:
- аудио лучше держать **моно 16kHz** (если нет — скрипт/`datasets` обычно ресемплит, но лучше заранее).
- `.txt` рядом — **точная расшифровка** (без таймкодов).
- язык/орфография — как ты хочешь, чтобы модель писала после обучения.

## 4) Сделать манифесты train/eval

Скрипт пройдётся по папке, возьмёт пары `audio + .txt` и создаст:
- `train.csv`, `eval.csv`
- `train.jsonl`, `eval.jsonl`
- `summary.json`

```bash
python scripts/prepare_local_dataset.py \
  --input_dir datasets/my_asr/raw \
  --out_dir datasets/my_asr/prepared
```

Формат `train.csv` по умолчанию:
- `audio`: путь к файлу
- `text`: транскрипт
- `uid`: идентификатор

## 5) Запустить fine-tuning (Whisper small + LoRA)

Минимальная команда:

```bash
python scripts/train_whisper_small_lora.py \
  --train_csv datasets/my_asr/prepared/train.csv \
  --eval_csv datasets/my_asr/prepared/eval.csv \
  --model openai/whisper-small \
  --language en \
  --task transcribe \
  --output_dir outputs/whisper-small-lora \
  --use_lora \
  --predict_with_generate
```

Полезные параметры:
- `--num_train_epochs`: сколько эпох (начни с 1–3).
- `--per_device_train_batch_size` и `--gradient_accumulation_steps`: подстраивай под память.
- `--learning_rate`: для LoRA часто ок \(1e-4..3e-4\), для полного fine-tune обычно ниже.
- `--fp16`/`--bf16`: включай только если твой девайс это поддерживает (CUDA/некоторые ускорители).

Выход:
- в `outputs/whisper-small-lora/` будет сохранена модель и процессор (токенайзер/фичи).

## 5b) Полный fine-tune (без LoRA)

Полный fine-tune обычно требует **больше памяти** и **меньший learning rate**. Пример:

```bash
python scripts/train_whisper_small_full.py \
  --train_csv datasets/my_asr/prepared/train.csv \
  --eval_csv datasets/my_asr/prepared/eval.csv \
  --model openai/whisper-small \
  --language en \
  --task transcribe \
  --output_dir outputs/whisper-small-full \
  --predict_with_generate \
  --gradient_checkpointing
```

## 6) Что тебе осталось сделать (чеклист)

- собрать `datasets/<name>/raw/*.wav + *.txt`
- прогнать `scripts/prepare_local_dataset.py`
- запустить `scripts/train_whisper_small_lora.py`

Если хочешь — могу дальше помочь: подсказать оптимальные гиперпараметры под твой объём данных/железо и сделать нормальный eval (WER по доменам, sanity-check на ручных примерах).
