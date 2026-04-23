# Whisper small fine-tuning (frisbie/frisbies)

Минимальный рабочий каркас для обучения `openai/whisper-small` под канонические слова
`frisbie` и `frisbies` через LoRA.

## 1) Что есть в проекте

- `scripts/download_whisper_model.py`: скачивание/прогрев кэша модели.
- `scripts/prepare_local_dataset.py`: сбор локального датасета в `train.csv/eval.csv`.
- `scripts/train_whisper_small_lora.py`: обучение `openai/whisper-small` на своих данных.
- `scripts/run_base_model.py`: запуск базовой small-модели.
- `scripts/transcribe_with_lora_adapter.py`: запуск дообученного LoRA-чекпоинта поверх base-модели.

## 2) Установить зависимости

Рекомендуется отдельное окружение (можно использовать уже существующее, если оно ок).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

## 3) Скачать small-модель (один раз)

```bash
python scripts/download_whisper_model.py --model openai/whisper-small
```

## 4) Подготовить датасет

Самый простой формат входа:

```
datasets/keywords/raw/
  record1.wav
  record1.txt
  record2.wav
  record2.txt
  ...
```

Требования:
- аудио лучше держать **моно 16kHz** (если нет — скрипт/`datasets` обычно ресемплит, но лучше заранее).
- `.txt` рядом — **точная расшифровка** (без таймкодов).
- язык/орфография — как ты хочешь, чтобы модель писала после обучения.

## 5) Сделать манифесты train/eval

Скрипт пройдётся по папке, возьмёт пары `audio + .txt` и создаст:
- `train.csv`, `eval.csv`
- `train.jsonl`, `eval.jsonl`
- `summary.json`

```bash
python scripts/prepare_local_dataset.py \
  --input_dir datasets/keywords/raw \
  --out_dir datasets/keywords/prepared
```

Формат `train.csv` по умолчанию:
- `audio`: путь к файлу
- `text`: транскрипт
- `uid`: идентификатор

## 6) Запустить fine-tuning (Whisper small + LoRA)

Минимальная команда:

```bash
python scripts/train_whisper_small_lora.py \
  --train_csv datasets/keywords/focused_frisbie_only/train.csv \
  --eval_csv datasets/keywords/focused_frisbie_only/eval.csv \
  --audio_column audio_path \
  --text_column text \
  --model openai/whisper-small \
  --language en \
  --task transcribe \
  --output_dir outputs/whisper-small-focused-frisbie-only \
  --use_lora \
  --predict_with_generate \
  --max_steps 120 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.5e-4 \
  --eval_steps 10 \
  --save_steps 10 \
  --logging_steps 5 \
  --load_best_model_at_end \
  --metric_for_best_model wer \
  --save_total_limit 5 \
  --num_proc 1
```

## 7) Сравнить base vs дообученную модель

Base:

```bash
python scripts/run_base_model.py \
  --audio "/Users/ihar/whisper-local/3272511_11459695_530_1.wav" \
  --device cpu \
  --language en \
  --task transcribe
```

Fine-tuned (лучший чекпоинт):

```bash
python scripts/transcribe_with_lora_adapter.py \
  --base_model openai/whisper-small \
  --adapter outputs/whisper-small-focused-frisbie-only/checkpoint-90 \
  --audio "/Users/ihar/whisper-local/3272511_11459695_530_1.wav" \
  --language en \
  --task transcribe \
  --device cpu
```

## 8) Мини-чеклист

- положить пары `recordN.wav + recordN.txt` в `datasets/keywords/raw`
- собрать манифесты через `scripts/prepare_local_dataset.py`
- обучить LoRA через `scripts/train_whisper_small_lora.py`
- тестировать через `scripts/transcribe_with_lora_adapter.py`
