import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS is present, drop it (Whisper adds decoder_start_token_id itself)
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _as_int(x: Union[str, int]) -> int:
    return int(x) if isinstance(x, str) else x


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Whisper small on a local CSV dataset (LoRA by default).")

    # Data
    ap.add_argument("--train_csv", type=str, required=True, help="Path to train.csv (columns: audio,text).")
    ap.add_argument("--eval_csv", type=str, required=True, help="Path to eval.csv (columns: audio,text).")
    ap.add_argument("--audio_column", type=str, default="audio")
    ap.add_argument("--text_column", type=str, default="text")
    ap.add_argument("--max_label_length", type=int, default=448, help="Max tokens for transcript (truncate longer).")

    # Model / task
    ap.add_argument("--model", type=str, default="openai/whisper-small")
    ap.add_argument("--language", type=str, default="en", help="Task language (e.g. en, ru).")
    ap.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])

    # Training
    ap.add_argument("--output_dir", type=str, default="outputs/whisper-small-lora")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=250)
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio [0,1]. If >0, overrides warmup_steps.")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, overrides num_train_epochs and trains for exact number of update steps.",
    )
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_mps_device", action="store_true", help="Try to use Apple MPS for training (if available).")
    ap.add_argument("--num_proc", type=int, default=1, help="Dataset preprocessing processes (default 1 for stability).")
    ap.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Custom temp directory. Useful when system temp is unavailable.",
    )

    # LoRA
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA (recommended).")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Generation / eval
    ap.add_argument("--predict_with_generate", action="store_true", help="Generate texts for eval (needed for WER).")
    ap.add_argument("--generation_max_length", type=int, default=225)
    ap.add_argument("--load_best_model_at_end", action="store_true", help="Load best checkpoint at the end.")
    ap.add_argument("--metric_for_best_model", type=str, default="wer", help="Metric to select best model (e.g. wer).")
    ap.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Set if higher metric is better. For WER keep this disabled.",
    )
    ap.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Enable early stopping if >0 (number of evals without improvement).",
    )
    ap.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard"],
        help="Reporting backend. Use tensorboard to watch WER/loss during training.",
    )

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        os.environ["TMPDIR"] = args.temp_dir
        os.environ["TEMP"] = args.temp_dir
        os.environ["TMP"] = args.temp_dir
        tempfile.tempdir = args.temp_dir

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Make generation consistent with chosen task/language
    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    except Exception:
        # Older/newer transformers may differ; generation still works without forced ids.
        pass

    model.config.suppress_tokens = []
    model.config.use_cache = False

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_files = {"train": args.train_csv, "eval": args.eval_csv}
    ds = load_dataset("csv", data_files=data_files)
    ds = DatasetDict(train=ds["train"], eval=ds["eval"])

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = batch[args.audio_column]
        text = batch[args.text_column]

        if text is None:
            text = ""
        text = str(text).strip()

        audio_array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        inputs = processor.feature_extractor(audio_array, sampling_rate=sr)
        batch["input_features"] = inputs["input_features"][0]

        labels = processor.tokenizer(
            text,
            max_length=args.max_label_length,
            truncation=True,
        ).input_ids
        batch["labels"] = labels
        return batch

    ds = ds.map(
        preprocess,
        remove_columns=[c for c in ds["train"].column_names if c not in ("uid",)],
        num_proc=max(1, args.num_proc),
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=_as_int(model.config.decoder_start_token_id),
    )

    wer = evaluate.load("wer")

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_value = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer_value)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        optim="adamw_torch",
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
        use_mps_device=args.use_mps_device,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Saved model + processor to: {args.output_dir}")
    if trainer.state.best_model_checkpoint:
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    if trainer.state.best_metric is not None:
        print(f"Best metric ({args.metric_for_best_model}): {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
