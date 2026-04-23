import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe audio with Whisper base model + LoRA adapter checkpoint.")
    ap.add_argument("--base_model", type=str, default="openai/whisper-small")
    ap.add_argument("--adapter", type=str, required=True, help="Path to LoRA checkpoint directory.")
    ap.add_argument("--audio", type=str, required=True, help="Path to audio file.")
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu/cuda/mps")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--chunk_length_s", type=float, default=20.0)
    ap.add_argument("--chunk_overlap_s", type=float, default=2.0)
    ap.add_argument("--out_text", type=str, default=None)
    args = ap.parse_args()

    adapter_path = Path(args.adapter).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")
    if not audio_path.exists():
        raise SystemExit(f"Audio path not found: {audio_path}")

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    audio_array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunk_len = int(args.chunk_length_s * sr)
    overlap = int(args.chunk_overlap_s * sr)
    step = max(1, chunk_len - overlap)
    max_target_positions = getattr(model.config, "max_target_positions", 448)
    max_new_tokens = min(args.max_new_tokens, max_target_positions - 4)

    texts: list[str] = []
    for start in range(0, len(audio_array), step):
        end = min(len(audio_array), start + chunk_len)
        chunk = audio_array[start:end]
        if len(chunk) < 1:
            continue

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            generated_ids = model.generate(input_features, max_new_tokens=max_new_tokens)

        part = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if part:
            texts.append(part)

    final_text = " ".join(texts).strip()
    if args.out_text:
        out_path = Path(args.out_text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_text + "\n", encoding="utf-8")
        print(f"Saved transcript to: {out_path}")
    else:
        print(final_text)


if __name__ == "__main__":
    main()
