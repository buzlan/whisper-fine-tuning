import argparse

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and cache a Whisper model + processor locally.")
    ap.add_argument("--model", type=str, default="openai/whisper-small", help="HF model id, e.g. openai/whisper-small")
    args = ap.parse_args()

    print(f"Downloading processor: {args.model}")
    WhisperProcessor.from_pretrained(args.model)
    print(f"Downloading model: {args.model}")
    WhisperForConditionalGeneration.from_pretrained(args.model)
    print(f"Done. Model cached: {args.model}")


if __name__ == "__main__":
    main()
