import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run transcription with base model openai/whisper-small."
    )
    ap.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu/cuda/mps")
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--chunk_length_s", type=float, default=20.0)
    ap.add_argument("--chunk_overlap_s", type=float, default=2.0)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    transcribe_script = project_root / "scripts" / "transcribe_with_finetuned.py"

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    cmd = [
        sys.executable,
        str(transcribe_script),
        "--model",
        "openai/whisper-small",
        "--audio",
        str(audio_path),
        "--language",
        args.language,
        "--task",
        args.task,
        "--device",
        args.device,
        "--chunk_length_s",
        str(args.chunk_length_s),
        "--chunk_overlap_s",
        str(args.chunk_overlap_s),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
