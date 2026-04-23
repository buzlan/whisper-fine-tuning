import argparse
import csv
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass(frozen=True)
class Row:
    audio: str
    text: str
    uid: str


def _iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            yield p


def _read_text_sidecar(audio_path: Path, text_ext: str) -> Optional[str]:
    txt = audio_path.with_suffix(text_ext)
    if not txt.exists():
        return None
    return txt.read_text(encoding="utf-8").strip()


def _safe_uid(p: Path, base: Path) -> str:
    rel = p.relative_to(base).as_posix()
    rel = rel.replace("/", "__")
    return rel


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a local ASR dataset manifest (CSV/JSONL) for Whisper fine-tuning.\n\n"
            "Expected input layout (default):\n"
            "  <audio>.wav (or mp3/flac/m4a/ogg)\n"
            "  <audio>.txt  (same basename, transcription)\n"
        )
    )
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with audio files (+ optional .txt sidecars).")
    ap.add_argument("--text_ext", type=str, default=".txt", help="Sidecar transcription extension (default: .txt).")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write manifests.")
    ap.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Column name used by training script for audio file path (default: audio).",
    )
    ap.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name used by training script for transcript (default: text).",
    )
    ap.add_argument(
        "--relative_paths",
        action="store_true",
        help="Store audio paths relative to out_dir (helps portability).",
    )
    ap.add_argument(
        "--exclude_audio_regex",
        type=str,
        default=None,
        help="Optional regex; if it matches the audio path, the sample is skipped.",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train fraction in split (default 0.9). Eval gets the rest.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    missing_txt: list[str] = []
    exclude_re = re.compile(args.exclude_audio_regex) if args.exclude_audio_regex else None

    for audio in sorted(_iter_audio_files(input_dir)):
        if exclude_re and exclude_re.search(str(audio)):
            continue
        text = _read_text_sidecar(audio, args.text_ext)
        if not text:
            missing_txt.append(str(audio))
            continue

        audio_path = audio
        if args.relative_paths:
            audio_path = Path(os.path.relpath(audio, out_dir))

        rows.append(
            Row(
                audio=str(audio_path),
                text=text,
                uid=_safe_uid(audio, input_dir),
            )
        )

    if not rows:
        raise SystemExit(
            "No usable items found. Make sure you have audio files and matching sidecar transcriptions."
        )

    # train/eval split: simple deterministic split by index
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit(f"--train_ratio must be in (0,1). Got: {args.train_ratio}")
    split_idx = max(1, int(len(rows) * args.train_ratio))
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:] if len(rows) > 1 else rows[:1]

    def write_csv(path: Path, items: list[Row]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[args.audio_column, args.text_column, "uid"])
            w.writeheader()
            for r in items:
                w.writerow({args.audio_column: r.audio, args.text_column: r.text, "uid": r.uid})

    def write_jsonl(path: Path, items: list[Row]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(
                    json.dumps({args.audio_column: r.audio, args.text_column: r.text, "uid": r.uid}, ensure_ascii=False)
                    + "\n"
                )

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "eval.csv", eval_rows)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "items_total": len(rows),
        "items_train": len(train_rows),
        "items_eval": len(eval_rows),
        "missing_sidecar_txt": missing_txt[:50],
        "missing_sidecar_txt_count": len(missing_txt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote manifests to: {out_dir}")
    print(f"Total items: {len(rows)} (train={len(train_rows)}, eval={len(eval_rows)})")
    if missing_txt:
        print(f"WARNING: {len(missing_txt)} audio files had no matching {args.text_ext} and were skipped.")


if __name__ == "__main__":
    main()
