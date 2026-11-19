"""Offline streaming simulation; see docs/guides/streaming-speaker-similarity.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import soundfile as sf
import torch

from quick_convert.components.metrics.streaming_speaker import RollingSpeakerSimilarity, SpeakerSimilarityResult


def create_encoder(args):
    # Keep optional backends lazy and reject CPU-only CAMPPlus on CUDA explicitly.
    from quick_convert.components.encoders.speaker import (
        CosyVoiceCAMPPlusSpeakerEncoder,
        SpeechBrainECAPASpeakerEncoder,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if args.encoder == "ecapa" and torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, cuda, cuda:N, or auto.")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly selected but unavailable.")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable.")
        print(f"GPU: {torch.cuda.get_device_name(index)}", flush=True)
    if args.encoder == "campplus":
        if args.checkpoint is not None:
            raise ValueError("--checkpoint applies only to --encoder ecapa.")
        if selected.type != "cpu":
            raise ValueError("Existing CAMPPlus runs ONNX on CPU. Use --encoder ecapa --checkpoint PATH for CUDA.")
        if args.window_seconds > 10:
            raise ValueError("CAMPPlus crops windows over 10 seconds randomly; use <=10 seconds or another encoder.")
        encoder = CosyVoiceCAMPPlusSpeakerEncoder(device="cpu")
    else:
        if args.checkpoint is None:
            raise ValueError("ECAPA requires --checkpoint pointing to embedding_model.ckpt.")
        encoder = SpeechBrainECAPASpeakerEncoder(args.checkpoint, device=selected)
    encoder.eval()
    print(f"Encoder: {type(encoder).__name__}; inference device: {selected}", flush=True)
    return encoder


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("original", type=Path)
    result.add_argument("anonymized", type=Path)
    result.add_argument("output_dir", type=Path)
    alignment = result.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--aligned", action="store_true", help="Assert equal time scale and known constant offset.")
    alignment.add_argument("--alignment-unavailable", action="store_true", help="Report unavailable, without encoding.")
    result.add_argument("--output-start-seconds", type=float, default=0.0)
    result.add_argument("--window-seconds", type=float, default=4.0)
    result.add_argument("--hop-seconds", type=float, default=2.0)
    result.add_argument("--chunk-seconds", type=float, default=0.5)
    result.add_argument(
        "--max-buffer-seconds", type=float, help="Default covers window, chunk skew, and absolute offset."
    )
    result.add_argument("--encoder", choices=("campplus", "ecapa"), default="campplus")
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="auto")
    result.add_argument("--session-id", default="wav-simulation")
    return result


def main():
    args = parser().parse_args()
    if not math.isfinite(args.chunk_seconds) or args.chunk_seconds <= 0:
        raise ValueError("chunk-seconds must be finite and positive.")
    started = perf_counter()
    encoder = create_encoder(args)
    load_seconds = perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # SoundFile reads bounded chunks rather than loading entire recordings.
    with sf.SoundFile(args.original) as original, sf.SoundFile(args.anonymized) as anonymized:
        if original.channels != 1 or anonymized.channels != 1:
            raise ValueError("Both WAV inputs must be mono.")
        sizes = [max(1, round(args.chunk_seconds * stream.samplerate)) for stream in (original, anonymized)]
        core = RollingSpeakerSimilarity(
            encoder,
            original.samplerate,
            anonymized.samplerate,
            session_id=args.session_id,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            aligned=args.aligned,
            output_start_seconds=args.output_start_seconds,
            max_buffer_seconds=(
                args.max_buffer_seconds
                if args.max_buffer_seconds is not None
                else max(12.0, args.window_seconds + 2 * args.chunk_seconds + abs(args.output_start_seconds))
            ),
        )
        counts = {"ok": 0, "error": 0, "unavailable": 0}
        simulation_started = perf_counter()
        try:
            with (
                (args.output_dir / "metrics.jsonl").open("w") as jsonl,
                (args.output_dir / "metrics.csv").open("w", newline="") as csv_file,
            ):
                writer = csv.DictWriter(csv_file, fieldnames=[field.name for field in fields(SpeakerSimilarityResult)])
                writer.writeheader()

                def save(results):
                    for item in results:
                        record = item.to_dict()
                        jsonl.write(json.dumps(record, allow_nan=False) + "\n")
                        writer.writerow(record)
                        counts[item.status] += 1
                        print(
                            f"{item.window_index}: {item.window_start_seconds:.3f}–"
                            f"{item.window_end_seconds:.3f}s {item.status} "
                            f"score={item.speaker_similarity} compute={item.compute_time_ms:.2f}ms",
                            flush=True,
                        )

                ended = [False, False]
                while not all(ended):
                    for index, (stream, size, add) in enumerate(
                        zip((original, anonymized), sizes, (core.add_original, core.add_anonymized), strict=True)
                    ):
                        if ended[index]:
                            continue
                        chunk = stream.read(size, dtype="float32")
                        if not len(chunk):
                            ended[index] = True
                        else:
                            add(torch.from_numpy(chunk))
                        save(core.poll())
                core.finish()
                save(core.poll())
        finally:
            core.close()
        elapsed = perf_counter() - simulation_started
        timing = {
            "encoder_load_seconds": load_seconds,
            "simulation_seconds": elapsed,
            "total_seconds": perf_counter() - started,
            "status_counts": counts,
            "original_duration_seconds": original.frames / original.samplerate,
            "anonymized_duration_seconds": anonymized.frames / anonymized.samplerate,
            "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        (args.output_dir / "timing.json").write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n")
        print(f"Saved metrics and timing to {args.output_dir}; {counts}", flush=True)


if __name__ == "__main__":
    main()
