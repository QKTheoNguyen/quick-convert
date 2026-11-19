"""Live demo: Moonshine streaming ASR with PII masked by OpenAI Privacy Filter.

Speak English; the current line is rewritten as it is transcribed and each
finished line is kept, with personal information shown as ``[LABEL]``.
Requires the ``moonshine`` and ``privacy-filter`` extras.

Microphone on this machine:

    uv run --extra moonshine --extra privacy-filter python scripts/pii_live_demo.py

Microphone on another machine, sent as raw mono float32 PCM at 16 kHz on stdin
(for example from a Mac with SoX installed):

    rec -q -c 1 -r 16000 -b 32 -e floating-point -t raw - | ssh HOST \\
        'cd quick-convert && uv run --extra moonshine --extra privacy-filter python scripts/pii_live_demo.py --stdin'

Add ``--beep`` to also get the audio back 2.5 s late with personal information
beeped, as the same raw format on stdout (the masked text moves to stderr).
Detected words are timed by forced alignment with torchaudio's MMS_FA model:

    rec -q -c 1 -r 16000 -b 32 -e floating-point -t raw - | ssh HOST \\
        'cd quick-convert && uv run --extra moonshine --extra privacy-filter python scripts/pii_live_demo.py --stdin --beep' \\
        | play -q -c 1 -r 16000 -b 32 -e floating-point -t raw -

On a GPU the repo's pinned torch does not support (such as an RTX 50xx), build a
separate environment and pass ``--fa-device cuda``:

    uv venv .venv-gpu --python 3.11
    printf 'torch==2.7.1\\ntorchaudio==2.7.1\\nhydra-core==1.3.2\\nomegaconf==2.3.0\\n' > /tmp/gpu-overrides.txt
    uv pip install --python .venv-gpu/bin/python --override /tmp/gpu-overrides.txt \\
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \\
        -e ".[moonshine,privacy-filter]"
    .venv-gpu/bin/python scripts/pii_live_demo.py --stdin --beep --fa-device cuda
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import moonshine_voice
import numpy as np
from moonshine_voice.transcriber import LineCompleted, LineTextChanged

from quick_convert.components.alignment import MMSForcedAligner
from quick_convert.systems.pii import OpenAIPrivacyFilter, StreamingPIIBeeper, StreamingPIIMasker


STDIN_SAMPLE_RATE = 16_000
CHUNK_BYTES = STDIN_SAMPLE_RATE // 10 * 4  # 100 ms of float32 samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stdin", action="store_true", help="read 16 kHz mono float32 PCM from stdin")
    parser.add_argument("--beep", action="store_true", help="with --stdin, write delayed beeped audio to stdout")
    parser.add_argument("--fa-device", default="cpu", help="torch device of the --beep alignment model, e.g. cuda")
    args = parser.parse_args()
    if args.beep and not args.stdin:
        parser.error("--beep requires --stdin")
    chunks = read_stdin_in_background() if args.stdin else None

    audio_out = None
    if args.beep:
        # Keep the real stdout for audio only; library prints (model downloads) go to stderr.
        audio_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    text_out = sys.stderr if args.beep else sys.stdout

    detector = OpenAIPrivacyFilter()
    detector.detect_text(["warm up"])  # load the weights before listening
    masker = StreamingPIIMasker(detector)

    # shortcut: rewrites one terminal row, so a line wider than the terminal smears; use a UI for long lines.
    def show(masked: str, done: bool) -> None:
        text_out.write("\r\033[K" + masked + ("\n" if done else ""))
        text_out.flush()

    if args.beep:
        # Streamed Moonshine word times are unreliable in moonshine-voice 0.1.6, so lines are force-aligned.
        aligner = MMSForcedAligner(args.fa_device)
        aligner.align(np.zeros(STDIN_SAMPLE_RATE, dtype=np.float32), ["warm"])  # load the weights before listening
        beeper = StreamingPIIBeeper(masker, aligner.align, STDIN_SAMPLE_RATE)
        stream_stdin(
            chunks, lambda line, done: show(beeper.update(line.text, line.start_time, done), done), beeper, audio_out
        )
    elif args.stdin:
        stream_stdin(chunks, lambda line, done: show(masker.mask(line.text, done), done))
    else:
        listen_microphone(lambda text, done: show(masker.mask(text, done), done))


def listen_microphone(show) -> None:
    mic = (
        moonshine_voice.MicTranscriber()
        .language("en")
        .on_text(lambda text: show(text, done=False))
        .on_line(lambda line: show(line.text, done=True))
    )
    mic.load()
    mic.start()
    print("Listening. Speak English, Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()


def read_stdin_in_background() -> queue.Queue:
    """Queue 100 ms stdin chunks from startup on; ``None`` marks the end of input.

    The sender (``rec``) records while the models load. Reading from the start
    lets that backlog be dropped; otherwise a real-time player hears everything
    late by the loading time, for the whole session.
    """
    chunks = queue.Queue()

    def read() -> None:
        while chunk := sys.stdin.buffer.read(CHUNK_BYTES):
            chunks.put(chunk)
        chunks.put(None)

    threading.Thread(target=read, daemon=True).start()
    return chunks


def drop_backlog(chunks: queue.Queue) -> float:
    """Discard the queued chunks; return their duration in seconds."""
    dropped = 0
    while True:
        try:
            chunk = chunks.get_nowait()
        except queue.Empty:
            return dropped / 4 / STDIN_SAMPLE_RATE
        if chunk is None:
            chunks.put(None)
            return dropped / 4 / STDIN_SAMPLE_RATE
        dropped += len(chunk)


def stream_stdin(chunks: queue.Queue, on_line, beeper: StreamingPIIBeeper | None = None, audio_out=None) -> None:
    model_path, model_arch = moonshine_voice.get_model_for_language("en")
    stream = moonshine_voice.Transcriber(model_path=model_path, model_arch=model_arch).create_stream()

    def listener(event) -> None:
        if isinstance(event, LineTextChanged | LineCompleted):
            on_line(event.line, done=isinstance(event, LineCompleted))

    def release(final: bool = False) -> None:
        if beeper is not None:
            audio_out.write(beeper.read(final).astype("<f4").tobytes())
            audio_out.flush()

    stream.add_listener(listener)
    stream.start()
    skipped = drop_backlog(chunks)
    print(
        f"Listening on stdin (skipped {skipped:.1f} s recorded while loading). Speak English, Ctrl+C to stop.",
        file=sys.stderr,
        flush=True,
    )
    try:
        while (chunk := chunks.get()) is not None:
            usable = len(chunk) - len(chunk) % 4
            samples = np.frombuffer(chunk[:usable], dtype="<f4")
            if beeper is not None:
                beeper.write(samples)
            # Line updates (and their beeps) run inside add_audio, before the delayed audio is released.
            stream.add_audio(samples.tolist(), STDIN_SAMPLE_RATE)
            release()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        stream.stop()
        try:
            release(final=True)
        except BrokenPipeError:
            pass


if __name__ == "__main__":
    main()
