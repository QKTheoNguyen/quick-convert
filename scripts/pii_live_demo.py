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
"""

from __future__ import annotations

import argparse
import sys
import time

import moonshine_voice
import numpy as np
from moonshine_voice.transcriber import LineCompleted, LineTextChanged

from quick_convert.systems.pii import OpenAIPrivacyFilter, StreamingPIIMasker


STDIN_SAMPLE_RATE = 16_000
CHUNK_BYTES = STDIN_SAMPLE_RATE // 10 * 4  # 100 ms of float32 samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stdin", action="store_true", help="read 16 kHz mono float32 PCM from stdin")
    args = parser.parse_args()

    detector = OpenAIPrivacyFilter()
    detector.detect(["warm up"])  # load the weights before listening
    masker = StreamingPIIMasker(detector)

    # shortcut: rewrites one terminal row, so a line wider than the terminal smears; use a UI for long lines.
    def show(text: str, done: bool) -> None:
        sys.stdout.write("\r\033[K" + masker.mask(text, done) + ("\n" if done else ""))
        sys.stdout.flush()

    if args.stdin:
        stream_stdin(show)
    else:
        listen_microphone(show)


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


def stream_stdin(show) -> None:
    model_path, model_arch = moonshine_voice.get_model_for_language("en")
    stream = moonshine_voice.Transcriber(model_path=model_path, model_arch=model_arch).create_stream()

    def listener(event) -> None:
        if isinstance(event, LineTextChanged | LineCompleted):
            show(event.line.text, done=isinstance(event, LineCompleted))

    stream.add_listener(listener)
    stream.start()
    print("Listening on stdin. Speak English, Ctrl+C to stop.", flush=True)
    try:
        while chunk := sys.stdin.buffer.read(CHUNK_BYTES):
            usable = len(chunk) - len(chunk) % 4
            stream.add_audio(np.frombuffer(chunk[:usable], dtype="<f4").tolist(), STDIN_SAMPLE_RATE)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
