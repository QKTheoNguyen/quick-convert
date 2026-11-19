from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from .base import PIIDetector, mask_text, timed_spans


class StreamingPIIMasker:
    """Mask a live transcript line by line, with recent finished words as context.

    The detector sees the last ``context_words`` words of finished lines
    followed by the current line, so an entity split across a pause keeps its
    cue (``"call me at" | "555 0132"``). Only the current line is masked; a span
    that starts in the context is clipped to the line.

    Args:
        detector: System that finds personal information in text.
        context_words: Number of previous finished words given as left context.
    """

    def __init__(self, detector: PIIDetector, context_words: int = 100):
        self.detector = detector
        self._context = deque(maxlen=context_words)

    def spans(self, text: str) -> list[dict]:
        """Return the spans of ``text`` found with the current context, in ``text`` offsets."""
        if not text.strip():
            return []
        prefix = " ".join(self._context)
        prefix = f"{prefix} " if prefix else ""
        return [
            {**span, "start": max(span["start"] - len(prefix), 0), "end": span["end"] - len(prefix)}
            for span in self.detector.detect_text([prefix + text])[0]
            if span["end"] > len(prefix)
        ]

    def add_context(self, text: str) -> None:
        """Add a finished line to the context of later lines."""
        self._context.extend(text.split())

    def mask(self, text: str, done: bool = False) -> str:
        """Mask the current line; ``done`` adds it to the context of later lines."""
        spans = self.spans(text)
        if done:
            self.add_context(text)
        return mask_text(text, spans)


class StreamingPIIBeeper:
    """Delay live audio and replace detected personal information with a beep.

    Audio given to :meth:`write` is released by :meth:`read` ``delay`` seconds
    of stream time later, so personal information found in a line update before
    then is beeped. Each update is checked on its streamed text; when the line's
    findings change, its words are force-aligned to the line's audio with
    ``align`` and each span's words are beeped, padded by ``pad_before`` and
    ``pad_after``. Beeped ranges are never withdrawn. When alignment fails, the
    whole line so far is beeped.

    Args:
        masker: Masker providing context-aware detection and the masked text.
        align: Maps a mono waveform at ``sample_rate`` and the line's words to
            one ``{"word", "start_s", "end_s"}`` entry per word, with times in
            seconds from the waveform start, or ``None`` when they cannot be
            aligned.
        sample_rate: Sample rate of the written audio.
        delay: Seconds of stream time between writing and releasing audio.
        pad_before: Seconds beeped before each detected range.
        pad_after: Seconds beeped after each detected range.
        beep_hz: Beep frequency.
        keep_seconds: Audio history kept for realigning long lines.
    """

    def __init__(
        self,
        masker: StreamingPIIMasker,
        align: Callable[[np.ndarray, list[str]], list[dict] | None],
        sample_rate: int = 16_000,
        delay: float = 2.5,
        pad_before: float = 0.1,
        pad_after: float = 0.1,
        beep_hz: float = 1000.0,
        keep_seconds: float = 60.0,
    ):
        self.masker = masker
        self.align = align
        self.sample_rate = sample_rate
        self.delay = delay
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.beep_hz = beep_hz
        self.keep_seconds = keep_seconds
        self._audio = np.zeros(0, dtype=np.float32)
        self._offset = 0  # absolute sample index of self._audio[0]
        self._released = 0
        self._last_found = None
        self.beeps: list[tuple[int, int]] = []  # absolute sample ranges

    @property
    def written(self) -> int:
        return self._offset + len(self._audio)

    def write(self, samples: np.ndarray) -> None:
        """Append mono samples to the stream."""
        self._audio = np.concatenate([self._audio, np.asarray(samples, dtype=np.float32)])

    def update(self, text: str, start_time: float, done: bool = False) -> str:
        """Check a line update starting at ``start_time`` seconds; return its masked text."""
        text = " ".join(text.split())
        spans = self.masker.spans(text)
        # Alignment costs more than detection, so repeat it only when the line's findings change.
        found_now = (start_time, frozenset((span["label"], text[span["start"] : span["end"]]) for span in spans))
        changed, self._last_found = found_now != self._last_found, found_now
        if spans and changed:
            start = max(int(start_time * self.sample_rate), self._offset)
            words = self.align(self._audio[start - self._offset :], text.split())
            base = start / self.sample_rate
            if words is None:
                self._beep(base, self.written / self.sample_rate)
            else:
                for span in timed_spans(spans, words):
                    self._beep(base + span["start_s"], base + span["end_s"])
        if done:
            self.masker.add_context(text)
        return mask_text(text, spans)

    def _beep(self, start: float, end: float) -> None:
        self.beeps.append(
            (int((start - self.pad_before) * self.sample_rate), int((end + self.pad_after) * self.sample_rate))
        )

    def read(self, final: bool = False) -> np.ndarray:
        """Release the audio older than ``delay``, or all of it when ``final``, with beeps applied."""
        target = self.written if final else self.written - int(self.delay * self.sample_rate)
        if target <= self._released:
            return np.zeros(0, dtype=np.float32)
        out = self._audio[self._released - self._offset : target - self._offset].copy()
        for start, end in self.beeps:
            low, high = max(start, self._released), min(end, target)
            if low < high:
                phase = 2 * np.pi * self.beep_hz * np.arange(low, high) / self.sample_rate
                out[low - self._released : high - self._released] = 0.2 * np.sin(phase)
        self._released = target
        self.beeps = [(start, end) for start, end in self.beeps if end > target]
        drop = min(self._released, self.written - int(self.keep_seconds * self.sample_rate)) - self._offset
        if drop > 0:
            self._audio = self._audio[drop:]
            self._offset += drop
        return out
