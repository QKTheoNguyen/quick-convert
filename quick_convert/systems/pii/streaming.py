from __future__ import annotations

from collections import deque

from .base import PIIDetector, mask_text


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

    def mask(self, text: str, done: bool = False) -> str:
        """Mask the current line; ``done`` adds it to the context of later lines."""
        if not text.strip():
            return text
        prefix = " ".join(self._context)
        prefix = f"{prefix} " if prefix else ""
        spans = [
            {**span, "start": max(span["start"] - len(prefix), 0), "end": span["end"] - len(prefix)}
            for span in self.detector.detect([prefix + text])[0]
            if span["end"] > len(prefix)
        ]
        if done:
            self._context.extend(text.split())
        return mask_text(text, spans)
