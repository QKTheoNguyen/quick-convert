from __future__ import annotations

from abc import ABC, abstractmethod

from quick_convert.systems.base import InferenceSystem


class PIIDetector(InferenceSystem, ABC):
    """Base interface for systems that find personal information in text.

    Each detected span is a dictionary with ``label``, character offsets
    ``start`` and ``end`` into the input text, the matched ``text`` and a
    ``score`` in ``[0, 1]``.
    """

    @abstractmethod
    def detect(self, texts: list[str]) -> list[list[dict]]:
        """Return the detected spans of each text, preserving input order."""
        ...


def mask_text(text: str, spans: list[dict]) -> str:
    """Replace each detected span with ``[LABEL]``; overlapping spans share one mask."""
    parts = []
    position = 0
    for span in sorted(spans, key=lambda span: span["start"]):
        if span["start"] < position:
            position = max(position, span["end"])
            continue
        parts += [text[position : span["start"]], f"[{span['label'].upper()}]"]
        position = span["end"]
    parts.append(text[position:])
    return "".join(parts)
