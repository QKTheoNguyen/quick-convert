from __future__ import annotations

from abc import ABC, abstractmethod

from quick_convert.systems.base import InferenceSystem


class PIIDetector(InferenceSystem, ABC):
    """Base interface for systems that find personal information in text and timed words.

    Subclasses implement :meth:`detect_text`; :meth:`detect` builds on it to
    time spans with the words of :meth:`ASRSystem.transcribe_words`.
    """

    name: str

    @abstractmethod
    def detect_text(self, texts: list[str]) -> list[list[dict]]:
        """Return the character spans of each text, preserving input order.

        Each span has ``label``, character offsets ``start`` and ``end``, the
        matched ``text`` and a ``confidence`` in ``[0, 1]``.
        """
        ...

    def detect(self, words: list[dict]) -> list[dict]:
        """Return the personal information in timed words, sorted by ``start_s``.

        Args:
            words: ``{"word", "start_s", "end_s"}`` entries, as returned by
                ``ASRSystem.transcribe_words``.

        Returns:
            One ``{"start_s", "end_s", "label", "confidence", "text"}`` entry per
            span: the start of its first word, the end of its last word, and
            those words joined by spaces. Spans may overlap.
        """
        text = " ".join(word["word"] for word in words)
        if not text.strip():
            return []
        return timed_spans(self.detect_text([text])[0], words)


def timed_spans(spans: list[dict], words: list[dict]) -> list[dict]:
    """Time character spans of ``" ".join(words)`` with the words they overlap, sorted by start.

    A span without a ``confidence`` gets ``1.0``; one that overlaps no word is dropped.
    """
    bounds = []
    position = 0
    for word in words:
        bounds.append((position, position + len(word["word"])))
        position += len(word["word"]) + 1
    timed = []
    for span in spans:
        hits = [
            word
            for word, (start, end) in zip(words, bounds, strict=True)
            if start < span["end"] and end > span["start"]
        ]
        if hits:
            timed.append(
                {
                    "start_s": hits[0]["start_s"],
                    "end_s": hits[-1]["end_s"],
                    "label": span["label"],
                    "confidence": span.get("confidence", 1.0),
                    "text": " ".join(word["word"] for word in hits),
                }
            )
    return sorted(timed, key=lambda span: span["start_s"])


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
