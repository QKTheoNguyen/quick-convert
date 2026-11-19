from __future__ import annotations

from quick_convert.data.types import AudioBatch
from quick_convert.systems.asr import ASRSystem
from quick_convert.systems.base import InferenceSystem

from .base import PIIDetector


class TranscriptPIISystem(InferenceSystem):
    """Transcribe audio, then detect personal information in the transcripts.

    ``get_labels`` returns one ``{"transcript": str, "pii": list[dict]}``
    record per sample, with span offsets relative to that transcript. When the
    ASR system has ``word_timestamps`` enabled, the transcript is its words
    joined by spaces and each span also gets ``start_time`` and ``end_time``
    in seconds from the start of the sample.

    Args:
        asr: System that produces the transcripts.
        detector: System that finds personal information in text.
        name: Human-readable system name.
    """

    def __init__(self, asr: ASRSystem, detector: PIIDetector, name: str = "transcript_pii"):
        super().__init__()
        self.asr = asr
        self.detector = detector
        self.name = name

    def get_labels(self, batch: AudioBatch) -> list[dict]:
        if getattr(self.asr, "word_timestamps", False):
            words = self.asr.transcribe_batch_words(batch)
            transcripts = [" ".join(word["word"] for word in sample_words) for sample_words in words]
        else:
            words = None
            transcripts = self.asr.transcribe_batch(batch)
        spans = self.detector.detect(transcripts)
        if words is not None:
            for sample_spans, sample_words in zip(spans, words, strict=True):
                add_span_times(sample_spans, sample_words)
        return [{"transcript": text, "pii": pii} for text, pii in zip(transcripts, spans, strict=True)]


def add_span_times(spans: list[dict], words: list[dict]) -> None:
    """Set each span's time range from the words it overlaps in ``" ".join(words)``."""
    bounds = []
    position = 0
    for word in words:
        bounds.append((position, position + len(word["word"])))
        position += len(word["word"]) + 1
    for span in spans:
        hits = [
            word
            for word, (start, end) in zip(words, bounds, strict=True)
            if start < span["end"] and end > span["start"]
        ]
        span["start_time"] = hits[0]["start"] if hits else None
        span["end_time"] = hits[-1]["end"] if hits else None
