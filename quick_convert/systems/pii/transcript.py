from __future__ import annotations

from quick_convert.data.types import AudioBatch
from quick_convert.systems.asr import ASRSystem
from quick_convert.systems.base import InferenceSystem

from .base import PIIDetector


class TranscriptPIISystem(InferenceSystem):
    """Transcribe audio into timed words, then detect personal information in them.

    ``get_labels`` returns one ``{"transcript": str, "pii": list[dict]}``
    record per sample: the words joined by spaces, and the spans of
    :meth:`PIIDetector.detect` with ``start_s`` and ``end_s`` in seconds from
    the start of the sample.

    Args:
        asr: System that produces timed words with ``transcribe_words``.
        detector: System that finds personal information in the words.
        name: Human-readable system name.
    """

    def __init__(self, asr: ASRSystem, detector: PIIDetector, name: str = "transcript_pii"):
        super().__init__()
        self.asr = asr
        self.detector = detector
        self.name = name

    def get_labels(self, batch: AudioBatch) -> list[dict]:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("TranscriptPIISystem requires a batch with loaded audio.")
        records = []
        for waveform, length, sr in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True):
            if int(sr) != self.asr.sr:
                raise ValueError(f"Expected audio at {self.asr.sr} Hz, got {int(sr)} Hz.")
            words = self.asr.transcribe_words(waveform[: int(length)])
            records.append({"transcript": " ".join(w["word"] for w in words), "pii": self.detector.detect(words)})
        return records
