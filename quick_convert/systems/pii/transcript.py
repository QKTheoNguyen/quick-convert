from __future__ import annotations

from quick_convert.data.types import AudioBatch
from quick_convert.systems.asr import ASRSystem
from quick_convert.systems.base import InferenceSystem

from .base import PIIDetector


class TranscriptPIISystem(InferenceSystem):
    """Transcribe audio, then detect personal information in the transcripts.

    ``get_labels`` returns one ``{"transcript": str, "pii": list[dict]}``
    record per sample, with span offsets relative to that transcript.

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
        transcripts = self.asr.transcribe_batch(batch)
        spans = self.detector.detect(transcripts)
        return [{"transcript": text, "pii": pii} for text, pii in zip(transcripts, spans, strict=True)]
