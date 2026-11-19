from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quick_convert.systems.base import InferenceSystem

from ...data.types import AudioBatch, MetadataSample


class ASRSystem(InferenceSystem, ABC):
    """Base interface for systems that transcribe audio samples and batches.

    Args:
        name: Human-readable system name.
        sr: Required audio sample rate in hertz.
        device: Device used for inference.
        pred_key: Resource or result key used for predicted transcripts.
    """

    def __init__(self, name: str, sr: int, device: str, pred_key="transcript"):
        super().__init__()
        self.device = device
        self.name = name
        self.sr = sr
        self.pred_key = pred_key

    @abstractmethod
    def transcribe(self, sample: MetadataSample) -> str:
        """Transcribe one sample."""
        ...

    @abstractmethod
    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        """Transcribe a batch and preserve input order."""
        ...

    def transcribe_words(self, wav: torch.Tensor) -> list[dict]:
        """Transcribe one mono clip at :attr:`sr` into timed words.

        Args:
            wav: Mono float waveform of shape ``[T]`` or ``[1, T]``.

        Returns:
            One ``{"word", "start_s", "end_s"}`` entry per word, in seconds from
            the clip start, sorted and non-overlapping with
            ``0 <= start_s < end_s <= T / sr``; ``[]`` for an empty transcript.
        """
        raise NotImplementedError(f"{type(self).__name__} does not produce word timestamps.")

    def predict_batch(self, batch: AudioBatch):
        return self.transcribe_batch(batch)

    def get_labels(self, batch: AudioBatch) -> list[str]:
        return self.transcribe_batch(batch)
