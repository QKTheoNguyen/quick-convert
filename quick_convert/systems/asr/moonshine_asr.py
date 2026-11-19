from __future__ import annotations

import torch
import torchaudio

from ...data.types import AudioBatch, MetadataSample
from .base import ASRSystem


class MoonshineASR(ASRSystem):
    """Moonshine v2 transcription adapter backed by ``moonshine-voice``.

    The ``moonshine`` extra is required. The backend runs ONNX Runtime on CPU,
    so this system has no device option. Model files are downloaded and loaded
    on the first transcription call. Non-English models are released under the
    non-commercial Moonshine Community License.

    Args:
        language: Moonshine language code or English language name.
        model_arch: Moonshine architecture name such as ``"tiny"``, ``"base"``
            or ``"medium-streaming"``. ``None`` selects the backend's default
            model for ``language``.
        pred_key: Resource or result key used for predicted transcripts.
        name: Human-readable system name.
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        language: str = "en",
        model_arch: str | None = None,
        pred_key="transcript",
        name="moonshine",
    ):
        super().__init__(name, self.SAMPLE_RATE, "cpu", pred_key=pred_key)
        try:
            import moonshine_voice
        except ImportError as error:
            raise ImportError("Moonshine ASR requires the `moonshine` extra.") from error

        self._moonshine = moonshine_voice
        self.language = language
        self.model_arch = model_arch
        # Fail on a misspelled architecture before any download.
        self._arch = None if model_arch is None else moonshine_voice.string_to_model_arch(model_arch)
        self._transcriber = None

    def _get_transcriber(self):
        if self._transcriber is None:
            model_path, arch = self._moonshine.get_model_for_language(self.language, self._arch)
            self._transcriber = self._moonshine.Transcriber(model_path=model_path, model_arch=arch)
        return self._transcriber

    def _transcribe_waveform(self, waveform: torch.Tensor, sr: int) -> str:
        transcript = self._get_transcriber().transcribe_without_streaming(waveform.float().tolist(), sr)
        return " ".join(line.text.strip() for line in transcript.lines if line.text.strip())

    def transcribe(self, sample: MetadataSample) -> str:
        waveform, sr = torchaudio.load(str(sample.path))
        return self._transcribe_waveform(waveform.mean(dim=0), sr)

    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("MoonshineASR requires a batch with loaded audio.")
        return [
            self._transcribe_waveform(waveform[: int(length)], int(sr))
            for waveform, length, sr in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True)
        ]
