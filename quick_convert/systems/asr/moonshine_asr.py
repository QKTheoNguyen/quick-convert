from __future__ import annotations

import torch
import torchaudio

from ...components.alignment import MMSForcedAligner
from ...data.types import AudioBatch, MetadataSample
from .base import ASRSystem


class MoonshineASR(ASRSystem):
    """Moonshine v2 transcription adapter backed by ``moonshine-voice``.

    The ``moonshine`` extra is required. The backend runs ONNX Runtime on CPU.
    Model files are downloaded and loaded on the first transcription call.
    Non-English models are released under the non-commercial Moonshine
    Community License.

    :meth:`transcribe_words` times the transcript's words with
    :class:`~quick_convert.components.alignment.MMSForcedAligner`, because
    Moonshine's own word timestamps are unreliable.

    Args:
        language: Moonshine language code or English language name.
        model_arch: Moonshine architecture name such as ``"tiny"``, ``"base"``
            or ``"medium-streaming"``. Defaults to ``"medium-streaming"``, which
            is only published for English; pick another architecture for other
            languages. ``None`` selects the backend's default model for
            ``language``, which can change between ``moonshine-voice`` releases.
        device: Device of the forced-alignment model used by
            :meth:`transcribe_words`; Moonshine itself always runs on CPU.
        pred_key: Resource or result key used for predicted transcripts.
        name: Human-readable system name.
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        language: str = "en",
        model_arch: str | None = "medium-streaming",
        device: str = "cpu",
        pred_key="transcript",
        name="moonshine",
    ):
        super().__init__(name, self.SAMPLE_RATE, device, pred_key=pred_key)
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
        self.aligner = MMSForcedAligner(device)

    def _get_transcriber(self):
        if self._transcriber is None:
            model_path, arch = self._moonshine.get_model_for_language(self.language, self._arch)
            self._transcriber = self._moonshine.Transcriber(model_path=model_path, model_arch=arch)
        return self._transcriber

    def _text(self, waveform: torch.Tensor, sr: int) -> str:
        lines = self._get_transcriber().transcribe_without_streaming(waveform.float().tolist(), sr).lines
        return " ".join(line.text.strip() for line in lines if line.text.strip())

    def transcribe(self, sample: MetadataSample) -> str:
        waveform, sr = torchaudio.load(str(sample.path))
        return self._text(waveform.mean(dim=0), sr)

    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("MoonshineASR requires a batch with loaded audio.")
        return [
            self._text(waveform[: int(length)], int(sr))
            for waveform, length, sr in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True)
        ]

    def transcribe_words(self, wav: torch.Tensor) -> list[dict]:
        """Transcribe with Moonshine, then force-align the words to ``wav``.

        When alignment fails (for example more characters than audio frames),
        the whole transcript is returned as one entry spanning the clip, so
        personal information in it is still covered.
        """
        waveform = wav.squeeze(0) if wav.dim() == 2 else wav
        if waveform.dim() != 1:
            raise ValueError(f"transcribe_words expects a mono [T] or [1, T] waveform, got {tuple(wav.shape)}.")
        words = self._text(waveform, self.sr).split()
        if not words:
            return []
        aligned = self.aligner.align(waveform, words)
        if aligned is None:
            return [{"word": " ".join(words), "start_s": 0.0, "end_s": len(waveform) / self.sr}]
        return aligned
