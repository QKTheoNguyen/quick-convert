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
            or ``"medium-streaming"``. Defaults to ``"medium-streaming"``, which
            is only published for English; pick another architecture for other
            languages. ``None`` selects the backend's default model for
            ``language``, which can change between ``moonshine-voice`` releases.
        word_timestamps: Download the extra alignment decoder and enable
            ``transcribe_batch_words``.
        pred_key: Resource or result key used for predicted transcripts.
        name: Human-readable system name.
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        language: str = "en",
        model_arch: str | None = "medium-streaming",
        word_timestamps: bool = False,
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
        self.word_timestamps = word_timestamps
        # Fail on a misspelled architecture before any download.
        self._arch = None if model_arch is None else moonshine_voice.string_to_model_arch(model_arch)
        self._transcriber = None

    def _get_transcriber(self):
        if self._transcriber is None:
            model_path, arch = self._moonshine.get_model_for_language(
                self.language, self._arch, include_word_timestamps=self.word_timestamps
            )
            options = {"word_timestamps": "true"} if self.word_timestamps else None
            self._transcriber = self._moonshine.Transcriber(model_path=model_path, model_arch=arch, options=options)
        return self._transcriber

    def _lines(self, waveform: torch.Tensor, sr: int) -> list:
        return self._get_transcriber().transcribe_without_streaming(waveform.float().tolist(), sr).lines

    def _join(self, lines: list) -> str:
        return " ".join(line.text.strip() for line in lines if line.text.strip())

    def _batch_waveforms(self, batch: AudioBatch):
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("MoonshineASR requires a batch with loaded audio.")
        for waveform, length, sr in zip(batch.waveforms, batch.lengths, batch.sample_rates, strict=True):
            yield waveform[: int(length)], int(sr)

    def transcribe(self, sample: MetadataSample) -> str:
        waveform, sr = torchaudio.load(str(sample.path))
        return self._join(self._lines(waveform.mean(dim=0), sr))

    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        return [self._join(self._lines(waveform, sr)) for waveform, sr in self._batch_waveforms(batch)]

    def transcribe_batch_words(self, batch: AudioBatch) -> list[list[dict]]:
        """Return ``{"word", "start", "end"}`` entries per sample, in seconds.

        A line the backend returns without word timings becomes one entry
        spanning the whole line, so no transcribed text is dropped.
        """
        if not self.word_timestamps:
            raise ValueError("transcribe_batch_words requires MoonshineASR(word_timestamps=True).")
        results = []
        for waveform, sr in self._batch_waveforms(batch):
            words = []
            for line in self._lines(waveform, sr):
                if line.words:
                    words += [
                        {"word": w.word.strip(), "start": w.start, "end": w.end} for w in line.words if w.word.strip()
                    ]
                elif line.text.strip():
                    end = line.start_time + line.duration
                    words.append({"word": line.text.strip(), "start": line.start_time, "end": end})
            results.append(words)
        return results
