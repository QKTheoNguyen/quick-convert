"""Forced alignment of known words to speech."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


class MMSForcedAligner:
    """Time words in a waveform with torchaudio's ``MMS_FA`` CTC aligner.

    The model covers the romanized Latin alphabet. Characters outside it are
    dropped, and a word left without characters (such as ``"415"``) aligns to
    a wildcard token. Weights (about 1.2 GB) are downloaded and loaded on the
    first call.

    Args:
        device: Device used for inference.
    """

    SAMPLE_RATE = 16_000

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._dictionary = None

    def _load(self):
        if self._model is None:
            bundle = torchaudio.pipelines.MMS_FA
            self._model = bundle.get_model(with_star=True).eval().to(self.device)
            # Index 0 ("-") is the CTC blank, which cannot appear in a target, so hyphens are dropped too.
            self._dictionary = {c: i for c, i in bundle.get_dict(star="*").items() if i != 0}
        return self._model, self._dictionary

    @torch.inference_mode()
    def align(self, waveform: torch.Tensor | np.ndarray, words: list[str]) -> list[dict] | None:
        """Return one ``{"word", "start_s", "end_s"}`` entry per word, or ``None`` when they cannot be aligned.

        ``waveform`` is mono at :attr:`SAMPLE_RATE`; times are seconds from its
        start, sorted and non-overlapping, with ``start_s < end_s``.
        """
        model, dictionary = self._load()
        waveform = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)
        tokens = [
            [dictionary[c] for c in w.lower() if c in dictionary and c != "*"] or [dictionary["*"]] for w in words
        ]
        try:
            emission, _ = model(waveform[None].to(self.device))
            emission = emission.cpu()
            path, scores = torchaudio.functional.forced_align(emission, torch.tensor([sum(tokens, [])]), blank=0)
        except (RuntimeError, ValueError):  # e.g. audio too short for its tokens
            return None
        token_spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp())
        seconds_per_frame = len(waveform) / emission.shape[1] / self.SAMPLE_RATE
        aligned, index = [], 0
        for word, word_tokens in zip(words, tokens, strict=True):
            first, last = token_spans[index], token_spans[index + len(word_tokens) - 1]
            aligned.append(
                {"word": word, "start_s": first.start * seconds_per_frame, "end_s": last.end * seconds_per_frame}
            )
            index += len(word_tokens)
        return aligned
