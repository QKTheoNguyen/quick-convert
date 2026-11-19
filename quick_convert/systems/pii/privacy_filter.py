from __future__ import annotations

import torch

from .base import PIIDetector


class OpenAIPrivacyFilter(PIIDetector):
    """OpenAI Privacy Filter token-classification adapter.

    The ``privacy-filter`` extra is required. Weights are downloaded and loaded
    on the first detection call. Labels are ``account_number``,
    ``private_address``, ``private_email``, ``private_person``,
    ``private_phone``, ``private_url``, ``private_date`` and ``secret``.
    The model is trained primarily on English text.

    Args:
        device: Device used for inference.
        model_name: Hugging Face model identifier or local directory.
        name: Human-readable system name.
    """

    def __init__(self, device: str = "cpu", model_name: str = "openai/privacy-filter", name: str = "privacy_filter"):
        super().__init__()
        try:
            import transformers
        except ImportError as error:
            raise ImportError("OpenAI Privacy Filter requires the `privacy-filter` extra.") from error

        self._transformers = transformers
        self.device = device
        self.model_name = model_name
        self.name = name
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(self.model_name)
            self._model = self._transformers.AutoModelForTokenClassification.from_pretrained(
                self.model_name, dtype="auto"
            ).to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    @torch.inference_mode()
    def detect(self, texts: list[str]) -> list[list[dict]]:
        tokenizer, model = self._load()
        results = []
        # One text per forward pass avoids relying on a padding token.
        for text in texts:
            encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")[0].tolist()
            probs = model(**encoded.to(self.device)).logits[0].float().softmax(dim=-1)
            scores, ids = probs.max(dim=-1)
            tags = [model.config.id2label[i] for i in ids.tolist()]
            results.append(decode_bioes(text, offsets, tags, scores.tolist()))
        return results


def decode_bioes(text: str, offsets: list, tags: list[str], scores: list[float]) -> list[dict]:
    """Merge per-token BIOES tags into character spans.

    shortcut: greedy per-token argmax, not the model card's constrained Viterbi
    decoder; implement it if fragmented spans show up on real transcripts.
    """
    spans = []
    current = None
    for (start, end), tag, score in zip(offsets, tags, scores, strict=True):
        if end <= start:  # special tokens carry empty offsets
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "O":
            current = None
            continue
        if current is None or prefix in "BS" or label != current["label"]:
            current = {"label": label, "start": start, "end": end, "scores": [score]}
            spans.append(current)
        else:
            current["end"] = end
            current["scores"].append(score)
        if prefix in "ES":
            current = None

    for span in spans:
        raw = text[span["start"] : span["end"]]
        span["start"] += len(raw) - len(raw.lstrip())
        span["end"] -= len(raw) - len(raw.rstrip())
        span["text"] = text[span["start"] : span["end"]]
        scores = span.pop("scores")
        span["score"] = sum(scores) / len(scores)
    return spans
