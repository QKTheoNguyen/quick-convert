import sys
from types import SimpleNamespace

import torch

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.dependencies import compose_dependency_plan
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline
from quick_convert.systems.pii import OpenAIPrivacyFilter, TranscriptPIISystem
from quick_convert.systems.pii.privacy_filter import decode_bioes


TEXT = "Call Alice Smith at 555 1234 now"
# Byte-level tokens carry their leading space, as in the model's tokenizer.
OFFSETS = [(0, 0), (0, 4), (4, 10), (10, 16), (16, 19), (19, 23), (23, 28), (28, 32)]
TAGS = ["O", "O", "B-private_person", "E-private_person", "O", "B-private_phone", "E-private_phone", "O"]


def test_decode_bioes_merges_tokens_into_trimmed_character_spans():
    spans = decode_bioes(TEXT, OFFSETS, TAGS, [1.0, 1.0, 0.8, 0.6, 1.0, 0.9, 0.7, 1.0])

    assert [(s["label"], s["text"], s["start"], s["end"]) for s in spans] == [
        ("private_person", "Alice Smith", 5, 16),
        ("private_phone", "555 1234", 20, 28),
    ]
    assert spans[0]["score"] == 0.7


def test_decode_bioes_splits_adjacent_singletons_and_label_changes():
    spans = decode_bioes("ab cd ef", [(0, 2), (2, 5), (5, 8)], ["S-secret", "S-secret", "I-private_url"], [1, 1, 1])

    assert [(s["label"], s["text"]) for s in spans] == [("secret", "ab"), ("secret", "cd"), ("private_url", "ef")]


def test_privacy_filter_runs_token_classification_with_offsets(monkeypatch):
    labels = sorted(set(TAGS))
    id2label = dict(enumerate(labels))

    class Encoding(dict):
        def to(self, device):
            return self

    class FakeModel:
        config = SimpleNamespace(id2label=id2label)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            logits = torch.nn.functional.one_hot(torch.tensor([labels.index(t) for t in TAGS]), len(labels))
            return SimpleNamespace(logits=logits[None].float() * 10)

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda name: (
                lambda text, **kw: Encoding(
                    input_ids=torch.zeros(1, len(OFFSETS), dtype=torch.long), offset_mapping=torch.tensor([OFFSETS])
                )
            )
        ),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=lambda name, **kw: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    spans = OpenAIPrivacyFilter().detect([TEXT])

    assert [[s["text"] for s in text_spans] for text_spans in spans] == [["Alice Smith", "555 1234"]]


def test_evaluation_without_metrics_records_transcript_pii_columns(tmp_path):
    asr = SimpleNamespace(transcribe_batch=lambda batch: ["hi Bob"] * len(batch))
    span = {"label": "private_person", "start": 3, "end": 6, "text": "Bob", "score": 0.9}
    detector = SimpleNamespace(detect=lambda texts: [[span] for _ in texts])
    batch = AudioBatch.from_samples([AudioSample(utt_id="u1", path=tmp_path / "u1.wav", split="test")])
    pipeline = EvalPipeline(
        dataset=SimpleNamespace(make_dataloader=lambda **kwargs: [batch]),
        system=TranscriptPIISystem(asr, detector),
        metrics=[],
        out_dir=tmp_path,
        batch_size=1,
    )

    assert pipeline.generate_records() == [
        {
            "utt_id": "u1",
            "path": str(tmp_path / "u1.wav"),
            "split": "test",
            "pred_transcript": "hi Bob",
            "pred_pii": [span],
        }
    ]


def test_pii_run_requires_asr_and_privacy_filter_extras():
    assert compose_dependency_plan("eval_pii_librispeech").extras == ("privacy-filter", "whisper")
