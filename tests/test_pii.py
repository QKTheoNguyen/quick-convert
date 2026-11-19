import sys
from types import SimpleNamespace

import torch

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.dependencies import compose_dependency_plan
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline
from quick_convert.systems.pii import OpenAIPrivacyFilter, TranscriptPIISystem
from quick_convert.systems.pii.privacy_filter import decode_bioes
from quick_convert.systems.pii.transcript import add_span_times


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
    assert compose_dependency_plan("eval_pii_librispeech").extras == ("moonshine", "privacy-filter")


def test_timed_asr_gives_each_span_the_time_range_of_its_words(tmp_path):
    words = [
        {"word": "call", "start": 0.0, "end": 0.4},
        {"word": "Bob", "start": 0.5, "end": 0.8},
        {"word": "Smith,", "start": 0.8, "end": 1.2},
        {"word": "bye", "start": 2.0, "end": 2.3},
    ]
    asr = SimpleNamespace(word_timestamps=True, transcribe_batch_words=lambda batch: [words])
    seen = []

    def detect(texts):
        seen.extend(texts)
        return [[{"label": "private_person", "start": 5, "end": 14, "text": "Bob Smith", "score": 1.0}]]

    system = TranscriptPIISystem(asr, SimpleNamespace(detect=detect))
    [record] = system.get_labels(batch=None)

    assert seen == ["call Bob Smith, bye"]
    assert (record["pii"][0]["start_time"], record["pii"][0]["end_time"]) == (0.5, 1.2)


def test_span_without_overlapping_word_has_no_time_range():
    spans = [{"start": 4, "end": 5}]
    add_span_times(spans, [{"word": "hi", "start": 0.0, "end": 0.2}])

    assert (spans[0]["start_time"], spans[0]["end_time"]) == (None, None)
