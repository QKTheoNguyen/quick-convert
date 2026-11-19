import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
from pii_contract import assert_pii_spans

from quick_convert.data import AudioBatch
from quick_convert.dependencies import compose_dependency_plan
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline
from quick_convert.systems.pii import (
    OpenAIPrivacyFilter,
    PIIDetector,
    StreamingPIIBeeper,
    StreamingPIIMasker,
    TranscriptPIISystem,
    mask_text,
)
from quick_convert.systems.pii.base import timed_spans
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
    assert spans[0]["confidence"] == 0.7


def test_decode_bioes_splits_adjacent_singletons_and_label_changes():
    spans = decode_bioes("ab cd ef", [(0, 2), (2, 5), (5, 8)], ["S-secret", "S-secret", "I-private_url"], [1, 1, 1])

    assert [(s["label"], s["text"]) for s in spans] == [("secret", "ab"), ("secret", "cd"), ("private_url", "ef")]


def _fake_transformers(monkeypatch):
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


def test_privacy_filter_runs_token_classification_with_offsets(monkeypatch):
    _fake_transformers(monkeypatch)

    spans = OpenAIPrivacyFilter().detect_text([TEXT])

    assert [[s["text"] for s in text_spans] for text_spans in spans] == [["Alice Smith", "555 1234"]]


def test_privacy_filter_detects_spans_timed_by_asr_words(monkeypatch):
    _fake_transformers(monkeypatch)
    words = [{"word": w, "start_s": i / 2, "end_s": i / 2 + 0.25} for i, w in enumerate(TEXT.split())]

    spans = OpenAIPrivacyFilter().detect(words)

    assert [(s["label"], s["text"], s["start_s"], s["end_s"]) for s in spans] == [
        ("private_person", "Alice Smith", 0.5, 1.25),
        ("private_phone", "555 1234", 2.0, 2.75),
    ]
    assert_pii_spans(spans, words)
    assert OpenAIPrivacyFilter().detect([]) == []


def _loaded_batch(path, samples=16):
    return AudioBatch(
        utt_ids=["u1"],
        paths=[path],
        splits=["test"],
        resources={},
        waveforms=torch.zeros(1, 32),
        lengths=torch.tensor([samples]),
        sample_rates=torch.tensor([16_000]),
    )


def test_evaluation_without_metrics_records_transcript_pii_columns(tmp_path):
    words = [{"word": "hi", "start_s": 0.0, "end_s": 0.2}, {"word": "Bob", "start_s": 0.3, "end_s": 0.6}]
    asr = SimpleNamespace(sr=16_000, transcribe_words=lambda wav: words)
    span = {"start_s": 0.3, "end_s": 0.6, "label": "private_person", "confidence": 0.9, "text": "Bob"}
    detector = SimpleNamespace(detect=lambda words: [span])
    pipeline = EvalPipeline(
        dataset=SimpleNamespace(make_dataloader=lambda **kwargs: [_loaded_batch(tmp_path / "u1.wav")]),
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


class _FixedDetector(PIIDetector):
    def __init__(self, spans):
        super().__init__()
        self.spans = spans
        self.seen = []

    def detect_text(self, texts):
        self.seen.extend(texts)
        return [self.spans for _ in texts]


def test_transcript_pii_gives_each_span_the_time_range_of_its_words(tmp_path):
    words = [
        {"word": "call", "start_s": 0.0, "end_s": 0.4},
        {"word": "Bob", "start_s": 0.5, "end_s": 0.8},
        {"word": "Smith,", "start_s": 0.8, "end_s": 1.2},
        {"word": "bye", "start_s": 2.0, "end_s": 2.3},
    ]
    lengths = []
    asr = SimpleNamespace(sr=16_000, transcribe_words=lambda wav: (lengths.append(len(wav)), words)[1])
    detector = _FixedDetector(
        [{"label": "private_person", "start": 5, "end": 14, "text": "Bob Smith", "confidence": 0.8}]
    )

    [record] = TranscriptPIISystem(asr, detector).get_labels(_loaded_batch(tmp_path / "u1.wav", samples=20))

    assert lengths == [20]  # padding trimmed
    assert detector.seen == ["call Bob Smith, bye"]
    assert record["pii"] == [
        {"start_s": 0.5, "end_s": 1.2, "label": "private_person", "confidence": 0.8, "text": "Bob Smith,"}
    ]
    assert_pii_spans(record["pii"], words)


def test_detect_sorts_spans_defaults_confidence_and_skips_empty_transcripts():
    words = [{"word": "Alice", "start_s": 0.1, "end_s": 0.4}, {"word": "Bob", "start_s": 0.5, "end_s": 0.7}]
    detector = _FixedDetector([{"label": "b", "start": 6, "end": 9}, {"label": "a", "start": 0, "end": 5}])

    spans = detector.detect(words)

    assert [(s["label"], s["confidence"]) for s in spans] == [("a", 1.0), ("b", 1.0)]
    assert_pii_spans(spans, words)
    assert detector.detect([]) == [] and detector.seen == ["Alice Bob"]


def test_span_without_overlapping_word_is_dropped():
    assert timed_spans([{"label": "x", "start": 4, "end": 5}], [{"word": "hi", "start_s": 0.0, "end_s": 0.2}]) == []


def test_mask_text_replaces_spans_and_merges_overlaps():
    spans = [
        {"label": "private_phone", "start": 20, "end": 28},
        {"label": "private_person", "start": 5, "end": 16},
        {"label": "secret", "start": 22, "end": 32},
    ]

    assert mask_text(TEXT, spans) == "Call [PRIVATE_PERSON] at [PRIVATE_PHONE]"


def test_streaming_masker_uses_finished_lines_as_context_and_clips_spans():
    seen = []

    def detect(texts):
        # Tags digits as a phone number only when the cue is present, like the real model.
        seen.append(texts[0])
        match = re.search(r"\d[\d ]*\d", texts[0])
        if "call me at" not in texts[0] or match is None:
            return [[]]
        return [[{"label": "private_phone", "start": match.start(), "end": match.end()}]]

    masker = StreamingPIIMasker(SimpleNamespace(detect_text=detect), context_words=4)

    assert masker.mask("you can call me at 415", done=True) == "you can call me at [PRIVATE_PHONE]"
    assert masker.mask("555 01", done=False) == "[PRIVATE_PHONE]"
    assert masker.mask("555 0132", done=True) == "[PRIVATE_PHONE]"
    assert seen == ["you can call me at 415", "call me at 415 555 01", "call me at 415 555 0132"]
    assert StreamingPIIMasker(SimpleNamespace(detect_text=detect), context_words=0).mask("555 0132") == "555 0132"


def _name_detector(texts):
    return [[{"label": "private_person", "start": m.start(), "end": m.end()} for m in re.finditer("sarah", texts[0])]]


def test_streaming_beeper_delays_audio_and_beeps_realigned_padded_spans():
    segments = []

    def align(waveform, words):
        segments.append((len(waveform), words))
        return [{"word": "hi", "start_s": 0.0, "end_s": 0.5}, {"word": "sarah", "start_s": 0.5, "end_s": 1.0}]

    masker = StreamingPIIMasker(SimpleNamespace(detect_text=_name_detector))
    beeper = StreamingPIIBeeper(masker, align, sample_rate=100, delay=1.0, pad_before=0.1, pad_after=0.1)
    beeper.write(np.ones(300, dtype=np.float32))

    assert beeper.update("hi sarah", start_time=1.0) == "hi [PRIVATE_PERSON]"
    assert segments == [(200, ["hi", "sarah"])]  # the line's audio from its start time, and its words
    beeper.update("hi sarah how", start_time=1.0)
    assert len(segments) == 1  # same finding on the same line: no realignment
    released = beeper.read()
    assert len(released) == 200  # 3 s written, 1 s held back
    assert (released[:140] == 1).all() and (released[140:] != 1).all()  # beep from 1.5 s - 0.1 s
    tail = beeper.read(final=True)
    assert (tail[:10] != 1).all() and (tail[10:] == 1).all()  # beep until 2.0 s + 0.1 s
    assert len(beeper.read(final=True)) == 0


def test_streaming_beeper_beeps_whole_line_when_alignment_fails():
    beeper = StreamingPIIBeeper(
        StreamingPIIMasker(SimpleNamespace(detect_text=_name_detector)),
        lambda waveform, words: None,
        sample_rate=100,
        delay=0.0,
        pad_before=0.0,
        pad_after=0.0,
    )
    beeper.write(np.ones(300, dtype=np.float32))
    beeper.update("hi sarah", start_time=1.0)
    released = beeper.read()
    assert (released[:100] == 1).all() and (released[100:] != 1).all()


def test_streaming_beeper_aligns_only_lines_with_findings_and_keeps_finished_context():
    calls = []
    masker = StreamingPIIMasker(SimpleNamespace(detect_text=lambda texts: (calls.append(texts[0]), [[]])[1]))
    beeper = StreamingPIIBeeper(masker, lambda waveform, words: calls.append("align"), sample_rate=100)
    beeper.write(np.ones(100, dtype=np.float32))

    beeper.update("hello", start_time=0.0)
    beeper.update("hello there", start_time=0.0, done=True)
    beeper.update("again", start_time=1.0)
    assert calls == ["hello", "hello there", "hello there again"]
