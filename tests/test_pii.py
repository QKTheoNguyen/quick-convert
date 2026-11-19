import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.dependencies import compose_dependency_plan
from quick_convert.pipelines.evaluation.pipeline import EvalPipeline
from quick_convert.systems.pii import (
    OpenAIPrivacyFilter,
    StreamingPIIBeeper,
    StreamingPIIMasker,
    TranscriptPIISystem,
    mask_text,
)
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

    masker = StreamingPIIMasker(SimpleNamespace(detect=detect), context_words=4)

    assert masker.mask("you can call me at 415", done=True) == "you can call me at [PRIVATE_PHONE]"
    assert masker.mask("555 01", done=False) == "[PRIVATE_PHONE]"
    assert masker.mask("555 0132", done=True) == "[PRIVATE_PHONE]"
    assert seen == ["you can call me at 415", "call me at 415 555 01", "call me at 415 555 0132"]
    assert StreamingPIIMasker(SimpleNamespace(detect=detect), context_words=0).mask("555 0132") == "555 0132"


def _name_detector(texts):
    return [[{"label": "private_person", "start": m.start(), "end": m.end()} for m in re.finditer("sarah", texts[0])]]


def test_streaming_beeper_delays_audio_and_beeps_realigned_padded_spans():
    segments = []

    def align(waveform, words):
        segments.append((len(waveform), words))
        return [{"word": "hi", "start": 0.0, "end": 0.5}, {"word": "sarah", "start": 0.5, "end": 1.0}]

    masker = StreamingPIIMasker(SimpleNamespace(detect=_name_detector))
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
        StreamingPIIMasker(SimpleNamespace(detect=_name_detector)),
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
    masker = StreamingPIIMasker(SimpleNamespace(detect=lambda texts: (calls.append(texts[0]), [[]])[1]))
    beeper = StreamingPIIBeeper(masker, lambda waveform, words: calls.append("align"), sample_rate=100)
    beeper.write(np.ones(100, dtype=np.float32))

    beeper.update("hello", start_time=0.0)
    beeper.update("hello there", start_time=0.0, done=True)
    beeper.update("again", start_time=1.0)
    assert calls == ["hello", "hello there", "hello there again"]
