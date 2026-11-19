import sys
from types import SimpleNamespace

import torch

from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.tokenizer.bpe import (
    SentencePieceBPETrainer as LegacySentencePieceBPETrainer,
)
from quick_convert.systems.asr.moonshine_asr import MoonshineASR
from quick_convert.systems.asr.whisper_asr import WhisperASR
from quick_convert.training.tokenizers.sentencepiece import SentencePieceBPETrainer


def test_legacy_tokenizer_trainer_import_is_compatible():
    assert LegacySentencePieceBPETrainer is SentencePieceBPETrainer


def test_jiwer_module_remains_available_after_construction(monkeypatch):
    fake_jiwer = SimpleNamespace(
        Compose=lambda transforms: transforms,
        ToLowerCase=lambda: object(),
        RemovePunctuation=lambda: object(),
        Strip=lambda: object(),
        ReduceToListOfListOfWords=lambda: object(),
        wer=lambda references, hypotheses, **kwargs: 0.25,
    )
    monkeypatch.setitem(sys.modules, "jiwer", fake_jiwer)

    from quick_convert.pipelines.evaluation.metrics import JiwerWER

    metric = JiwerWER()

    assert metric.compute("reference", "hypothesis") == {"wer": 0.25}


def test_sentencepiece_module_remains_available_after_construction(monkeypatch, tmp_path):
    trained_with = {}

    class FakeTrainer:
        @staticmethod
        def train(**kwargs):
            trained_with.update(kwargs)

    class FakeProcessor:
        def load(self, path):
            self.path = path

        def get_piece_size(self):
            return 17

    fake_sentencepiece = SimpleNamespace(
        SentencePieceTrainer=FakeTrainer,
        SentencePieceProcessor=FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_sentencepiece)

    trainer = SentencePieceBPETrainer(vocab_size=17)
    trainer._train(model_prefix="tokenizer")
    trainer._model = trainer._load(tmp_path / "tokenizer.model")

    assert trained_with["model_prefix"] == "tokenizer"
    assert trainer.vocab_size_actual == 17


def test_whisper_batch_path_uses_retained_optional_module(monkeypatch):
    class FakeModel:
        def to(self, device):
            return self

        def decode(self, mel, options):
            return [SimpleNamespace(text="hello")]

    fake_whisper = SimpleNamespace(
        DecodingOptions=lambda **kwargs: kwargs,
        load_model=lambda name: FakeModel(),
        pad_or_trim=lambda waveform: waveform,
        log_mel_spectrogram=lambda waveform: waveform,
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    system = WhisperASR(device="cpu")
    batch = AudioBatch(
        utt_ids=["sample"],
        paths=[],
        splits=[None],
        resources={},
        waveforms=torch.zeros(1, 16),
        lengths=torch.tensor([16]),
        sample_rates=torch.tensor([16_000]),
    )

    assert system.get_labels(batch) == ["hello"]


def test_moonshine_batch_path_trims_padding_and_joins_lines(monkeypatch):
    calls = []

    class FakeTranscriber:
        def __init__(self, model_path, model_arch, options=None):
            calls.append(("load", model_path, model_arch))

        def transcribe_without_streaming(self, audio, sample_rate):
            calls.append((len(audio), sample_rate))
            lines = [SimpleNamespace(text=" hello "), SimpleNamespace(text=""), SimpleNamespace(text="world")]
            return SimpleNamespace(lines=lines)

    fake_moonshine = SimpleNamespace(
        string_to_model_arch=lambda name: {"tiny": 0}[name],
        get_model_for_language=lambda language, arch, include_word_timestamps: (f"/models/{language}", arch),
        Transcriber=FakeTranscriber,
    )
    monkeypatch.setitem(sys.modules, "moonshine_voice", fake_moonshine)

    system = MoonshineASR(model_arch="tiny")
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[],
        splits=[None, None],
        resources={},
        waveforms=torch.zeros(2, 16),
        lengths=torch.tensor([16, 9]),
        sample_rates=torch.tensor([16_000, 16_000]),
    )

    assert system.sr == MoonshineASR.SAMPLE_RATE
    assert system.get_labels(batch) == ["hello world", "hello world"]
    assert calls == [("load", "/models/en", 0), (16, 16_000), (9, 16_000)]


def test_moonshine_word_timestamps_keep_lines_without_word_timings(monkeypatch):
    loaded = {}

    def get_model(language, arch, include_word_timestamps):
        loaded["include_word_timestamps"] = include_word_timestamps
        return "/models/en", arch

    class FakeTranscriber:
        def __init__(self, model_path, model_arch, options=None):
            loaded["options"] = options

        def transcribe_without_streaming(self, audio, sample_rate):
            timed = SimpleNamespace(
                text="Hi Bob.",
                words=[
                    SimpleNamespace(word="Hi", start=0.1, end=0.3),
                    SimpleNamespace(word=" Bob.", start=0.3, end=0.6),
                ],
            )
            untimed = SimpleNamespace(text=" Bye ", words=None, start_time=1.0, duration=0.5)
            return SimpleNamespace(lines=[timed, untimed])

    monkeypatch.setitem(
        sys.modules,
        "moonshine_voice",
        SimpleNamespace(string_to_model_arch=None, get_model_for_language=get_model, Transcriber=FakeTranscriber),
    )
    batch = AudioBatch(
        utt_ids=["a"],
        paths=[],
        splits=[None],
        resources={},
        waveforms=torch.zeros(1, 16),
        lengths=torch.tensor([16]),
        sample_rates=torch.tensor([16_000]),
    )

    words = MoonshineASR(word_timestamps=True).transcribe_batch_words(batch)

    assert loaded == {"include_word_timestamps": True, "options": {"word_timestamps": "true"}}
    assert words == [
        [
            {"word": "Hi", "start": 0.1, "end": 0.3},
            {"word": "Bob.", "start": 0.3, "end": 0.6},
            {"word": "Bye", "start": 1.0, "end": 1.5},
        ]
    ]
