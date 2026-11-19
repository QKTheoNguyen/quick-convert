"""Contract checks of the real ASR and PII models on a short English clip.

Opt in with ``QC_PII_MODEL_TESTS=1`` and the ``moonshine`` and
``privacy-filter`` extras; models are downloaded on first use. Set
``QC_PII_SAMPLE_WAV`` to a clip of your own, otherwise a LibriSpeech-style
clip is required at that path.
"""

import os

import pytest
import torchaudio
from pii_contract import assert_pii_spans, assert_timed_words


pytestmark = pytest.mark.skipif(os.environ.get("QC_PII_MODEL_TESTS") != "1", reason="Optional model downloads")


@pytest.fixture(scope="module")
def timed_words():
    from quick_convert.systems.asr import MoonshineASR

    asr = MoonshineASR()
    wav, sr = torchaudio.load(os.environ["QC_PII_SAMPLE_WAV"])
    wav = torchaudio.functional.resample(wav.mean(dim=0), sr, asr.sr)
    words = asr.transcribe_words(wav)
    assert asr.transcribe_words(wav) == words  # safe to call repeatedly
    return words, len(wav) / asr.sr


def test_moonshine_transcribe_words_follows_contract(timed_words):
    words, duration = timed_words

    assert words
    assert_timed_words(words, duration)


def test_privacy_filter_detect_follows_contract(timed_words):
    from quick_convert.systems.pii import OpenAIPrivacyFilter

    words, _ = timed_words
    spans = OpenAIPrivacyFilter().detect(words)

    assert_pii_spans(spans, words)
