"""Rolling speaker metrics with deterministic encoders; no model downloads."""

import json
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from quick_convert.components.encoders.speaker.base import SpeakerEmbedding
from quick_convert.components.metrics.streaming_speaker import (
    AsyncRollingSpeakerSimilarity,
    RollingSpeakerSimilarity,
)


class MockEncoder:
    sample_rate = 8
    output = "pooled"

    def __init__(self, vector=None):
        self.calls = []
        self.vector = vector

    def encode(self, audio, sr):
        assert sr == self.sample_rate
        assert torch.is_inference_mode_enabled()
        self.calls.append(audio.clone())
        vector = self.vector if self.vector is not None else torch.stack((audio.mean(), audio.square().mean()))
        return SpeakerEmbedding(vector.clone(), 2, "mock", "deterministic")


def evaluator(encoder=None, **kwargs):
    return RollingSpeakerSimilarity(encoder or MockEncoder(), 8, 8, aligned=True, **kwargs)


def test_four_second_window_two_second_hop_and_no_duplicates():
    core = evaluator(session_id="recording")
    original = torch.arange(1, 81, dtype=torch.float32)
    saved = original.clone()
    core.add_original(original)
    core.add_anonymized(original)
    results = core.poll()
    assert [(r.window_start_seconds, r.window_end_seconds) for r in results] == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert [r.window_index for r in results] == list(range(4))
    assert all(r.session_id == "recording" and r.status == "ok" for r in results)
    assert [r.speaker_similarity for r in results] == pytest.approx([1] * 4)
    assert core.poll() == []
    torch.testing.assert_close(original, saved)
    for i in range(4):
        torch.testing.assert_close(core.encoder.calls[2 * i], original[16 * i : 16 * i + 32])
    assert json.loads(json.dumps(results[0].to_dict(), allow_nan=False))["session_id"] == "recording"


def test_partial_unequal_and_independent_arrival():
    core = evaluator()
    core.add_original(torch.ones(31))
    core.add_anonymized(torch.ones(12))
    assert core.poll() == []
    core.add_anonymized(torch.ones(36))
    assert core.poll() == []
    core.add_original(torch.ones(1))
    assert [r.window_index for r in core.poll()] == [0]
    core.add_original(torch.ones(16))
    assert [r.window_index for r in core.poll()] == [1]
    assert core.poll() == []


def test_different_rates_resample_complete_windows_independent_of_chunking():
    def run(chunk_size):
        encoder = MockEncoder()
        core = RollingSpeakerSimilarity(encoder, 16, 8, aligned=True)
        a = torch.linspace(0.1, 1.0, 96)
        b = torch.linspace(0.1, 1.0, 48)
        for part in a.split(chunk_size):
            core.add_original(part)
        for part in b.split(chunk_size):
            core.add_anonymized(part)
        result = core.poll()
        assert all(r.status == "ok" for r in result)
        assert all(call.numel() == 32 for call in encoder.calls)
        assert [(r.window_start_seconds, r.window_end_seconds) for r in result] == [(0, 4), (2, 6)]
        return [r.speaker_similarity for r in result], encoder.calls

    a, calls_a = run(7)
    b, calls_b = run(19)
    assert a == b
    for left, right in zip(calls_a, calls_b, strict=True):
        torch.testing.assert_close(left, right)


def test_unknown_alignment_never_calls_encoder():
    encoder = MockEncoder()
    core = RollingSpeakerSimilarity(encoder, 8, 8)
    core.add_original(torch.ones(32))
    core.add_anonymized(torch.ones(32))
    result = core.poll()[0]
    assert result.status == "unavailable" and result.speaker_similarity is None
    assert "Alignment" in result.error_message
    assert encoder.calls == []


def test_trimmed_leading_audio_and_output_latency_are_sample_offsets():
    source = torch.arange(1, 65, dtype=torch.float32)
    trimmed = evaluator(output_start_seconds=2)
    trimmed.add_original(source)
    trimmed.add_anonymized(source[16:])
    results = trimmed.poll()
    assert [r.status for r in results] == ["unavailable", "ok", "ok"]
    torch.testing.assert_close(trimmed.encoder.calls[0], source[16:48])
    torch.testing.assert_close(trimmed.encoder.calls[1], source[16:48])
    delayed = evaluator(output_start_seconds=-1)
    delayed.add_original(source[:32])
    delayed.add_anonymized(torch.cat((torch.zeros(8), source[:24])))
    assert delayed.poll() == []
    delayed.add_anonymized(source[24:32])
    assert delayed.poll()[0].speaker_similarity == pytest.approx(1)
    torch.testing.assert_close(delayed.encoder.calls[0], delayed.encoder.calls[1])


def test_finish_insufficient_and_missing_audio():
    core = evaluator()
    core.add_original(torch.ones(31))
    core.add_anonymized(torch.ones(32))
    core.finish()
    assert core.poll() == []
    core.reset()
    core.add_original(torch.ones(48))
    core.add_anonymized(torch.ones(32))
    assert core.poll()[0].status == "ok"
    assert core.poll() == []
    core.finish()
    result = core.poll()[0]
    assert result.window_index == 1 and result.status == "unavailable"
    with pytest.raises(RuntimeError, match="finish"):
        core.add_original(torch.ones(1))


@pytest.mark.parametrize("vector", [torch.zeros(2), torch.tensor([float("nan"), 1]), torch.tensor([float("inf"), 1])])
def test_invalid_embeddings(vector):
    core = evaluator(MockEncoder(vector))
    core.add_original(torch.ones(32))
    core.add_anonymized(torch.ones(32))
    result = core.poll()[0]
    assert result.status == "error" and result.speaker_similarity is None
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("bad_audio", [torch.zeros(32), torch.full((32,), float("nan"))])
def test_silence_and_nonfinite_audio(bad_audio):
    core = evaluator()
    core.add_original(bad_audio)
    core.add_anonymized(torch.ones(32))
    result = core.poll()[0]
    assert result.status == "error" and result.speaker_similarity is None
    assert not core.encoder.calls


def test_standard_cosine_can_be_negative():
    class SignedEncoder(MockEncoder):
        def encode(self, audio, sr):
            return SpeakerEmbedding(torch.tensor([float(audio.mean()), 0]), 2, "mock", "signed")

    core = evaluator(SignedEncoder())
    core.add_original(torch.ones(32))
    core.add_anonymized(-torch.ones(32))
    assert core.poll()[0].speaker_similarity == pytest.approx(-1)


def test_buffer_overrun_is_unavailable_and_memory_is_bounded():
    core = evaluator(max_buffer_seconds=4)
    core.add_original(torch.ones(80))
    assert core.original.values.numel() == 32
    core.add_anonymized(torch.ones(80))
    results = core.poll()
    assert [r.status for r in results] == ["unavailable"] * 3 + ["ok"]
    assert core.anonymized.values.numel() <= 32


def test_reset_and_reproducibility():
    core = evaluator()

    def run():
        core.add_original(torch.ones(32))
        core.add_anonymized(torch.ones(32))
        return core.poll()[0]

    first = run()
    core.reset("new")
    second = run()
    assert second.window_index == 0 and second.session_id == "new"
    assert replace(first, session_id="new", compute_time_ms=0) == replace(second, compute_time_ms=0)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.add_original(torch.ones(1))


def test_async_overload_order_result_bound_and_reset_inflight():
    entered, release = threading.Event(), threading.Event()

    class BlockingEncoder(MockEncoder):
        def encode(self, audio, sr):
            entered.set()
            assert release.wait(5), "test failed to release inference"
            return super().encode(audio, sr)

    core = evaluator(BlockingEncoder())
    worker = AsyncRollingSpeakerSimilarity(core, max_pending=1, max_results=1)
    try:
        worker.add_original(torch.ones(32))
        worker.add_anonymized(torch.ones(32))
        assert entered.wait(5)
        # These callbacks must complete while inference is blocked.
        worker.add_original(torch.ones(32))
        worker.add_anonymized(torch.ones(32))
        assert worker.skipped_windows == 1
        assert worker.poll() == []
        release.set()
        assert worker.wait(5)
        assert worker.dropped_results == 1
        assert [r.window_index for r in worker.poll()] == [2]
        release.clear()
        entered.clear()
        worker.add_original(torch.ones(16))
        worker.add_anonymized(torch.ones(16))
        assert entered.wait(5)
        worker.reset("replacement")
        worker.add_original(torch.ones(32))
        worker.add_anonymized(torch.ones(32))
        release.set()
        assert worker.wait(5)
        results = worker.poll()
        assert [(r.session_id, r.window_index) for r in results] == [("replacement", 0)]
        assert worker.skipped_windows == 0
    finally:
        release.set()
        worker.close()


def test_campplus_pooled_contract_without_downloads(monkeypatch):
    from quick_convert.components.encoders.speaker import CosyVoiceCAMPPlusSpeakerEncoder

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **kw: "unused.onnx"))

    class Extractor:
        def __init__(self, model_path):
            assert model_path == "unused.onnx"

        def inference(self, waveform):
            assert waveform.shape == (1, 64000)
            return torch.ones(192)

    monkeypatch.setitem(
        sys.modules, "quick_convert.external.cosyvoice.utils.onnx", SimpleNamespace(EmbeddingExtractor=Extractor)
    )
    encoder = CosyVoiceCAMPPlusSpeakerEncoder(device="cpu")
    core = RollingSpeakerSimilarity(encoder, 16000, 16000, aligned=True)
    core.add_original(torch.ones(64000))
    core.add_anonymized(torch.ones(64000))
    result = core.poll()[0]
    assert result.status == "ok" and result.speaker_similarity == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": 0},
        {"hop_seconds": float("nan")},
        {"max_buffer_seconds": 1},
        {"window_seconds": 0.01},
        {"output_start_seconds": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        evaluator(**kwargs)


def test_nonmono_and_non_cpu_inputs_rejected():
    core = evaluator()
    with pytest.raises(ValueError, match="mono"):
        core.add_original(torch.ones(2, 32))
    with pytest.raises(ValueError, match="CPU"):
        core.add_original(torch.empty(32, device="meta"))
