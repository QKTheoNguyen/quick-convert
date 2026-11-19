"""Sample-clock rolling speaker similarity, independent of workflow orchestration."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from quick_convert.components.encoders.speaker.base import SpeakerEncoder


@dataclass(frozen=True)
class SpeakerSimilarityResult:
    session_id: str
    window_index: int
    window_start_seconds: float
    window_end_seconds: float
    speaker_similarity: float | None
    speaker_encoder: str
    input_sample_rate: int
    output_sample_rate: int
    compute_time_ms: float
    status: str
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-compatible primitives; unavailable scores are null."""
        return asdict(self)


@dataclass
class _Window:
    result: SpeakerSimilarityResult
    original: torch.Tensor | None = None
    anonymized: torch.Tensor | None = None


class _AudioBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.values = torch.empty(0)
        self.start = 0
        self.end = 0
        self.finished = False

    def add(self, chunk: torch.Tensor):
        if self.finished:
            raise RuntimeError("Cannot append after finish().")
        if not isinstance(chunk, torch.Tensor) or chunk.device.type != "cpu":
            raise ValueError("Chunks must be CPU tensors; transfer audio outside the callback.")
        if chunk.ndim == 2 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 1 or chunk.is_complex():
            raise ValueError("Expected mono audio [T] or [1,T].")
        # Copy only a bounded suffix, never retaining caller-owned storage.
        tail = chunk[-self.capacity :].detach().to(dtype=torch.float32).clone()
        self.end += chunk.numel()
        self.values = torch.cat((self.values, tail))[-self.capacity :].clone()
        self.start = self.end - self.values.numel()

    def get(self, start: int, end: int):
        if start < self.start or end > self.end or start < 0:
            return None
        return self.values[start - self.start : end - self.start].clone()

    def discard_before(self, sample: int):
        count = min(max(sample - self.start, 0), self.values.numel())
        self.values = self.values[count:].clone()
        self.start += count


class RollingSpeakerSimilarity:
    """One session, synchronous core; ``poll`` performs embedding inference.

    ``aligned=True`` asserts equal time scale with a constant offset:
    output sample zero corresponds to ``output_start_seconds`` on the original
    clock. Positive offsets describe trimmed leading original speech; negative
    offsets describe leading output latency/padding. Wall-clock latency alone
    requires no offset. Leave ``aligned=False`` for unknown/time-varying alignment;
    complete windows then report unavailable and never invoke the encoder.

    Window/hop durations must be integral original sample counts. Output bounds
    are rounded on its native sample clock. Resampling is applied to complete
    windows, so chunk boundaries do not change preprocessing. No partial final
    windows or padding are scored. This core is single-owner, not thread-safe.
    The caller owns the encoder and must provide exclusive inference access.
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        input_sample_rate: int,
        output_sample_rate: int,
        *,
        session_id: str = "default",
        window_seconds: float = 4.0,
        hop_seconds: float = 2.0,
        aligned: bool = False,
        output_start_seconds: float = 0.0,
        max_buffer_seconds: float = 12.0,
    ):
        for rate in (input_sample_rate, output_sample_rate, encoder.sample_rate):
            if type(rate) is not int or rate <= 0:
                raise ValueError("Sample rates must be positive integers.")
        for value in (window_seconds, hop_seconds, max_buffer_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Window, hop, and buffer durations must be finite and positive.")
        if not math.isfinite(output_start_seconds):
            raise ValueError("Output offset must be finite.")
        if max_buffer_seconds < window_seconds:
            raise ValueError("Buffer duration must cover at least one window.")
        if getattr(encoder, "output", "pooled") != "pooled":
            raise ValueError("Rolling similarity requires pooled speaker embeddings.")
        self.encoder = encoder
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.window_samples = self._sample_count(window_seconds, input_sample_rate)
        self.hop_samples = self._sample_count(hop_seconds, input_sample_rate)
        self.aligned = aligned
        self.output_offset = Fraction(str(output_start_seconds))
        self.capacity = (
            math.ceil(max_buffer_seconds * input_sample_rate),
            math.ceil(max_buffer_seconds * output_sample_rate),
        )
        self.closed = False
        self.reset(session_id)

    @staticmethod
    def _sample_count(seconds, rate):
        count = Fraction(str(seconds)) * rate
        if count.denominator != 1 or count < 1:
            raise ValueError("Window and hop must be integral input sample counts.")
        return int(count)

    def reset(self, session_id: str = "default"):
        """Discard audio and scheduling state, retaining the loaded encoder."""
        self._check_open()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string.")
        self.session_id = session_id
        self.original = _AudioBuffer(self.capacity[0])
        self.anonymized = _AudioBuffer(self.capacity[1])
        self.window_index = 0

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Evaluator is closed.")

    def add_original(self, chunk: torch.Tensor):
        self._check_open()
        self.original.add(chunk)

    def add_anonymized(self, chunk: torch.Tensor):
        self._check_open()
        self.anonymized.add(chunk)

    def finish(self):
        """Declare both streams ended; poll to report missing output windows."""
        self._check_open()
        self.original.finished = self.anonymized.finished = True

    def _take_window(self) -> _Window | None:
        self._check_open()
        start = self.window_index * self.hop_samples
        end = start + self.window_samples
        if self.original.end < end:
            return None
        out_start = round((Fraction(start, self.input_sample_rate) - self.output_offset) * self.output_sample_rate)
        out_end = round((Fraction(end, self.input_sample_rate) - self.output_offset) * self.output_sample_rate)
        if out_end > self.anonymized.end and not self.anonymized.finished:
            return None
        a = self.original.get(start, end)
        b = self.anonymized.get(out_start, out_end)
        error = None
        if not self.aligned:
            error = "Alignment unknown or duration/time scale changed; supply aligned constant-offset audio."
        elif a is None or b is None or out_end <= out_start:
            error = "Corresponding audio unavailable (trimming, buffer overrun, or end of stream)."
        result = SpeakerSimilarityResult(
            session_id=self.session_id,
            window_index=self.window_index,
            window_start_seconds=start / self.input_sample_rate,
            window_end_seconds=end / self.input_sample_rate,
            speaker_similarity=None,
            speaker_encoder=type(self.encoder).__name__,
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
            compute_time_ms=0.0,
            status="unavailable" if error else "pending",
            error_message=error,
        )
        self.window_index += 1
        next_start = self.window_index * self.hop_samples
        self.original.discard_before(next_start)
        self.anonymized.discard_before(
            round((Fraction(next_start, self.input_sample_rate) - self.output_offset) * self.output_sample_rate)
        )
        return _Window(result, a if not error else None, b if not error else None)

    @torch.inference_mode()
    def _evaluate(self, window: _Window) -> SpeakerSimilarityResult:
        if window.result.status == "unavailable":
            return window.result
        started = perf_counter()
        score = None
        status, error = "ok", None
        try:
            vectors = []
            for audio, rate in (
                (window.original, self.input_sample_rate),
                (window.anonymized, self.output_sample_rate),
            ):
                if not torch.isfinite(audio).all():
                    raise ValueError("Audio contains nonfinite samples.")
                if not torch.any(audio != 0):
                    raise ValueError("Silent window has no measurable speaker identity.")
                if rate != self.encoder.sample_rate:
                    from torchaudio.functional import resample

                    audio = resample(audio, rate, self.encoder.sample_rate)
                embedding = self.encoder.encode(audio, self.encoder.sample_rate).values
                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    embedding = embedding[0]
                embedding = embedding.detach().to(device="cpu", dtype=torch.float64)
                if (
                    embedding.ndim != 1
                    or embedding.numel() == 0
                    or not torch.isfinite(embedding).all()
                    or embedding.abs().max() == 0
                ):
                    raise ValueError("Speaker embedding must be a finite, nonzero vector.")
                # Rescaling preserves cosine while avoiding extreme vector norms.
                embedding = embedding / embedding.abs().max()
                vectors.append(embedding)
            if vectors[0].shape != vectors[1].shape:
                raise ValueError("Speaker embedding dimensions differ.")
            score = float(F.cosine_similarity(vectors[0], vectors[1], dim=0).clamp(-1, 1))
            if not math.isfinite(score):
                raise ValueError("Cosine score is nonfinite.")
        # Interchangeable backends can raise arbitrary errors; isolate them per window.
        except Exception as exc:  # noqa: BLE001
            score, status, error = None, "error", f"{type(exc).__name__}: {exc}"
        fields = window.result.to_dict()
        fields.update(
            speaker_similarity=score,
            status=status,
            error_message=error,
            compute_time_ms=(perf_counter() - started) * 1000,
        )
        return SpeakerSimilarityResult(**fields)

    def poll(self) -> list[SpeakerSimilarityResult]:
        """Compute newly ready windows exactly once, synchronously."""
        results = []
        while (window := self._take_window()) is not None:
            results.append(self._evaluate(window))
        return results

    def close(self):
        """Release session audio; the caller retains ownership of the encoder."""
        self.original.values = self.anonymized.values = torch.empty(0)
        self.closed = True


class AsyncRollingSpeakerSimilarity:
    """Bounded single-worker wrapper; callbacks never wait for model inference.

    Transfer exclusive ownership of an unused core to this wrapper. Producers
    briefly lock for bounded audio copies/window extraction. Only the worker
    accesses the model. Pending overload drops oldest windows; result overflow
    drops oldest results. Index gaps and counters expose loss. Reset invalidates
    old in-flight results without reloading or concurrently accessing the model.
    """

    def __init__(self, core: RollingSpeakerSimilarity, *, max_pending: int = 2, max_results: int = 128):
        if type(max_pending) is not int or type(max_results) is not int or min(max_pending, max_results) < 1:
            raise ValueError("Queue capacities must be positive integers.")
        core._check_open()
        self.core = core
        self.max_pending, self.max_results = max_pending, max_results
        self.pending = deque()
        self.results = deque()
        self.condition = threading.Condition()
        self.generation = 0
        self.skipped_windows = self.dropped_results = 0
        self.stopping = False
        self.active = False
        self.worker = threading.Thread(target=self._run, name="speaker-similarity", daemon=True)
        self.worker.start()

    def _enqueue(self):
        while (window := self.core._take_window()) is not None:
            if len(self.pending) == self.max_pending:
                self.pending.popleft()
                self.skipped_windows += 1
            self.pending.append((self.generation, window))
        self.condition.notify_all()

    def _add(self, method, chunk):
        with self.condition:
            if self.stopping:
                raise RuntimeError("Evaluator is closed.")
            method(chunk)
            self._enqueue()

    def add_original(self, chunk: torch.Tensor):
        self._add(self.core.add_original, chunk)

    def add_anonymized(self, chunk: torch.Tensor):
        self._add(self.core.add_anonymized, chunk)

    def finish(self):
        with self.condition:
            if self.stopping:
                raise RuntimeError("Evaluator is closed.")
            self.core.finish()
            self._enqueue()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.stopping)
                if not self.pending:
                    return
                generation, window = self.pending.popleft()
                self.active = True
            result = self.core._evaluate(window)
            with self.condition:
                self.active = False
                if generation == self.generation:
                    if len(self.results) == self.max_results:
                        self.results.popleft()
                        self.dropped_results += 1
                    self.results.append(result)
                self.condition.notify_all()

    def poll(self) -> list[SpeakerSimilarityResult]:
        """Return completed results without waiting for inference."""
        with self.condition:
            results = list(self.results)
            self.results.clear()
            return results

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for idle, only for offline callers/shutdown; return False on timeout."""
        with self.condition:
            return self.condition.wait_for(lambda: not self.pending and not self.active, timeout)

    def reset(self, session_id: str = "default"):
        with self.condition:
            if self.stopping:
                raise RuntimeError("Evaluator is closed.")
            self.core.reset(session_id)
            self.generation += 1
            self.pending.clear()
            self.results.clear()
            self.skipped_windows = self.dropped_results = 0

    def close(self):
        """Drain pending work and join the worker. Call outside the audio callback."""
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.worker.join()
        self.core.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
