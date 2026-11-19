"""Checks of the timed-word and PII-span contracts shared by ASR and PII tests."""


def assert_timed_words(words: list[dict], duration: float) -> None:
    previous_end = 0.0
    for word in words:
        assert set(word) == {"word", "start_s", "end_s"}
        assert isinstance(word["word"], str) and word["word"]
        assert previous_end <= word["start_s"] < word["end_s"] <= duration
        previous_end = word["end_s"]


def assert_pii_spans(spans: list[dict], words: list[dict]) -> None:
    starts = {word["start_s"] for word in words}
    ends = {word["end_s"] for word in words}
    assert [span["start_s"] for span in spans] == sorted(span["start_s"] for span in spans)
    for span in spans:
        assert set(span) == {"start_s", "end_s", "label", "confidence", "text"}
        assert span["start_s"] in starts and span["end_s"] in ends and span["start_s"] < span["end_s"]
        assert isinstance(span["label"], str) and span["text"]
        assert 0.0 <= span["confidence"] <= 1.0
