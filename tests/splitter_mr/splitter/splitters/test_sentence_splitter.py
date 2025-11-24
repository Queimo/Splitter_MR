import warnings
from types import SimpleNamespace

import pytest

from splitter_mr.schema import (
    ChunkUnderflowWarning,
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutputException,
    SplitterOutputWarning,
)
from splitter_mr.splitter import SentenceSplitter

# ---- Mocks, fixtures and helpers ---- #


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text=(
            "Hello world! How are you? I am fine. "
            "Testing sentence splitting. "
            "Short. End! And another?"
        ),
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
        metadata={},
    )


# ---- Test cases ---- #


def test_basic_split(reader_output):
    splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "Hello world! How are you? I am fine."
    assert result.chunks[1] == "Testing sentence splitting. Short. End!"
    assert result.chunks[2] == "And another?"
    assert result.split_method == "sentence_splitter"
    assert result.split_params["chunk_size"] == 3
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=2)
    result = splitter.split(reader_output)
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    first_words = first_chunk.split()[-2:]
    assert " ".join(first_words) in second_chunk


def test_split_with_overlap_float(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    if len(result.chunks) > 1:
        prev_words = result.chunks[0].split()
        overlap = set(prev_words) & set(result.chunks[1].split())
        assert len(overlap) >= 1


def test_separator_variants():
    text = "A|B|C|D"
    ro = ReaderOutput(text=text, document_path="/tmp/sample.txt")
    # IMPORTANT: pass a literal separator as a LIST (or escape the regex as r"\|")
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0, separators=["|"])
    result = splitter.split(ro)
    assert result.chunks[0] == "A| B|"
    assert result.chunks[1] == "C| D"


def test_output_contains_metadata(reader_output):
    splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
    result = splitter.split(reader_output)
    for field in [
        "chunks",
        "chunk_id",
        "document_name",
        "document_path",
        "document_id",
        "conversion_method",
        "ocr_method",
        "split_method",
        "split_params",
        "metadata",
    ]:
        assert hasattr(result, field)


# ---- Error and warning handling ---- #

# Config validation tests


def test_invalid_chunk_size_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_size=0)

    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_size=-1)

    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_size=1.5)  # not an int


def test_negative_chunk_overlap_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_overlap=-0.1)

    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_overlap=-1)


def test_non_numeric_chunk_overlap_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(chunk_overlap="not-a-number")  # type: ignore[arg-type]


def test_invalid_separators_type_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(separators=123)  # type: ignore[arg-type]


def test_empty_separator_string_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(separators="")


def test_invalid_separators_list_raises_splitter_config_exception():
    # Empty list
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(separators=[])

    # List with invalid entries (non-str or empty)
    with pytest.raises(SplitterConfigException):
        SentenceSplitter(separators=[".", ""])  # empty string is invalid

    with pytest.raises(SplitterConfigException):
        SentenceSplitter(separators=[".", 123])  # type: ignore[list-item]


# ReaderOutput validation tests


def test_missing_text_attribute_raises_reader_output_exception():
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
    ro = SimpleNamespace(document_path="/tmp/sample.txt")  # no `text` attr

    with pytest.raises(ReaderOutputException):
        splitter.split(ro)  # type: ignore[arg-type]


def test_non_string_text_raises_reader_output_exception():
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)

    # Fake ReaderOutput-like object that bypasses Pydantic validation
    ro = SimpleNamespace(
        text=123,
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method=None,
        reader_method=None,
        ocr_method=None,
    )

    with pytest.raises(ReaderOutputException):
        splitter.split(ro)


# Warning behaviour tests


def test_empty_text_emits_input_and_output_warnings():
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
    ro = ReaderOutput(text="", document_path="/tmp/sample.txt")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = splitter.split(ro)

    assert out.chunks == [""]
    _ = {issubclass(wi.category, SplitterInputWarning) for wi in w}
    # At least one SplitterInputWarning
    assert any(issubclass(wi.category, SplitterInputWarning) for wi in w)
    # At least one SplitterOutputWarning
    assert any(issubclass(wi.category, SplitterOutputWarning) for wi in w)


def test_chunk_underflow_emits_warning(reader_output):
    # chunk_size is larger than the number of produced chunks
    splitter = SentenceSplitter(chunk_size=10, chunk_overlap=0)

    with pytest.warns(ChunkUnderflowWarning):
        result = splitter.split(reader_output)

    # We still get at least one valid chunk
    assert len(result.chunks) >= 1


# Output integrity / exception wrapping tests


def test_mismatched_chunk_ids_raises_invalid_chunk_exception(
    monkeypatch, reader_output
):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)

    # Force _generate_chunk_ids to generate the wrong number of IDs
    def fake_generate_chunk_ids(self, n: int):
        return ["only-one-id"]  # length 1 regardless of n

    monkeypatch.setattr(
        "splitter_mr.splitter.splitters.sentence_splitter.SentenceSplitter._generate_chunk_ids",
        fake_generate_chunk_ids,
    )

    with pytest.raises(InvalidChunkException):
        splitter.split(reader_output)


def test_splitter_output_exception_wrapped(monkeypatch, reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)

    class FakeOutput:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom in SplitterOutput")

    # Patch the SplitterOutput used inside the module to raise
    monkeypatch.setattr(
        "splitter_mr.splitter.splitters.sentence_splitter.SplitterOutput",
        FakeOutput,
    )

    with pytest.raises(SplitterOutputException) as excinfo:
        splitter.split(reader_output)

    assert "SentenceSplitter" in str(excinfo.value)
