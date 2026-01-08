import warnings

import pytest
from pydantic_core import ValidationError

from splitter_mr.schema import ReaderOutput
from splitter_mr.schema.exceptions import InvalidChunkException, SplitterConfigException
from splitter_mr.schema.warnings import ChunkUnderflowWarning, SplitterInputWarning
from splitter_mr.splitter import WordSplitter

# ---- Mocks, fixtures and helpers ---- #


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text="The quick brown fox jumps over the lazy dog and runs away",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        reader_method="plain",
        ocr_method=None,
        metadata={},
    )


# ---- Test cases ---- #


def test_basic_split(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert hasattr(result, "chunks")
    assert result.chunks == [
        "The quick brown fox",
        "jumps over the lazy",
        "dog and runs away",
    ]
    assert result.split_method == "word_splitter"
    assert result.split_params["chunk_size"] == 4
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=2)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "The quick brown fox"
    assert result.chunks[1] == "brown fox jumps over"
    assert result.chunks[2] == "jumps over the lazy"
    assert result.chunks[3] == "the lazy dog and"
    assert result.chunks[4] == "dog and runs away"


def test_split_with_overlap_float(reader_output):
    splitter = WordSplitter(chunk_size=6, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "The quick brown fox jumps over"
    assert result.chunks[1] == "fox jumps over the lazy dog"
    assert result.chunks[2] == "the lazy dog and runs away"


def test_chunk_overlap_equals_chunk_size_raises(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=4)
    with pytest.raises(SplitterConfigException) as exc_info:
        splitter.split(reader_output)
    assert "chunk_overlap must be smaller than chunk_size" in str(exc_info.value)


def test_negative_chunk_overlap_raises(reader_output):
    with pytest.raises(
        SplitterConfigException, match="chunk_overlap cannot be negative"
    ):
        WordSplitter(chunk_size=4, chunk_overlap=-1)


def test_float_chunk_overlap_out_of_range_raises(reader_output):
    with pytest.raises(
        SplitterConfigException,
        match="When chunk_overlap is a float, it must be between 0 and 1",
    ):
        WordSplitter(chunk_size=4, chunk_overlap=1.1)

    with pytest.raises(SplitterConfigException):
        WordSplitter(chunk_size=4, chunk_overlap=-0.1)


def test_invalid_chunk_overlap_type_raises(reader_output):
    with pytest.raises(
        SplitterConfigException,
        match="chunk_overlap must be an int or float",
    ):
        WordSplitter(chunk_size=4, chunk_overlap="bad")


def test_invalid_chunk_size_raises():
    with pytest.raises(
        SplitterConfigException,
        match="chunk_size must be a positive integer",
    ):
        WordSplitter(chunk_size=0, chunk_overlap=0)


def test_output_contains_metadata(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=0)
    result = splitter.split(reader_output)
    for field in [
        "chunks",
        "chunk_id",
        "document_name",
        "document_path",
        "document_id",
        "conversion_method",
        "reader_method",
        "ocr_method",
        "split_method",
        "split_params",
        "metadata",
    ]:
        assert hasattr(result, field)


def test_empty_text_warns_and_validation_error():
    splitter = WordSplitter(chunk_size=5, chunk_overlap=0)
    reader_output = ReaderOutput(
        text="",
        document_name="empty.txt",
        document_path="/tmp/empty.txt",
        document_id="empty",
        conversion_method="text",
        reader_method="plain",
        ocr_method=None,
        metadata={},
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # SplitterOutput will reject empty chunks -> ValidationError
        with pytest.raises(ValidationError):
            splitter.split(reader_output)

    categories = {warn.category for warn in w}
    assert any(issubclass(cat, SplitterInputWarning) for cat in categories), (
        "Expected SplitterInputWarning for empty text"
    )
    assert any(issubclass(cat, ChunkUnderflowWarning) for cat in categories), (
        "Expected ChunkUnderflowWarning when no chunks are produced"
    )


def test_invalid_chunk_ids_raise_invalid_chunk_exception(monkeypatch, reader_output):
    """Force a mismatch between chunks and chunk_ids to hit InvalidChunkException."""
    splitter = WordSplitter(chunk_size=4, chunk_overlap=0)

    # Fake method must accept (self, n)
    def fake_generate_chunk_ids(self, n: int):
        return ["only-one-id"]

    # Patch on the class so it becomes a bound method
    monkeypatch.setattr(
        "splitter_mr.splitter.splitters.word_splitter.WordSplitter._generate_chunk_ids",
        fake_generate_chunk_ids,
    )

    with pytest.raises(InvalidChunkException) as exc_info:
        splitter.split(reader_output)

    assert "Chunk ID generation mismatch" in str(exc_info.value)
