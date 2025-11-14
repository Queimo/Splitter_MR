from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.schema import (
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutputException,
    SplitterOutputWarning,
)
from splitter_mr.splitter import RecursiveCharacterSplitter

# ---- Mocks, Fixtures & Helpers ---- #


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text="A long test text that should be split recursively.",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
    )


# ---- Test cases ---- #


def test_recursive_character_splitter_instantiates_and_calls_splitter(reader_output):
    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        # Setup the mock to return fake chunks as page_content
        mock_splitter = MockSplitter.return_value
        mock_doc1 = MagicMock(page_content="Chunk 1")
        mock_doc2 = MagicMock(page_content="Chunk 2")
        mock_splitter.create_documents.return_value = [mock_doc1, mock_doc2]

        splitter = RecursiveCharacterSplitter(
            chunk_size=10, chunk_overlap=2, separators=["."]
        )
        result = splitter.split(reader_output)

        # Check instantiation
        MockSplitter.assert_called_once_with(
            chunk_size=10, chunk_overlap=2, separators=["."]
        )
        # Check method called
        mock_splitter.create_documents.assert_called_once_with([reader_output.text])

        # Check output structure
        assert hasattr(result, "chunks")
        assert result.chunks == ["Chunk 1", "Chunk 2"]
        assert hasattr(result, "split_method")
        assert result.split_method == "recursive_character_splitter"
        assert result.split_params["chunk_size"] == 10
        assert result.split_params["chunk_overlap"] == 2
        assert result.split_params["separators"] == ["."]
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


# ---- Error 6 Warning Handling ---- #


def test_empty_text_emits_warnings_and_returns_single_empty_chunk():
    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        # No documents produced
        mock_splitter.create_documents.return_value = []

        splitter = RecursiveCharacterSplitter(
            chunk_size=10, chunk_overlap=0, separators=["."]
        )
        ro = ReaderOutput(text="")

        # Expect both: input warning (empty text) and output warning (no chunks)
        with pytest.warns((SplitterInputWarning, SplitterOutputWarning)) as record:
            result = splitter.split(ro)

        input_warns = [w for w in record if isinstance(w.message, SplitterInputWarning)]
        output_warns = [
            w for w in record if isinstance(w.message, SplitterOutputWarning)
        ]
        assert input_warns
        assert output_warns

        # Fallback behaviour: single empty chunk
        assert result.chunks == [""]
        assert len(result.chunk_id) == 1


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_init_invalid_chunk_size_raises_splitter_config_exception(chunk_size):
    with pytest.raises(SplitterConfigException, match="chunk_size must be"):
        RecursiveCharacterSplitter(chunk_size=chunk_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_overlap", [-1, -0.1, "bad"])
def test_init_invalid_chunk_overlap_raises_splitter_config_exception(chunk_overlap):
    with pytest.raises(SplitterConfigException, match="chunk_overlap must be"):
        RecursiveCharacterSplitter(chunk_overlap=chunk_overlap)  # type: ignore[arg-type]


@pytest.mark.parametrize("separators", [123, 1.5, None])
def test_init_invalid_separators_type_raises_splitter_config_exception(separators):
    with pytest.raises(SplitterConfigException, match="separators must be a string"):
        RecursiveCharacterSplitter(separators=separators)  # type: ignore[arg-type]


def test_init_invalid_separators_contents_raises_splitter_config_exception():
    # Empty string or non-string entries should be rejected
    with pytest.raises(SplitterConfigException, match="separators must contain"):
        RecursiveCharacterSplitter(separators=["", "."])

    with pytest.raises(SplitterConfigException, match="separators must contain"):
        RecursiveCharacterSplitter(separators=[".", 123])  # type: ignore[list-item]


def test_init_overlap_greater_or_equal_chunk_size_raises_splitter_config_exception():
    # Effective overlap >= chunk_size -> invalid
    with pytest.raises(
        SplitterConfigException, match="must be smaller than chunk_size"
    ):
        RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=10)

    # Fractional overlap; 1.0 * 10 = 10 -> also invalid
    with pytest.raises(
        SplitterConfigException, match="must be smaller than chunk_size"
    ):
        RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=1.0)


def test_split_missing_text_attribute_raises_reader_output_exception():
    splitter = RecursiveCharacterSplitter()
    bogus = SimpleNamespace()  # no 'text' attribute
    with pytest.raises(
        ReaderOutputException,
        match="ReaderOutput object must expose a 'text' attribute.",
    ):
        splitter.split(bogus)  # type: ignore[arg-type]


def test_split_non_string_text_raises_reader_output_exception():
    splitter = RecursiveCharacterSplitter()
    ro = ReaderOutput(text="valid text")
    # Bypass pydantic by mutating after creation
    ro.text = 123  # type: ignore[assignment]

    with pytest.raises(
        ReaderOutputException,
        match=r"ReaderOutput\.text must be of type 'str' or None",
    ):
        splitter.split(ro)


def test_split_mismatched_chunk_ids_raises_invalid_chunk_exception(
    monkeypatch, reader_output
):
    # Patch _generate_chunk_ids to force a mismatch
    splitter = RecursiveCharacterSplitter(
        chunk_size=10, chunk_overlap=0, separators=["."]
    )

    monkeypatch.setattr(
        splitter,
        "_generate_chunk_ids",
        lambda _n: ["only-one-id"],  # type: ignore[method-assign]
    )

    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.create_documents.return_value = [
            MagicMock(page_content="Chunk 1"),
            MagicMock(page_content="Chunk 2"),
        ]

        with pytest.raises(
            InvalidChunkException, match="Number of chunk IDs does not match"
        ):
            splitter.split(reader_output)


def test_split_wraps_langchain_errors_in_splitter_output_exception(reader_output):
    splitter = RecursiveCharacterSplitter(
        chunk_size=10, chunk_overlap=0, separators=["."]
    )

    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.create_documents.side_effect = RuntimeError("boom")

        with pytest.raises(SplitterOutputException) as excinfo:
            splitter.split(reader_output)

    assert "RecursiveCharacterTextSplitter failed during split" in str(excinfo.value)


def test_split_wraps_splitteroutput_construction_errors_in_splitter_output_exception(
    reader_output, monkeypatch
):
    splitter = RecursiveCharacterSplitter(
        chunk_size=10, chunk_overlap=0, separators=["."]
    )

    # Patch SplitterOutput in the module where RecursiveCharacterSplitter is defined
    from splitter_mr.splitter.splitters import recursive_splitter as rs_mod

    class BoomSplitterOutput:
        def __init__(self, *args, **kwargs):
            raise TypeError("boom")

    monkeypatch.setattr(rs_mod, "SplitterOutput", BoomSplitterOutput)

    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.create_documents.return_value = [
            MagicMock(page_content="Chunk 1"),
        ]

        with pytest.raises(SplitterOutputException) as excinfo:
            splitter.split(reader_output)

    assert "Failed to build SplitterOutput in RecursiveCharacterSplitter" in str(
        excinfo.value
    )
