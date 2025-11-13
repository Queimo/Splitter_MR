import json
from unittest.mock import patch

import pytest

from splitter_mr.schema import (
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterOutputException,
)
from splitter_mr.splitter import RecursiveJSONSplitter

# ---- Helpers, mocks and fixtures ---- #


@pytest.fixture
def reader_output():
    data = {"foo": {"bar": [1, 2, 3]}, "baz": "qux"}
    return ReaderOutput(
        text=json.dumps(data),
        document_name="sample.json",
        document_path="/tmp/sample.json",
        document_id="123",
        conversion_method="json",
        ocr_method=None,
    )


# ---- Test cases ---- #


def test_recursive_json_splitter_instantiates_and_calls_splitter(reader_output):
    with patch(
        "splitter_mr.splitter.splitters.json_splitter.RecursiveJsonSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.split_text.return_value = [
            '{"foo": {"bar": [1, 2]}}',
            '{"foo": {"bar": [3]}, "baz": "qux"}',
        ]
        splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)
        result = splitter.split(reader_output)

        MockSplitter.assert_called_once_with(max_chunk_size=100, min_chunk_size=90)
        mock_splitter.split_text.assert_called_once_with(
            json_data=json.loads(reader_output.text), convert_lists=True
        )

        # Check output structure and values
        assert hasattr(result, "chunks")
        assert result.chunks == [
            '{"foo": {"bar": [1, 2]}}',
            '{"foo": {"bar": [3]}, "baz": "qux"}',
        ]
        assert hasattr(result, "split_method")
        assert result.split_method == "recursive_json_splitter"
        assert result.split_params["max_chunk_size"] == 100
        assert result.split_params["min_chunk_size"] == 10
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


def test_empty_text():
    with patch(
        "splitter_mr.splitter.splitters.json_splitter.RecursiveJsonSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.split_json.return_value = []
        splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)
        reader_output = ReaderOutput(text=json.dumps({}))
        with pytest.raises(SplitterOutputException):
            splitter.split(reader_output)


# ---- Exception handling ---- #


def test_recursive_json_splitter_invalid_chunk_size_type_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException, match="chunk_size"):
        RecursiveJSONSplitter(chunk_size="100", min_chunk_size=10)


def test_recursive_json_splitter_invalid_min_chunk_size_type_raises_splitter_config_exception():
    with pytest.raises(SplitterConfigException, match="min_chunk_size"):
        RecursiveJSONSplitter(chunk_size=100, min_chunk_size="10")


def test_recursive_json_splitter_invalid_json_raises_reader_output_exception():
    bad_reader_output = ReaderOutput(
        text="{ invalid json ",
        document_name="bad.json",
        document_path="/tmp/bad.json",
        document_id="bad-id",
        conversion_method="json",
        ocr_method=None,
    )
    splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)

    with pytest.raises(ReaderOutputException, match="valid JSON"):
        splitter.split(bad_reader_output)


def test_empty_text_produces_void_chunks_and_raises_invalid_chunk_exception():
    with patch(
        "splitter_mr.splitter.splitters.json_splitter.RecursiveJsonSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.split_text.return_value = []
        splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)

        reader_output = ReaderOutput(
            text=json.dumps({}),
            document_name="empty.json",
            document_path="/tmp/empty.json",
            document_id="empty-id",
            conversion_method="json",
            ocr_method=None,
        )

        with pytest.raises(InvalidChunkException, match="void or missing chunks"):
            splitter.split(reader_output)


def test_recursive_json_splitter_runtime_error_from_underlying_splitter_raises_invalid_chunk_exception(
    reader_output,
):
    with patch(
        "splitter_mr.splitter.splitters.json_splitter.RecursiveJsonSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.split_text.side_effect = RuntimeError("boom")

        splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)

        with pytest.raises(InvalidChunkException, match="error trying to split"):
            splitter.split(reader_output)


def test_recursive_json_splitter_build_output_failure_raises_splitter_output_exception(
    reader_output, monkeypatch
):
    # Force SplitterOutput to blow up when called
    def boom_constructor(*_args, **_kwargs):
        raise RuntimeError("output boom")

    monkeypatch.setattr(
        "splitter_mr.splitter.splitters.json_splitter.SplitterOutput",
        boom_constructor,
    )

    splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=10)

    with pytest.raises(SplitterOutputException, match="build SplitterOutput response"):
        splitter.split(reader_output)
