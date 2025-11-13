from types import SimpleNamespace

import pytest

from splitter_mr.schema import (
    InvalidChunkException,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutputException,
    SplitterOutputWarning,
)
from splitter_mr.schema.models import ReaderOutput, SplitterOutput
from splitter_mr.splitter.splitters.paged_splitter import PagedSplitter

# ---- Fixtures, helpers and mocks ---- #


def make_reader_output(
    text,
    page_placeholder="<!-- page -->",
    document_name="doc.md",
    document_path="tmp/doc.md",
    document_id=None,
    conversion_method=None,
    reader_method=None,
    ocr_method=None,
):
    return ReaderOutput(
        text=text,
        document_name=document_name,
        document_path=document_path,
        document_id=document_id,
        conversion_method=conversion_method,
        reader_method=reader_method,
        ocr_method=ocr_method,
        page_placeholder=page_placeholder,
    )


# ---- Tests cases ---- #


def test_init_valid():
    s = PagedSplitter(chunk_size=2, chunk_overlap=3)
    assert s.chunk_size == 2
    assert s.chunk_overlap == 3


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_init_chunk_size_invalid(chunk_size):
    with pytest.raises(SplitterConfigException, match="chunk_size must be"):
        PagedSplitter(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, -100])
def test_init_chunk_overlap_invalid(chunk_overlap):
    with pytest.raises(SplitterConfigException, match="chunk_overlap must be"):
        PagedSplitter(chunk_overlap=chunk_overlap)


def test_split_basic_pages():
    text = "<!-- page --> First <!-- page --> Second <!-- page --> Third"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    # Three pages, each a chunk
    assert out.chunks == ["First", "Second", "Third"]
    assert out.split_method == "paged_splitter"
    assert out.split_params["chunk_size"] == 1


def test_split_multiple_pages_per_chunk():
    text = "<!-- page --> 1 <!-- page --> 2 <!-- page --> 3 <!-- page --> 4"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=2)
    out = splitter.split(ro)
    # Should group: [1 + 2], [3 + 4]
    assert out.chunks == ["1\n2", "3\n4"]


def test_split_with_overlap():
    text = "<!-- page --> abcde <!-- page --> fghij <!-- page --> klmno"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1, chunk_overlap=2)
    out = splitter.split(ro)
    # overlap 2 chars from previous chunk
    # 1st chunk: "abcde"
    # 2nd chunk: last 2 chars of "abcde" + "fghij" = "de" + "fghij" = "defghij"
    # 3rd chunk: last 2 chars of "defghij" + "klmno" = "ij" + "klmno" = "ijklmno"
    assert out.chunks == ["abcde", "defghij", "ijklmno"]


def test_split_removes_empty_pages():
    text = "<!-- page --> foo <!-- page -->   <!-- page --> bar <!-- page -->"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    # empty pages between markers should be ignored
    assert out.chunks == ["foo", "bar"]


def test_split_handles_leading_trailing_whitespace():
    text = "  <!-- page -->   page1   <!-- page -->   page2   "
    ro = make_reader_output(text)
    splitter = PagedSplitter()
    out = splitter.split(ro)
    # whitespace should be stripped from chunks
    assert out.chunks == ["page1", "page2"]


def test_split_with_missing_placeholder_raises():
    text = "Just a plain text with no pages"
    ro = make_reader_output(text, page_placeholder="")
    splitter = PagedSplitter()
    with pytest.raises(
        ReaderOutputException, match="page_placeholder must be a non-empty string"
    ):
        splitter.split(ro)


def test_split_returns_correct_metadata():
    text = "<!-- page --> X <!-- page --> Y"
    ro = make_reader_output(text, document_name="abc", document_path="zzz")
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    assert out.document_name == "abc"
    assert out.document_path == "zzz"
    assert out.split_method == "paged_splitter"
    # check ids (chunk_id is generated, so just check length matches)
    assert len(out.chunks) == len(out.chunk_id)


def test_split_output_is_splitteroutput():
    text = "<!-- page --> hi"
    ro = make_reader_output(text)
    splitter = PagedSplitter()
    out = splitter.split(ro)
    assert isinstance(out, SplitterOutput)


# ---- Error handling ---- #


def test_split_missing_text_attribute_raises_reader_output_exception():
    # Use a simple object without 'text' attribute
    bogus = SimpleNamespace(page_placeholder="<!-- page -->")
    splitter = PagedSplitter()
    with pytest.raises(ReaderOutputException, match="must expose a 'text' attribute"):
        splitter.split(bogus)  # type: ignore[arg-type]


def test_split_empty_text_emits_input_and_output_warnings():
    ro = make_reader_output(text="   ")
    splitter = PagedSplitter()

    with pytest.warns((SplitterInputWarning, SplitterOutputWarning)) as record:
        out = splitter.split(ro)

    # We expect at least one input warning and one output warning recorded
    input_warns = [w for w in record if isinstance(w.message, SplitterInputWarning)]
    output_warns = [w for w in record if isinstance(w.message, SplitterOutputWarning)]
    assert input_warns, "Expected a SplitterInputWarning for empty text"
    assert output_warns, "Expected a SplitterOutputWarning for no non-empty pages"
    # Still returns a single empty-ish chunk as fallback behaviour
    assert out.chunks == [""]


def test_split_no_non_empty_pages_triggers_output_warning_and_returns_empty_chunk():
    # All pages are empty/whitespace after splitting on placeholder
    text = "<!-- page -->   <!-- page -->   "
    ro = make_reader_output(text=text)
    splitter = PagedSplitter()

    with pytest.warns(SplitterOutputWarning, match="did not find any non-empty pages"):
        out = splitter.split(ro)

    assert out.chunks == [""]


def test_split_wraps_build_output_errors_in_splitter_output_exception(monkeypatch):
    text = "<!-- page --> foo"
    ro = make_reader_output(text=text)
    splitter = PagedSplitter()

    def boom_build_output(
        _self, _reader_output, _chunks
    ):  # signature matches _build_output
        raise TypeError("boom")

    monkeypatch.setattr(
        splitter,
        "_build_output",
        boom_build_output,  # type: ignore[assignment]
    )

    with pytest.raises(SplitterOutputException) as excinfo:
        splitter.split(ro)

    assert "Failed to build SplitterOutput in PagedSplitter" in str(excinfo.value)


def test_split_mismatched_chunk_ids_raises_invalid_chunk_exception(monkeypatch):
    text = "<!-- page --> page1 <!-- page --> page2"
    ro = make_reader_output(text=text)
    splitter = PagedSplitter(chunk_size=1)

    # Force mismatch: always return a single ID, signature must be (n), not (self, n)
    monkeypatch.setattr(
        splitter,
        "_generate_chunk_ids",
        lambda _n: ["only-one-id"],  # type: ignore[method-assign]
    )

    with pytest.raises(
        InvalidChunkException, match="Number of chunk IDs does not match"
    ):
        splitter.split(ro)
