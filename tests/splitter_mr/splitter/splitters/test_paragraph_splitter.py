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
from splitter_mr.schema.models import ReaderOutput
from splitter_mr.splitter.splitters.paragraph_splitter import ParagraphSplitter

# ---- Mocks, fixtures and helpers ---- #


@pytest.fixture
def reader_output():
    # 5 paragraphs, mixed line breaks
    return ReaderOutput(
        text=(
            "Para1 first sentence. Para1 second sentence.\n"
            "Para2 here.\n"
            "Para3 is this line.\n"
            "Para4 once more.\n"
            "Para5 and last."
        ),
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
    )


# ---- Test cases ---- #


def test_basic_split(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert hasattr(result, "chunks")
    assert (
        result.chunks[0] == "Para1 first sentence. Para1 second sentence.\nPara2 here."
    )
    assert result.chunks[1] == "Para3 is this line.\nPara4 once more."
    assert result.chunks[2] == "Para5 and last."
    assert result.split_method == "paragraph_splitter"
    assert result.split_params["chunk_size"] == 2
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=3)
    result = splitter.split(reader_output)
    # Each chunk after the first should start with the last 3 words of the previous chunk
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    first_words = first_chunk.split()[-3:]
    assert " ".join(first_words) in second_chunk


def test_split_with_overlap_float(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    if len(result.chunks) > 1:
        prev_words = result.chunks[0].split()
        overlap = set(prev_words) & set(result.chunks[1].split())
        assert len(overlap) >= 1


def test_custom_linebreak():
    text = "P1||P2||P3"
    reader_output = ReaderOutput(text=text, document_path="/tmp/sample.txt")
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0, line_break="||")
    result = splitter.split(reader_output)
    assert result.chunks[0] == "P1||P2"
    assert result.chunks[1] == "P3"


def test_output_contains_metadata(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
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


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_init_invalid_chunk_size_raises_splitter_config_exception(chunk_size):
    with pytest.raises(SplitterConfigException, match="chunk_size must be"):
        ParagraphSplitter(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, -0.1, "bad"])
def test_init_invalid_chunk_overlap_raises_splitter_config_exception(chunk_overlap):
    with pytest.raises(SplitterConfigException, match="chunk_overlap must be"):
        ParagraphSplitter(chunk_overlap=chunk_overlap)  # type: ignore[arg-type]


@pytest.mark.parametrize("line_break", [123, 1.5, None])
def test_init_invalid_line_break_type_raises_splitter_config_exception(line_break):
    with pytest.raises(SplitterConfigException, match="line_break must be a string"):
        ParagraphSplitter(line_break=line_break)  # type: ignore[arg-type]


def test_init_invalid_line_break_contents_raises_splitter_config_exception():
    # Empty string and non-string entries should be rejected
    with pytest.raises(SplitterConfigException, match="line_break must contain"):
        ParagraphSplitter(line_break=["", "\n\n"])

    with pytest.raises(SplitterConfigException, match="line_break must contain"):
        ParagraphSplitter(line_break=["\n\n", 123])  # type: ignore[list-item]


def test_split_missing_text_attribute_raises_reader_output_exception():
    splitter = ParagraphSplitter()
    bogus = SimpleNamespace()  # no 'text' attribute at all
    with pytest.raises(ReaderOutputException, match="must expose a 'text' attribute"):
        splitter.split(bogus)  # type: ignore[arg-type]


def test_split_non_string_text_raises_reader_output_exception():
    splitter = ParagraphSplitter()
    # Build a valid ReaderOutput then corrupt the text (to avoid Pydantic ValidationError)
    ro = ReaderOutput(text="valid text")
    ro.text = 123  # type: ignore[assignment]

    with pytest.raises(
        ReaderOutputException,
        match=r"ReaderOutput\.text must be of type 'str' or None",
    ):
        splitter.split(ro)


def test_split_empty_text_emits_input_and_output_warnings_and_returns_empty_chunk():
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
    ro = ReaderOutput(text="")

    # Expect both an input warning (empty/whitespace text) and an output warning
    # (no non-empty paragraphs).
    with pytest.warns((SplitterInputWarning, SplitterOutputWarning)) as record:
        out = splitter.split(ro)

    input_warns = [w for w in record if isinstance(w.message, SplitterInputWarning)]
    output_warns = [w for w in record if isinstance(w.message, SplitterOutputWarning)]

    assert input_warns, "Expected a SplitterInputWarning for empty text"
    assert output_warns, "Expected a SplitterOutputWarning for no non-empty paragraphs"

    # Fallback behaviour: single empty chunk
    assert out.chunks == [""]
    assert len(out.chunk_id) == 1


def test_split_text_with_only_separators_triggers_output_warning_and_empty_chunk():
    splitter = ParagraphSplitter(line_break="\n\n")
    ro = ReaderOutput(text="\n\n   \n\n")

    with pytest.warns(
        SplitterOutputWarning, match="did not find any non-empty paragraphs"
    ):
        out = splitter.split(ro)

    assert out.chunks == [""]
    assert len(out.chunk_id) == 1


def test_split_mismatched_chunk_ids_raises_invalid_chunk_exception(monkeypatch):
    splitter = ParagraphSplitter(chunk_size=1)
    ro = ReaderOutput(text="Para 1.\n\nPara 2.")

    # Force mismatch: always return a single ID; signature must be (n) when
    # attached directly to the instance.
    monkeypatch.setattr(
        splitter,
        "_generate_chunk_ids",
        lambda _n: ["only-one-id"],  # type: ignore[method-assign]
    )

    with pytest.raises(
        InvalidChunkException, match="Number of chunk IDs does not match"
    ):
        splitter.split(ro)


def test_split_wraps_splitteroutput_construction_errors_in_splitter_output_exception(
    monkeypatch,
):
    splitter = ParagraphSplitter(chunk_size=1)
    ro = ReaderOutput(text="Para 1.")

    # Patch SplitterOutput *import* in this module so that construction fails
    # with a generic Exception, which should be wrapped.
    from splitter_mr.splitter.splitters import paragraph_splitter as ps_mod

    class BoomSplitterOutput:
        def __init__(self, *args, **kwargs):
            raise TypeError("boom")

    monkeypatch.setattr(ps_mod, "SplitterOutput", BoomSplitterOutput)

    with pytest.raises(SplitterOutputException) as excinfo:
        splitter.split(ro)

    assert "Failed to build SplitterOutput in ParagraphSplitter" in str(excinfo.value)


def test_underflow_warning_when_fewer_chunks_than_expected():
    # Only 2 paragraphs but chunk_size=5 → expected_chunks = ceil(2/5) = 1,
    # but limit warnings still should go off since paragraphs < chunk_size overall.
    from splitter_mr.schema import ChunkUnderflowWarning

    splitter = ParagraphSplitter(chunk_size=5)
    ro = ReaderOutput(text="p1\n\np2")

    with pytest.warns(ChunkUnderflowWarning) as record:
        out = splitter.split(ro)

    print(out)
    assert out.chunks == ["p1\np2"]
    assert len(record) == 1
    assert "fewer chunks" in str(record[0].message)
