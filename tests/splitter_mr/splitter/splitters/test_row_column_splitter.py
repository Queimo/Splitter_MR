import pandas as pd
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
from splitter_mr.splitter.splitters import row_column_splitter as rcs_module
from splitter_mr.splitter.splitters.row_column_splitter import RowColumnSplitter

# ---- Mocks, fixtures and helpers ---- #


def make_reader_output(text, method):
    return ReaderOutput(
        text=text,
        document_name="test",
        document_path="test",
        conversion_method=method,
        document_id="doc1",
        ocr_method=None,
        metadata={},
    )


def parse_data_rows_from_markdown(markdown_chunk):
    """Return list of tuples representing the data rows in a markdown table chunk."""
    lines = markdown_chunk.strip().splitlines()
    data_lines = lines[2:]
    rows = []
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and any(cells):
            rows.append(tuple(cells))
    return rows


def is_markdown_header_line(line, required_cols):
    cols = [c.strip().lower() for c in line.strip("|").split("|")]
    return all(col in cols for col in required_cols)


# ---- Test cases ---- #


@pytest.mark.parametrize(
    "num_rows, num_cols, overlap, expected_chunks",
    [
        (2, 0, 0, 2),  # 2 rows per chunk, no overlap
        (1, 0, 1, 4),  # 1 row per chunk, 1 overlap (4 rows → 4 chunks)
        (0, 2, 1, 4),  # 2 cols per chunk, 1 col overlap
    ],
)
def test_json_tabular(num_rows, num_cols, overlap, expected_chunks):
    json_tabular = """
    [
        {"id": 1, "name": "A", "amount": 1, "Remark": "x"},
        {"id": 2, "name": "B", "amount": 2, "Remark": "y"},
        {"id": 3, "name": "C", "amount": 3, "Remark": "z"},
        {"id": 4, "name": "D", "amount": 4, "Remark": "w"}
    ]"""
    reader_output = make_reader_output(json_tabular, "json")
    splitter = RowColumnSplitter(
        num_rows=num_rows, num_cols=num_cols, chunk_overlap=overlap
    )
    output = splitter.split(reader_output)
    assert len(output.chunks) == expected_chunks


def test_markdown_table():
    md_table = (
        "| id | name | amount | Remark |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | Alice | 1 | ok |\n"
        "| 2 | Bob | 2 | good |\n"
        "| 3 | Carol | 3 | fair |\n"
        "| 4 | Dan | 4 | bad |"
    )
    reader_output = make_reader_output(md_table, "markdown")
    splitter = RowColumnSplitter(num_rows=2)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_csv_split():
    csv_content = (
        "id,name,amount,Remark\n"
        "1,Alice,1,ok\n"
        "2,Bob,2,good\n"
        "3,Carol,3,fair\n"
        "4,Dan,4,bad\n"
    )
    reader_output = make_reader_output(csv_content, "csv")
    splitter = RowColumnSplitter(num_rows=2)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_tsv_split():
    tsv_content = (
        "id\tname\tamount\tRemark\n"
        "1\tAlice\t1\tok\n"
        "2\tBob\t2\tgood\n"
        "3\tCarol\t3\tfair\n"
        "4\tDan\t4\tbad\n"
    )
    reader_output = make_reader_output(tsv_content, "tsv")
    splitter = RowColumnSplitter(num_rows=3)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2  # first 3, last 1 row
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_chunk_size_too_small():
    md_table = "| id | name |\n|----|------|\n| 1  | A    |\n"
    splitter = RowColumnSplitter(chunk_size=10, num_rows=0, num_cols=0)
    reader_output = make_reader_output(md_table, "markdown")
    with pytest.raises(
        SplitterConfigException,
        match=r"chunk_size is too small to fit the header and at least one row;",
    ):
        splitter.split(reader_output)


def test_chunk_size_only():
    md_table = (
        "| id | name |\n"
        "|----|------|\n"
        "| 1  | A    |\n"
        "| 2  | B    |\n"
        "| 3  | C    |\n"
        "| 4  | D    |\n"
    )
    splitter = RowColumnSplitter(chunk_size=60, num_rows=0, num_cols=0)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    for chunk in output.chunks:
        header_line = chunk.strip().splitlines()[0]
        assert is_markdown_header_line(header_line, ["id", "name"])
        assert len(chunk) <= 60

    found_rows = set()
    for chunk in output.chunks:
        for row in parse_data_rows_from_markdown(chunk):
            found_rows.add(row)
    expected = [("1", "A"), ("2", "B"), ("3", "C"), ("4", "D")]
    for row in expected:
        assert row in found_rows


def test_chunk_size_with_overlap():
    md_table = (
        "| id | name |\n|----|------|\n| 1  | A    |\n| 2  | B    |\n| 3  | C    |\n"
    )
    splitter = RowColumnSplitter(
        chunk_size=80, chunk_overlap=10, num_rows=0, num_cols=0
    )
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)

    all_rows = [parse_data_rows_from_markdown(chunk) for chunk in output.chunks]
    for i in range(len(all_rows) - 1):
        assert all_rows[i][-1] == all_rows[i + 1][0]

    expected = [("1", "A"), ("2", "B"), ("3", "C")]
    found = [row for rows in all_rows for row in rows]
    for row in expected:
        assert row in found


def test_single_row():
    md_table = "| id | name |\n|----|------|\n| 1  | A    |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    assert len(output.chunks) == 1
    for col in ["id", "name", "A"]:
        assert col in output.chunks[0]


def test_one_column():
    md_table = "| id |\n|----|\n| 1  |\n| 2  |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    for chunk in output.chunks:
        assert "id" in chunk


def test_missing_headers():
    # No header line at all
    md_table = "| 1 | A |\n| 2 | B |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    assert "1" in output.chunks[0]
    assert "A" in output.chunks[0]


def test_malformed_table():
    md_table = (
        "| id | name |\n"
        "|----|------|\n"
        "| 1  | A |\n"  # Too few columns
        "2 | B |\n"  # Missing leading pipe
        "| 3 | C | X |\n"  # Too many columns
    )
    splitter = RowColumnSplitter(num_rows=2)
    reader_output = make_reader_output(md_table, "markdown")
    with pytest.raises(pd.errors.ParserError, match="Malformed markdown table"):
        splitter.split(reader_output)


# ---- Error and warning handling ---- #


def test_empty_input():
    splitter = RowColumnSplitter(num_rows=2)
    reader_output = make_reader_output("", "markdown")
    # 1) input warning from _load_tabular
    with pytest.warns(SplitterInputWarning, match="empty or whitespace-only text"):
        # 2) building SplitterOutput with empty chunks → wrapped as SplitterOutputException
        with pytest.raises(SplitterOutputException):
            splitter.split(reader_output)


def test_split_nonempty_text_but_empty_dataframe_emits_output_warning():
    # "markdown" method but no table lines (no pipes) → _parse_markdown_table
    # returns empty DataFrame, and `split` should warn.
    text = "This is not a table at all"
    reader_output = make_reader_output(text, "markdown")
    splitter = RowColumnSplitter(num_rows=2)
    with pytest.warns(SplitterOutputWarning, match="empty DataFrame from non-empty"):
        try:
            splitter.split(reader_output)
        except SplitterOutputException:
            # Expected: building SplitterOutput with empty chunks fails
            pass


def test_config_num_rows_and_num_cols_mutually_exclusive():
    with pytest.raises(SplitterConfigException, match="mutually exclusive"):
        RowColumnSplitter(num_rows=1, num_cols=1)


def test_config_negative_num_rows_or_cols_raises():
    with pytest.raises(SplitterConfigException, match="must be non-negative"):
        RowColumnSplitter(num_rows=-1)
    with pytest.raises(SplitterConfigException, match="must be non-negative"):
        RowColumnSplitter(num_cols=-2)


def test_config_negative_overlap_raises():
    with pytest.raises(SplitterConfigException, match="must be non-negative"):
        RowColumnSplitter(chunk_overlap=-1)


def test_config_float_overlap_out_of_range_raises():
    with pytest.raises(SplitterConfigException, match="range \\[0, 1\\)"):
        RowColumnSplitter(chunk_overlap=1.5)


def test_split_missing_text_attribute_raises_reader_output_exception():
    from types import SimpleNamespace

    splitter = RowColumnSplitter(num_rows=1)
    bad_ro = SimpleNamespace(
        document_name="x",
        document_path="x",
        conversion_method="csv",
        document_id="doc1",
        reader_method=None,
        ocr_method=None,
        metadata={},
    )
    with pytest.raises(ReaderOutputException, match="must expose a 'text' attribute"):
        splitter.split(bad_ro)  # type: ignore[arg-type]


def test_split_non_string_text_raises_reader_output_exception():
    from types import SimpleNamespace

    splitter = RowColumnSplitter(num_rows=1)
    bad_ro = SimpleNamespace(
        text=123,  # not str / None
        document_name="x",
        document_path="x",
        conversion_method="csv",
        document_id="doc1",
        reader_method=None,
        ocr_method=None,
        metadata={},
    )
    with pytest.raises(ReaderOutputException, match="must be of type 'str' or None"):
        splitter.split(bad_ro)  # type: ignore[arg-type]


def test_load_tabular_unknown_method_json_fallback_warns():
    json_tabular = """
    [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"}
    ]"""
    reader_output = make_reader_output(json_tabular, "weird-format")
    splitter = RowColumnSplitter(num_rows=1)
    with pytest.warns(SplitterInputWarning, match="Unknown conversion_method"):
        df = splitter._load_tabular(reader_output)
    assert not df.empty
    assert list(df.columns) == ["id", "name"]


def test_load_tabular_unknown_method_csv_fallback_warns():
    csv_content = "id,name\n1,A\n2,B\n"
    reader_output = make_reader_output(csv_content, "unknown")
    splitter = RowColumnSplitter(num_rows=1)
    with pytest.warns(SplitterInputWarning, match="falling back to CSV parser"):
        df = splitter._load_tabular(reader_output)
    assert not df.empty
    assert list(df.columns) == ["id", "name"]


def test_load_tabular_empty_text_warns_and_returns_empty_df():
    reader_output = make_reader_output("", "csv")
    splitter = RowColumnSplitter(num_rows=1)
    with pytest.warns(SplitterInputWarning, match="empty or whitespace-only text"):
        df = splitter._load_tabular(reader_output)
    assert df.empty


def test_invalid_chunk_ids_raise_invalid_chunk_exception(monkeypatch):
    # Use 2 rows → num_rows=1 ⇒ 2 chunks
    md_table = "| id | name |\n|----|------|\n| 1  | A    |\n| 2  | B    |\n"
    reader_output = make_reader_output(md_table, "markdown")
    splitter = RowColumnSplitter(num_rows=1)

    def fake_generate_chunk_ids(self, n: int):
        # Always return a single ID regardless of n
        return ["only-one-id"]

    monkeypatch.setattr(
        RowColumnSplitter, "_generate_chunk_ids", fake_generate_chunk_ids, raising=True
    )

    with pytest.raises(
        InvalidChunkException, match="Number of chunk IDs does not match"
    ):
        splitter.split(reader_output)


def test_splitter_output_exception_when_splitteroutput_construction_fails(monkeypatch):
    # Simulate SplitterOutput failing validation (e.g. pydantic internal error)
    md_table = "| id | name |\n|----|------|\n| 1  | A    |\n"
    reader_output = make_reader_output(md_table, "markdown")
    splitter = RowColumnSplitter(num_rows=1)

    rcs_module.SplitterOutput

    def broken_splitter_output(*args, **kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(
        rcs_module, "SplitterOutput", broken_splitter_output, raising=True
    )

    with pytest.raises(SplitterOutputException, match="Failed to build SplitterOutput"):
        splitter.split(reader_output)
