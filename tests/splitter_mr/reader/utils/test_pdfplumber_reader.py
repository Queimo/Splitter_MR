import base64
import importlib
import logging
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.reader.utils import PDFPlumberReader
from splitter_mr.schema import DEFAULT_IMAGE_EXTRACTION_PROMPT

# ---- Helpers, mocks and fixtures ---- #

PKG = "splitter_mr.reader.utils"


class DummyModel:
    """
    Mimics a vision model: always returns the same caption but records calls.
    Matches the DummyModel used in the original test-suite so expectations stay
    consistent across all files.
    """

    def __init__(self):
        self.calls = []

    def analyze_content(self, file, prompt=None, **kw):
        self.calls.append({"file": file, "prompt": prompt, **kw})
        return "Dummy caption"


@pytest.fixture
def mock_word_lines():
    return [
        {"text": "Hello", "top": 10.0, "bottom": 18.0, "x0": 50},
        {"text": "world", "top": 10.1, "bottom": 18.1, "x0": 80},
        {"text": "This", "top": 35.0, "bottom": 45.0, "x0": 50},
        {"text": "is", "top": 35.2, "bottom": 45.2, "x0": 80},
    ]


@pytest.fixture
def fake_table():
    return [["Header1", "Header2"], ["Row1Cell1", "Row1Cell2"], ["Row2Cell1", ""]]


@pytest.fixture
def fake_image_dict():
    return [{"x0": 100, "top": 200, "x1": 300, "bottom": 400}]


def _fake_to_image_factory(payload: bytes = b"fakeimg"):
    """Builds a stub for pdfplumber.Page.to_image()."""

    class _FakeImage:
        def save(self, buf: BytesIO, format: str = "PNG"):
            buf.write(payload)

    fake_image = _FakeImage()

    def _to_image(resolution=150):
        return fake_image

    return _to_image


# ---- Test cases ---- #


def test_group_by_lines(mock_word_lines):
    reader = PDFPlumberReader()
    result = reader.group_by_lines(mock_word_lines)
    assert len(result) == 2
    assert result[0]["content"] == "Hello world"
    assert result[1]["content"] == "This is"


def test_is_real_table():
    reader = PDFPlumberReader()
    assert reader.is_real_table([["a", "b"], ["c", "d"]])
    assert not reader.is_real_table([["a"], ["b"], ["c"]])  # mostly single col
    assert not reader.is_real_table([[]])  # blank row
    assert not reader.is_real_table([])  # empty


def test_table_to_markdown(fake_table):
    reader = PDFPlumberReader()
    md = reader.table_to_markdown(fake_table)
    assert "| Header1 | Header2 |" in md
    assert "| Row1Cell1 | Row1Cell2 |" in md


@patch("pdfplumber.open")
def test_read_extracts_text(
    mock_pdfplumber_open, mock_word_lines, fake_table, fake_image_dict
):
    # Set up fake pdfplumber
    fake_page = MagicMock()
    fake_page.extract_words.return_value = mock_word_lines
    fake_page.find_tables.return_value = []
    fake_page.images = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf", show_base64_images=False)
    assert "Hello world" in md
    assert "This is" in md


@patch("pdfplumber.open")
def test_read_with_table(mock_pdfplumber_open, mock_word_lines, fake_table):
    fake_table_obj = MagicMock()
    fake_table_obj.bbox = (10, 20, 30, 40)
    fake_table_obj.extract.return_value = fake_table

    fake_page = MagicMock()
    fake_page.extract_words.return_value = mock_word_lines
    fake_page.find_tables.return_value = [fake_table_obj]
    fake_page.images = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf")
    assert "| Header1 | Header2 |" in md


@patch("pdfplumber.open")
def test_read_with_images_and_annotations(mock_pdfplumber_open):
    # Fake image object
    fake_image = {"x0": 10, "top": 20, "x1": 30, "bottom": 40}
    # Patch the image extraction chain
    fake_img_obj = MagicMock()
    fake_img_obj.to_image.return_value = MagicMock(
        save=lambda buf, format: buf.write(b"fakeimg")
    )
    fake_page = MagicMock()
    fake_page.extract_words.return_value = []
    fake_page.find_tables.return_value = []
    fake_page.images = [fake_image]
    fake_page.within_bbox.return_value = fake_img_obj

    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    # Without annotation
    md = reader.read("fakefile.pdf", show_base64_images=True)
    assert "data:image/png;base64" in md

    # With annotation
    md = reader.read("fakefile.pdf", show_base64_images=False, model=DummyModel())
    assert "Dummy caption" in md


@patch("pdfplumber.open")
def test_blocks_to_markdown_omitted_image_indicator(mock_pdfplumber_open):
    # Test for image omitted placeholder
    fake_image = {"x0": 10, "top": 20, "x1": 30, "bottom": 40}
    fake_img_obj = MagicMock()
    fake_img_obj.to_image.return_value = MagicMock(
        save=lambda buf, format: buf.write(b"fakeimg")
    )
    fake_page = MagicMock()
    fake_page.extract_words.return_value = []
    fake_page.find_tables.return_value = []
    fake_page.images = [fake_image]
    fake_page.within_bbox.return_value = fake_img_obj
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf", show_base64_images=False)
    assert "Image" in md or "image" in md


def test_blocks_to_markdown_table_and_text():
    reader = PDFPlumberReader()
    blocks = [
        {"type": "text", "top": 10, "bottom": 20, "content": "Text 1", "page": 1},
        {
            "type": "table",
            "top": 30,
            "bottom": 40,
            "content": [["a", "b"], ["c", "d"]],
            "page": 1,
        },
        {"type": "text", "top": 50, "bottom": 60, "content": "Text 2", "page": 1},
    ]
    md = reader.blocks_to_markdown(blocks, show_base64_images=True)
    assert "Text 1" in md
    assert "| a | b |" in md
    assert "Text 2" in md


def test_blocks_to_markdown_handles_blank():
    reader = PDFPlumberReader()
    blocks = []
    md = reader.blocks_to_markdown(blocks)
    assert isinstance(md, str)


@patch("pdfplumber.open")
def test_extract_tables_filters_invalid(mock_open):
    valid_tbl = [["H1", "H2"], ["v1", "v2"]]
    trash_tbl = [["solo"], ["x"], ["y"], ["z"]]

    good = MagicMock(bbox=(0, 10, 100, 50))
    good.extract.return_value = valid_tbl

    bad = MagicMock(bbox=(0, 60, 100, 90))
    bad.extract.return_value = trash_tbl

    fake_page = MagicMock()
    fake_page.find_tables.return_value = [good, bad]
    fake_page.extract_words.return_value = []
    fake_page.images = []

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    mock_open.return_value.__enter__.return_value = fake_pdf

    reader = PDFPlumberReader()
    tables, bboxes = reader.extract_tables(fake_page, 1)

    assert tables == [
        {
            "type": "table",
            "top": 10,
            "bottom": 50,
            "content": valid_tbl,
            "page": 1,
        }
    ]
    assert bboxes == [(0, 10, 100, 50)]


def test_analyze_content_excludes_table_rows():
    reader = PDFPlumberReader()
    words = [
        {"text": "Header", "top": 20, "bottom": 30, "x0": 10},
        {"text": "Outside", "top": 70, "bottom": 80, "x0": 10},
    ]
    table_bbox = [(0, 10, 100, 30)]

    page = MagicMock()
    page.extract_words.return_value = words

    lines = reader.analyze_content(page, 1, table_bbox)
    assert [line["content"] for line in lines] == ["Outside"]


@patch("pdfplumber.open")
def test_extract_pages_as_images_base64(mock_open):
    fake_page = MagicMock()
    fake_page.to_image.side_effect = _fake_to_image_factory(b"abc123")
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page, fake_page]
    mock_open.return_value.__enter__.return_value = fake_pdf

    reader = PDFPlumberReader()
    b64_list = reader.extract_pages_as_images("dummy.pdf")

    expected = base64.b64encode(b"abc123").decode()
    assert b64_list == [expected, expected]


def test_describe_pages_calls_vlm(monkeypatch):
    reader = PDFPlumberReader()

    dummy_imgs = ["a", "b", "c"]
    monkeypatch.setattr(reader, "extract_pages_as_images", lambda *a, **kw: dummy_imgs)

    model = DummyModel()
    output = reader.describe_pages("file.pdf", model=model)

    assert output == ["Dummy caption"] * 3
    assert len(model.calls) == 3
    for idx, call in enumerate(model.calls):
        assert call["file"] == dummy_imgs[idx]
        assert (
            call["prompt"] is None or call["prompt"] == DEFAULT_IMAGE_EXTRACTION_PROMPT
        )


def test_extract_images_encodes_and_annotates():
    reader = PDFPlumberReader()

    bbox = {"x0": 0, "top": 0, "x1": 10, "bottom": 10}
    page = MagicMock()
    page.images = [bbox]
    page.within_bbox.return_value.to_image = _fake_to_image_factory(b"img")

    # Without model: placeholder only
    imgs = reader.extract_images(page, page_num=1, model=None)
    assert imgs and imgs[0]["annotation"].startswith("<!-- image -->")

    # With model: caption appended
    model = DummyModel()
    imgs_annot = reader.extract_images(page, page_num=1, model=model)
    assert imgs_annot and "Dummy caption" in imgs_annot[0]["annotation"]


def _reload_pkg():
    """
    Force a clean re-import of the package under test so monkeypatching
    importlib.import_module affects the import path resolution each time.
    """
    for key in list(sys.modules.keys()):  # NOSONAR (S7504)
        if key == PKG or key.startswith(PKG + "."):
            sys.modules.pop(key, None)
    return importlib.import_module(PKG)


def test_unknown_attribute_raises_attributeerror():
    mod = _reload_pkg()
    with pytest.raises(AttributeError):
        getattr(mod, "DoesNotExist")


def test___dir___matches___all__():
    mod = _reload_pkg()
    assert set(dir(mod)) >= set(
        mod.__all__
    )  # dir() can include more, but must include __all__


def test_pdfplumber_import_succeeds_when_docling_missing(monkeypatch):
    """
    Simulate 'docling' extra missing. Accessing PDFPlumberReader should still work.
    """
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        # Resolve absolute module name the same way __getattr__ does
        abs_name = importlib.util.resolve_name(name, package) if package else name
        if abs_name.endswith(".docling_utils"):
            # Simulate missing optional extra
            raise ModuleNotFoundError("docling")
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    mod = _reload_pkg()
    pdf_plumber_reader = getattr(mod, "PDFPlumberReader")
    # Basic sanity: it's a class/type or callable
    assert hasattr(pdf_plumber_reader, "__name__")
    assert pdf_plumber_reader.__name__ == "PDFPlumberReader"


def test_read_invalid_file_path_none_raises():
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="file_path must be a non-empty string"):
        reader.read(None)  # type: ignore[arg-type]


def test_read_invalid_file_path_empty_raises():
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="file_path must be a non-empty string"):
        reader.read("   ")


@patch("pdfplumber.open")
def test_read_pdfplumber_open_failure_is_wrapped(mock_open):
    mock_open.side_effect = Exception("cannot open")
    reader = PDFPlumberReader()

    with pytest.raises(
        RuntimeError, match=r"Failed to open/read PDF 'fakefile\.pdf': cannot open"
    ):
        reader.read("fakefile.pdf")


@patch("pdfplumber.open")
def test_read_page_extraction_failure_is_wrapped_with_page_context(mock_open):
    # Make pdfplumber.open succeed, but extraction fail for page 1
    fake_page = MagicMock()
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    mock_open.return_value.__enter__.return_value = fake_pdf

    reader = PDFPlumberReader()
    with patch.object(reader, "extract_page_blocks", side_effect=Exception("boom")):
        with pytest.raises(
            RuntimeError,
            match=r"Failed extracting content blocks on page 1 of 'fakefile\.pdf': boom",
        ):
            reader.read("fakefile.pdf")


def test_group_by_lines_invalid_tolerance_raises(mock_word_lines):
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="tolerance must be a positive number"):
        reader.group_by_lines(mock_word_lines, tolerance=0)


def test_group_by_lines_words_not_list_raises():
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="words must be a list"):
        reader.group_by_lines(words="not-a-list")  # type: ignore[arg-type]


def test_group_by_lines_none_returns_empty():
    reader = PDFPlumberReader()
    assert reader.group_by_lines(None) == []  # type: ignore[arg-type]


def test_extract_tables_find_tables_failure_logs_and_returns_empty(caplog):
    reader = PDFPlumberReader()
    page = MagicMock()
    page.find_tables.side_effect = Exception("nope")

    caplog.set_level(logging.WARNING)
    tables, bboxes = reader.extract_tables(page, page_num=1)

    assert tables == []
    assert bboxes == []
    assert any(
        "Failed to find tables on page 1" in rec.message for rec in caplog.records
    )


def test_extract_tables_single_table_extract_failure_logs_and_continues(caplog):
    reader = PDFPlumberReader()

    good_tbl = MagicMock()
    good_tbl.bbox = (0, 10, 100, 50)
    good_tbl.extract.return_value = [["H1", "H2"], ["v1", "v2"]]

    bad_tbl = MagicMock()
    bad_tbl.bbox = (0, 60, 100, 90)
    bad_tbl.extract.side_effect = Exception("bad table")

    page = MagicMock()
    page.find_tables.return_value = [bad_tbl, good_tbl]

    caplog.set_level(logging.WARNING)
    tables, bboxes = reader.extract_tables(page, page_num=2)

    # bad one is skipped, good one remains
    assert len(tables) == 1
    assert tables[0]["type"] == "table"
    assert bboxes == [(0, 10, 100, 50)]
    assert any(
        "Failed to extract a table on page 2" in rec.message for rec in caplog.records
    )


def test_extract_images_annotation_failure_logged_and_in_annotation(caplog):
    reader = PDFPlumberReader()

    # Build a page with one image bbox
    page = MagicMock()
    page.images = [{"x0": 0, "top": 0, "x1": 10, "bottom": 10}]
    page.within_bbox.return_value.to_image = _fake_to_image_factory(b"imgbytes")

    # Model fails during annotation
    model = DummyModel()
    model.analyze_content = MagicMock(side_effect=Exception("model down"))

    caplog.set_level(logging.WARNING)
    imgs = reader.extract_images(page, page_num=1, model=model, prompt="p")

    assert len(imgs) == 1
    assert "**Annotation error:**" in imgs[0]["annotation"]
    assert any(
        "Model annotation failed for image" in rec.message for rec in caplog.records
    )


def test_extract_images_malformed_bbox_is_logged_and_skipped(caplog):
    reader = PDFPlumberReader()

    # Missing required keys like x0/top/x1/bottom
    page = MagicMock()
    page.images = [{"top": 0}]  # malformed
    caplog.set_level(logging.WARNING)

    imgs = reader.extract_images(page, page_num=1, model=None)
    assert imgs == []
    assert any(
        "Error extracting/encoding image" in rec.message for rec in caplog.records
    )


def test_extract_pages_as_images_invalid_resolution_raises():
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="resolution must be an int in the range"):
        reader.extract_pages_as_images("file.pdf", resolution=10)


@patch("pdfplumber.open")
def test_extract_pages_as_images_open_failure_is_wrapped(mock_open):
    mock_open.side_effect = Exception("open fail")
    reader = PDFPlumberReader()

    with pytest.raises(
        RuntimeError, match=r"Failed to open/read PDF 'file\.pdf': open fail"
    ):
        reader.extract_pages_as_images("file.pdf")


@patch("pdfplumber.open")
def test_extract_pages_as_images_page_rasterize_failure_is_wrapped_with_page(mock_open):
    fake_page = MagicMock()
    fake_page.to_image.side_effect = Exception("raster fail")

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    mock_open.return_value.__enter__.return_value = fake_pdf

    reader = PDFPlumberReader()

    with pytest.raises(
        RuntimeError, match=r"Failed to rasterize page 1 from 'file\.pdf': raster fail"
    ):
        reader.extract_pages_as_images("file.pdf")


def test_describe_pages_requires_model():
    reader = PDFPlumberReader()
    with pytest.raises(RuntimeError, match="model is required for describe_pages"):
        reader.describe_pages("file.pdf", model=None)  # type: ignore[arg-type]


def test_blocks_to_markdown_none_returns_empty():
    reader = PDFPlumberReader()
    assert reader.blocks_to_markdown(None) == ""  # type: ignore[arg-type]


def test_analyze_content_extract_words_failure_logs_and_returns_empty(caplog):
    reader = PDFPlumberReader()
    page = MagicMock()
    page.extract_words.side_effect = Exception("word fail")

    caplog.set_level(logging.WARNING)
    out = reader.analyze_content(page, page_num=1, table_bboxes=[])

    assert out == []
    assert any(
        "Failed to extract words on page 1" in rec.message for rec in caplog.records
    )
