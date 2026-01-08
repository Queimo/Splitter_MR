import json
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest
import requests

from src.splitter_mr.reader.readers.vanilla_reader import VanillaReader
from src.splitter_mr.schema.exceptions import (
    ReaderConfigException,
    VanillaReaderException,
)

# ---- Fixtures ----


@pytest.fixture
def mock_vision_model():
    """Mocks the BaseVisionModel."""
    model = MagicMock()
    model.model_name = "mock_vision_model"
    model.analyze_content.return_value = "Image description"
    return model


@pytest.fixture
def reader(mock_vision_model):
    """Returns a VanillaReader instance with a mock model."""
    return VanillaReader(model=mock_vision_model)


@pytest.fixture
def reader_no_model():
    """Returns a VanillaReader instance without a model."""
    return VanillaReader(model=None)


# ---- Tests: Input Dispatch & Validation ----


def test_read_missing_input(reader):
    """Test that calling read without arguments checks priority but fails if empty."""
    with pytest.raises(ReaderConfigException, match="file_path cannot be None"):
        reader.read()


def test_read_invalid_path_type(reader):
    """Test passing a non-string/non-Path object as file_path."""
    with pytest.raises(ReaderConfigException, match="must be a string or Path"):
        reader.read(file_path=123)


def test_read_url_as_path_argument(reader):
    """Test that a URL passed as a positional arg is routed to the URL handler."""
    # We must mock the return value of _handle_url because _dispatch_source unpacks 5 values
    with patch.object(
        reader, "_handle_url", return_value=("name", "url", "content", "conv", "ocr")
    ) as mock_url_handler:
        reader.read("https://example.com/doc.txt")
        mock_url_handler.assert_called_once()


# ---- Tests: Local File Handling ----


@pytest.mark.parametrize(
    "ext, content, expected_type",
    [
        ("txt", "simple text", "txt"),
        ("py", "print('hello')", "txt"),
    ],
)
def test_read_simple_text_files(reader, ext, content, expected_type):
    """Test reading simple text-based files."""
    filename = f"test.{ext}"
    with patch("builtins.open", mock_open(read_data=content)):
        # Important: Mock is_valid_file_path or os.path.isfile to True
        # otherwise it falls back to treating the filename as content
        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            # Also mock os.path.relpath to avoid filesystem errors
            with patch("os.path.relpath", return_value=filename):
                result = reader.read(filename)

                assert result.text == content
                assert result.conversion_method == expected_type
                assert result.document_name == filename


def test_read_json_file(reader):
    """Test reading a .json file."""
    data = {"key": "value"}
    json_str = json.dumps(data)

    with patch("builtins.open", mock_open(read_data=json_str)):
        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            result = reader.read("test.json")
            # The reader ensures string output, defaulting to json.dumps for dicts
            assert '"key": "value"' in result.text
            assert result.conversion_method == "json"


def test_read_yaml_file(reader):
    """Test reading a .yaml file."""
    # The code reads YAML, parses it, then DUMPS it back as YAML (via yaml.safe_dump)
    # So we expect YAML formatted string in the output, even if conv method says "json"
    yaml_content = "key: value"

    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            result = reader.read("test.yaml")

            # The output text should be valid YAML/string
            assert "key: value" in result.text
            assert result.conversion_method == "json"


def test_read_html_raw(reader):
    """Test reading HTML returning raw content."""
    html = "<html><body><h1>Hello</h1></body></html>"
    with patch("builtins.open", mock_open(read_data=html)):
        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            result = reader.read("test.html", html_to_markdown=False)
            assert result.text == html
            assert result.conversion_method == "html"


def test_read_html_to_markdown(reader):
    """Test reading HTML converting to Markdown."""
    html = "<h1>Hello</h1>"
    md_out = "# Hello"

    with patch("builtins.open", mock_open(read_data=html)):
        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            # Patch the HtmlToMarkdown class where it is IMPORTED in the reader file
            with patch(
                "src.splitter_mr.reader.readers.vanilla_reader.HtmlToMarkdown"
            ) as mock_cls:
                mock_cls.return_value.convert.return_value = md_out

                result = reader.read("test.html", html_to_markdown=True)
                assert result.text == md_out
                assert result.conversion_method == "md"


# ---- Tests: PDF Handling ----


def test_read_pdf_standard(reader):
    """Test standard element-wise PDF extraction."""
    mock_pdf_content = "Extracted text"

    reader.pdf_reader.read = MagicMock(return_value=mock_pdf_content)

    with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
        result = reader.read("doc.pdf")

        assert result.text == mock_pdf_content
        assert result.conversion_method == "pdf"
        reader.pdf_reader.read.assert_called_once()


def test_read_pdf_scanned(reader, mock_vision_model):
    """Test scanned PDF extraction (Vision Model)."""
    reader.pdf_reader.describe_pages = MagicMock(
        return_value=["Page 1 desc", "Page 2 desc"]
    )

    with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
        result = reader.read("scan.pdf", scan_pdf_pages=True)

        assert "Page 1 desc" in result.text
        assert "Page 2 desc" in result.text
        assert result.conversion_method == "png"
        reader.pdf_reader.describe_pages.assert_called_with(
            file_path="scan.pdf",
            model=mock_vision_model,
            prompt=ANY,  # Fixed: using unittest.mock.ANY
            resolution=300,
        )


def test_read_pdf_scanned_missing_model(reader_no_model):
    """Test error when scan_pdf_pages is True but no model provided."""
    with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
        with pytest.raises(
            ReaderConfigException, match="requires a vision-capable model"
        ):
            reader_no_model.read("scan.pdf", scan_pdf_pages=True)


# ---- Tests: Office & Binary Handling ----


def test_read_office_doc_success(reader):
    """Test automated conversion of .docx to PDF."""
    # We must patch is_valid_file_path=True so it enters the file handling logic
    with (
        patch.object(VanillaReader, "is_valid_file_path", return_value=True),
        patch("shutil.which", return_value="/usr/bin/soffice"),
        patch("tempfile.mkdtemp", return_value="/tmp/tmpx"),
        patch("subprocess.run") as mock_run,
        patch("os.path.exists", return_value=True),
        patch.object(reader, "_process_pdf", return_value=("PDF Content", "pdf", None)),
    ):
        mock_run.return_value.returncode = 0

        result = reader.read("document.docx")

        assert result.text == "PDF Content"
        # Verify libreoffice command was constructed
        cmd_args = mock_run.call_args[0][0]
        assert "soffice" in cmd_args
        assert "--convert-to" in cmd_args
        assert "pdf" in cmd_args


def test_read_office_doc_conversion_failure(reader):
    """Test failure during LibreOffice conversion."""
    # Must patch is_valid_file_path=True to trigger office logic
    with (
        patch.object(VanillaReader, "is_valid_file_path", return_value=True),
        patch("shutil.which", return_value="/usr/bin/soffice"),
        patch("tempfile.mkdtemp", return_value="/tmp/tmpx"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = b"Conversion failed"

        with pytest.raises(
            VanillaReaderException, match="LibreOffice failed converting"
        ):
            reader.read("document.docx")


def test_read_office_no_libreoffice(reader):
    """Test missing LibreOffice binary."""
    with (
        patch.object(VanillaReader, "is_valid_file_path", return_value=True),
        patch("shutil.which", return_value=None),
    ):
        with pytest.raises(
            VanillaReaderException, match="LibreOffice/soffice is required"
        ):
            reader.read("document.docx")


def test_read_excel_as_table(reader):
    """Test reading Excel with as_table=True (Pandas)."""
    with patch("pandas.read_excel") as mock_read_excel:
        mock_df = MagicMock()
        mock_df.to_csv.return_value = "col1,col2\n1,2"
        mock_read_excel.return_value = mock_df

        with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
            result = reader.read("data.xlsx", as_table=True)

            assert result.text == "col1,col2\n1,2"
            assert result.conversion_method == "xlsx"


def test_read_image(reader, mock_vision_model):
    """Test reading an image file."""
    with (
        patch("builtins.open", mock_open(read_data=b"fake_image_bytes")),
        patch.object(VanillaReader, "is_valid_file_path", return_value=True),
    ):
        result = reader.read("photo.png")

        assert result.text == "Image description"
        assert result.conversion_method == "image"
        mock_vision_model.analyze_content.assert_called_once()


# ---- Tests: URL Handling ----


def test_read_url_json(reader):
    """Test fetching JSON from URL."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {"api": "data"}

    with patch("requests.get", return_value=mock_response):
        result = reader.read(file_url="http://api.com/data")
        assert '"api": "data"' in result.text
        assert result.conversion_method == "json"


def test_read_url_html(reader):
    """Test fetching HTML from URL."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html></html>"

    with patch("requests.get", return_value=mock_response):
        result = reader.read(file_url="http://web.com")
        assert result.text == "<html></html>"
        assert result.conversion_method == "html"


def test_read_url_404(reader):
    """Test HTTP error handling."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with pytest.raises(VanillaReaderException, match="HTTP request failed"):
            reader.read(file_url="http://broken.com")


def test_read_url_invalid_arg(reader):
    """Test malformed URL."""
    with pytest.raises(ReaderConfigException, match="file_url must start with"):
        reader.read(file_url="ftp://unsupported")


# ---- Tests: Explicit Inputs (kwargs) ----


def test_read_explicit_json_dict(reader):
    """Test passing a dict directly."""
    data = {"foo": "bar"}
    result = reader.read(json_document=data)
    assert '"foo": "bar"' in result.text
    assert result.conversion_method == "json"


def test_read_explicit_text(reader):
    """Test passing raw text."""
    result = reader.read(text_document="Just raw text")
    assert result.text == "Just raw text"
    assert result.conversion_method == "txt"


def test_read_explicit_text_auto_parse_json(reader):
    """Test passing a JSON string in text_document autodetects JSON."""
    json_txt = '{"auto": "detect"}'
    result = reader.read(text_document=json_txt)
    assert '"auto": "detect"' in result.text
    assert result.conversion_method == "json"


# ---- Tests: Fallback & Edge Cases ----


def test_fallback_unsupported_extension(reader):
    """Test file with unknown extension raises exception."""
    # Mocking is_valid_file_path=True forces it into the logic block where extension matters
    with patch.object(VanillaReader, "is_valid_file_path", return_value=True):
        with pytest.raises(ReaderConfigException, match="Unsupported file extension"):
            reader.read("unknown.xyz")


def test_fallback_file_not_found_on_disk(reader):
    """Test handling of file that doesn't exist on disk (fallback logic)."""
    # If is_valid_file_path returns False (default or explicit), it goes to fallback.
    # Fallback tries to parse path as JSON/Text.

    with patch.object(VanillaReader, "is_valid_file_path", return_value=False):
        # 1. As valid JSON string
        result_json = reader.read('{"valid": "json"}')
        assert '"valid": "json"' in result_json.text

        # 2. As plain text (filename treated as content)
        result_text = reader.read("just_some_text_not_a_file")
        assert result_text.text == "just_some_text_not_a_file"


# ---- Tests: Metadata & Placeholders ----


def test_read_metadata_passthrough(reader):
    """Test that metadata provided in kwargs appears in output."""
    with (
        patch("builtins.open", mock_open(read_data="content")),
        patch.object(VanillaReader, "is_valid_file_path", return_value=True),
    ):
        result = reader.read("test.txt", metadata={"user": "admin"})
        assert result.metadata == {"user": "admin"}


def test_page_placeholder_surface(reader):
    """Test page placeholder logic."""
    # 1. Standard text (no placeholder in text) -> should be None
    res1 = reader._surface_page_placeholder(
        scan=False, placeholder="---", text="Hello world"
    )
    assert res1 is None

    # 2. Text containing placeholder -> should be returned
    res2 = reader._surface_page_placeholder(
        scan=False, placeholder="---", text="Hello\n---\nWorld"
    )
    assert res2 == "---"

    # 3. Scanned -> Always returned
    res3 = reader._surface_page_placeholder(
        scan=True, placeholder="---", text="Irrelevant"
    )
    assert res3 == "---"

    # 4. Placeholder has % (unsafe) -> None
    res4 = reader._surface_page_placeholder(
        scan=True, placeholder="--%--", text="Irrelevant"
    )
    assert res4 is None
