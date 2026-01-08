import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import requests
import yaml
from pandas import DataFrame
from requests import Response

from ...model import BaseVisionModel
from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DEFAULT_IMAGE_EXTRACTION_PROMPT,
    DEFAULT_IMAGE_PLACEHOLDER,
    DEFAULT_PAGE_PLACEHOLDER,
    SUPPORTED_PROGRAMMING_LANGUAGES,
    SUPPORTED_VANILLA_IMAGE_EXTENSIONS,
    VANILLA_TXT_FILES_EXTENSIONS,
    ReaderOutput,
)
from ...schema.exceptions import (
    HtmlConversionError,
    ReaderConfigException,
    VanillaReaderException,
)
from ..base_reader import BaseReader
from ..utils import PDFPlumberReader
from ..utils.html_to_markdown import HtmlToMarkdown


class VanillaReader(BaseReader):
    """
    Read multiple file types using Python's built-in and standard libraries.

    Supported formats include: .json, .html/.htm, .txt, .xml, .yaml/.yml,
    .csv, .tsv, .parquet, .pdf, and various image formats.
    """

    def __init__(self, model: Optional[BaseVisionModel] = None) -> None:
        """
        Initialize the VanillaReader.

        Args:
            model (Optional[BaseVisionModel]): A vision-capable model used for
                image captioning, scanned PDF processing, or image file analysis.
                Defaults to None.
        """
        super().__init__()
        self.model = model
        self.pdf_reader = PDFPlumberReader()

    # ---- Public method ---- #

    def read(
        self,
        file_path: str | Path = None,
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Read a document from a file path, URL, or raw content.

        This method supports local files, URLs, JSON objects, or raw text strings.
        Priority of sources: ``kwargs['file_path']`` > ``file_path`` (arg) >
        ``kwargs['file_url']`` > ``kwargs['json_document']`` > ``kwargs['text_document']``.

        Args:
            file_path (str | Path, optional): Path to the input file (local path) or a URL.
            **kwargs: Configuration options. Common keys include:
                file_path (str | Path): Same as the positional arg; takes precedence if provided.
                file_url (str): HTTPS/HTTP URL to read from.
                json_document (dict | str): JSON-like document or JSON string.
                text_document (str): Raw text content (auto-detects JSON/YAML if possible).
                document_name (str): Name to use when input is json_document/text_document or fallback.
                document_id (str): Explicit ID for the output document.
                metadata (dict): Metadata to attach to the output.
                html_to_markdown (bool): If True, convert HTML to Markdown.
                scan_pdf_pages (bool): If True, rasterize PDF pages for VLM processing.
                resolution (int): DPI for scan_pdf_pages rasterization (default 300).
                model (BaseVisionModel): Override model for specific calls.
                prompt (str): Prompt for image/page description.
                vlm_parameters (dict): Extra kwargs forwarded to the vision model.
                image_placeholder (str): Placeholder inserted for images when extracting PDFs.
                show_base64_images (bool): If True, include base64 images in extracted PDF output.
                page_placeholder (str): Placeholder used for PDF page separation / surfacing.
                as_table (bool): If True, read Excel via pandas and return CSV text (first sheet).
                excel_engine (str): pandas.read_excel engine (default "openpyxl").
                parquet_engine (str | None): pandas.read_parquet engine override.

        Returns:
            ReaderOutput: A standardized result containing the extracted text,
            metadata, and processing details.

        Raises:
            ReaderConfigException: If arguments are invalid (e.g., malformed URL,
                unsupported extension, missing required model).
            VanillaReaderException: If an error occurs during file I/O, parsing,
                PDF extraction, or external tool execution (e.g., LibreOffice).
            HtmlConversionError: If HTML-to-Markdown conversion fails.
        """
        try:
            source_type, source_val = _guess_source(kwargs, file_path)

            name, path, text, conv, ocr = self._dispatch_source(
                source_type, source_val, kwargs
            )

            page_ph: str = kwargs.get("page_placeholder", DEFAULT_PAGE_PLACEHOLDER)
            page_ph_out: str | None = self._surface_page_placeholder(
                scan=bool(kwargs.get("scan_pdf_pages")),
                placeholder=page_ph,
                text=text,
            )

            return ReaderOutput(
                text=_ensure_str(text),
                document_name=name,
                document_path=path or "",
                document_id=kwargs.get("document_id", str(uuid.uuid4())),
                conversion_method=conv,
                reader_method="vanilla",
                ocr_method=ocr,
                page_placeholder=page_ph_out,
                metadata=kwargs.get("metadata", {}),
            )

        except (ReaderConfigException, VanillaReaderException, HtmlConversionError):
            raise
        except Exception as e:
            raise VanillaReaderException(
                f"Unexpected error processing document: {e}"
            ) from e

    # ---- Internal helpers ---- #

    def _dispatch_source(  # noqa: WPS231
        self,
        src_type: str,
        src_val: Any,
        kw: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Any, str, Optional[str]]:
        """Route the request to a specialised handler based on source type."""
        handlers: dict[str, callable] = {
            "file_path": self._handle_local_path,
            "file_url": self._handle_url,
            "json_document": self._handle_explicit_json,
            "text_document": self._handle_explicit_text,
        }
        if src_type not in handlers:
            raise ReaderConfigException(f"Unrecognized document source: {src_type}")

        return handlers[src_type](src_val, kw)

    # ---- individual strategies below – each ~20 lines or fewer ---------- #

    # 1) Local / drive paths
    def _handle_local_path(
        self,
        path_like: str | Path,
        kw: Dict[str, Any],
    ) -> Tuple[str, str, Any, str, Optional[str]]:
        """Handle content loading from a local filesystem path."""
        if path_like is None:
            raise ReaderConfigException("file_path cannot be None.")

        path_str: str | Path = (
            os.fspath(path_like) if isinstance(path_like, Path) else path_like
        )

        if not isinstance(path_str, str):
            raise ReaderConfigException("file_path must be a string or Path object.")

        if self.is_url(path_str):
            return self._handle_url(path_str, kw)

        if not self.is_valid_file_path(path_str):
            return self._handle_fallback(path_str, kw)

        ext: str = os.path.splitext(path_str)[1].lower().lstrip(".")
        doc_name: str = os.path.basename(path_str)

        try:
            rel_path = os.path.relpath(path_str)
        except ValueError:
            rel_path = path_str

        try:
            if ext == "pdf":
                return (
                    doc_name,
                    rel_path,
                    *self._process_pdf(path_str, kw),
                )
            if ext == "html" or ext == "htm":
                content, conv = _read_html_file(
                    path_str, html_to_markdown=bool(kw.get("html_to_markdown", False))
                )
                return doc_name, rel_path, content, conv, None
            if ext in VANILLA_TXT_FILES_EXTENSIONS:
                return doc_name, rel_path, _read_text_file(path_str, ext), ext, None
            if ext == "parquet":
                return (
                    doc_name,
                    rel_path,
                    _read_parquet(path_str, engine=kw.get("parquet_engine")),
                    "csv",
                    None,
                )
            if ext in ("yaml", "yml"):
                return doc_name, rel_path, _read_text_file(path_str, ext), "json", None
            if ext in ("xlsx", "xls"):
                if kw.get("as_table", False):
                    excel_engine = kw.get("excel_engine", "openpyxl")
                    return (
                        doc_name,
                        rel_path,
                        _read_excel(path_str, engine=excel_engine),
                        ext,
                        None,
                    )
                pdf_path = self._convert_office_to_pdf(path_str)
                return (
                    os.path.basename(pdf_path),
                    os.path.relpath(pdf_path),
                    *self._process_pdf(pdf_path, kw),
                )
            if ext in ("docx", "pptx"):
                pdf_path = self._convert_office_to_pdf(path_str)
                return (
                    os.path.basename(pdf_path),
                    os.path.relpath(pdf_path),
                    *self._process_pdf(pdf_path, kw),
                )
            if ext in SUPPORTED_VANILLA_IMAGE_EXTENSIONS:
                model = kw.get("model", self.model)
                prompt = kw.get("prompt", DEFAULT_IMAGE_EXTRACTION_PROMPT)
                vlm_parameters = kw.get("vlm_parameters", {})
                return self._handle_image_to_llm(
                    model, path_str, prompt=prompt, vlm_parameters=vlm_parameters
                )
            if ext in SUPPORTED_PROGRAMMING_LANGUAGES:
                return doc_name, rel_path, _read_text_file(path_str, ext), "txt", None

            raise ReaderConfigException(
                f"Unsupported file extension: .{ext}. Please check documentation for supported formats."
            )

        except (VanillaReaderException, ReaderConfigException):
            raise
        except Exception as e:
            raise VanillaReaderException(f"Error reading file '{path_str}': {e}") from e

    # 2) Remote URL
    def _handle_url(
        self,
        url: str,
        kw: Dict[str, Any],
    ) -> Tuple[str, str, Any, str, Optional[str]]:  # noqa: D401
        """Fetch content via HTTP(S) and handle content-type detection."""
        if not isinstance(url, str):
            raise ReaderConfigException("file_url must be a string.")

        if not url.startswith(("http://", "https://")):
            raise ReaderConfigException("file_url must start with http:// or https://")

        content, conv = _load_via_requests(
            url, html_to_markdown=bool(kw.get("html_to_markdown", False))
        )
        name: str = url.split("/")[-1] or "downloaded_file"
        return name, url, content, conv, None

    # 3) Explicit JSON (dict or str)
    def _handle_explicit_json(
        self,
        json_doc: Any,
        _kw: Dict[str, Any],
    ) -> Tuple[str, None, Any, str, None]:
        """Process a JSON object or string passed directly."""
        try:
            return (
                _kw.get("document_name", None),
                None,
                self.parse_json(json_doc),
                "json",
                None,
            )
        except Exception as e:
            raise VanillaReaderException(
                f"Failed to parse provided JSON document: {e}"
            ) from e

    # 4) Explicit raw text
    def _handle_explicit_text(
        self,
        txt: str,
        _kw: Dict[str, Any],
    ) -> Tuple[str, None, Any, str, None]:  # noqa: D401
        """Process raw text, attempting to auto-detect structured formats (JSON/YAML)."""
        for parser, conv in ((self.parse_json, "json"), (yaml.safe_load, "json")):
            try:
                parsed = parser(txt)
                if isinstance(parsed, (dict, list)):
                    return _kw.get("document_name", None), None, parsed, conv, None
            except Exception:
                continue

        return _kw.get("document_name", None), None, txt, "txt", None

    # ----- shared utilities ------------------------------------------------ #

    def _process_pdf(
        self,
        path: str,
        kw: Dict[str, Any],
    ) -> Tuple[Any, str, Optional[str]]:
        """Extract content from a PDF, supporting both text extraction and visual scanning."""
        if kw.get("scan_pdf_pages"):
            model = kw.get("model", self.model)
            if model is None:
                raise ReaderConfigException(
                    "scan_pdf_pages=True requires a vision-capable model (kwarg 'model' or init 'model')."
                )
            joined = self._scan_pdf_pages(path, model=model, **kw)
            return joined, "png", model.model_name

        try:
            content = self.pdf_reader.read(
                path,
                model=kw.get("model", self.model),
                prompt=kw.get("prompt") or DEFAULT_IMAGE_CAPTION_PROMPT,
                show_base64_images=kw.get("show_base64_images", False),
                image_placeholder=kw.get(
                    "image_placeholder", DEFAULT_IMAGE_PLACEHOLDER
                ),
                page_placeholder=kw.get("page_placeholder", DEFAULT_PAGE_PLACEHOLDER),
            )
        except Exception as e:
            raise VanillaReaderException(
                f"PDF extraction failed for {path}: {e}"
            ) from e

        ocr_name: str | None = (
            (kw.get("model") or self.model).model_name
            if kw.get("model") or self.model
            else None
        )
        return content, "pdf", ocr_name

    def _scan_pdf_pages(self, file_path: str, model: BaseVisionModel, **kw) -> str:
        """Rasterize PDF pages and describe them using a vision model."""
        page_ph = kw.get("page_placeholder", DEFAULT_PAGE_PLACEHOLDER)
        try:
            pages: list[str] = self.pdf_reader.describe_pages(
                file_path=file_path,
                model=model,
                prompt=kw.get("prompt") or DEFAULT_IMAGE_EXTRACTION_PROMPT,
                resolution=kw.get("resolution", 300),
                **kw.get("vlm_parameters", {}),
            )
            return "\n\n---\n\n".join(f"{page_ph}\n\n{md}" for md in pages)
        except Exception as e:
            raise VanillaReaderException(
                f"Failed to scan PDF pages for {file_path}: {e}"
            ) from e

    def _handle_fallback(self, raw: str, kw: Dict[str, Any]):
        """Attempt to handle unrecognized sources by trying explicit JSON or text handlers."""
        try:
            return self._handle_explicit_json(raw, kw)
        except Exception:
            try:
                return self._handle_explicit_text(raw, kw)
            except Exception:
                return kw.get("document_name", None), None, raw, "txt", None

    def _handle_image_to_llm(
        self,
        model: BaseVisionModel,
        file_path: str,
        prompt: Optional[str] = None,
        vlm_parameters: Optional[dict] = None,
    ) -> Tuple[str, str, Any, str, str]:
        """Extract information from an image file using a Vision Language Model."""
        if model is None:
            raise ReaderConfigException(
                "No vision model provided for image extraction. Pass 'model' to init or read()."
            )

        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
        except OSError as e:
            raise VanillaReaderException(
                f"Could not read image file {file_path}: {e}"
            ) from e

        ext: str = os.path.splitext(file_path)[1].lstrip(".").lower()
        img_b64: str = base64.b64encode(img_bytes).decode("utf-8")
        prompt: str = prompt or DEFAULT_IMAGE_EXTRACTION_PROMPT
        vlm_parameters: dict[str, Any] = vlm_parameters or {}

        try:
            extracted: str = model.analyze_content(
                img_b64, prompt=prompt, file_ext=ext, **vlm_parameters
            )
        except Exception as e:
            raise VanillaReaderException(f"Vision model analysis failed: {e}") from e

        doc_name: str = os.path.basename(file_path)
        rel_path: str = os.path.relpath(file_path)
        return doc_name, rel_path, extracted, "image", model.model_name

    @staticmethod
    def _surface_page_placeholder(
        scan: bool, placeholder: str, text: Any
    ) -> Optional[str]:
        """Determine if the page placeholder should be exposed in the output text."""
        if "%" in placeholder:
            return None
        txt: str = _ensure_str(text)
        return placeholder if (scan or placeholder in txt) else None

    def _convert_office_to_pdf(self, file_path: str) -> str:
        """Convert a Microsoft Office document to PDF using a headless LibreOffice process."""
        if not shutil.which("soffice"):
            raise VanillaReaderException(
                "LibreOffice/soffice is required for Office-to-PDF conversion "
                "but was not found in PATH."
            )

        try:
            outdir: str = tempfile.mkdtemp(prefix="vanilla_office2pdf_")
        except OSError as e:
            raise VanillaReaderException(
                f"Failed to create temp directory for PDF conversion: {e}"
            ) from e

        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            outdir,
            file_path,
        ]

        try:
            proc: CompletedProcess[bytes] = subprocess.run(
                cmd, capture_output=True, check=False
            )
        except Exception as e:
            raise VanillaReaderException(
                f"Subprocess failed when executing LibreOffice: {e}"
            ) from e

        if proc.returncode != 0:
            err_msg = proc.stderr.decode() if proc.stderr else "Unknown error"
            raise VanillaReaderException(
                f"LibreOffice failed converting {file_path} -> PDF. Exit code {proc.returncode}.\nError: {err_msg}"
            )

        pdf_name: str = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        pdf_path: str = os.path.join(outdir, pdf_name)

        if not os.path.exists(pdf_path):
            raise VanillaReaderException(
                f"LibreOffice finished, but expected PDF was not found at: {pdf_path}"
            )

        return pdf_path


# -------- Helpers --------- #


def _ensure_str(val: Any) -> str:
    """Convert a value to a string, preferring JSON/YAML formatting for structures."""
    if isinstance(val, (dict, list)):
        for dumper in (
            lambda v: json.dumps(v, indent=2, ensure_ascii=False),
            lambda v: yaml.safe_dump(v, allow_unicode=True),
        ):
            try:
                return dumper(val)
            except Exception:
                pass
    return "" if val is None else str(val)


def _guess_source(
    kwargs: Dict[str, Any], file_path: Union[str, Path]
) -> Tuple[str, Any]:
    """Identify the input source key and value from arguments."""
    for key in ("file_path", "file_url", "json_document", "text_document"):
        if kwargs.get(key) is not None:
            return key, kwargs[key]
    return "file_path", file_path


def _read_text_file(path: Union[str, Path], ext: str) -> str:
    """Read a simple text file, optionally parsing and dumping YAML for consistency."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if ext not in ("yaml", "yml"):
                return fh.read()

            try:
                content = yaml.safe_load(fh)
            except yaml.YAMLError as e:
                raise VanillaReaderException(f"Invalid YAML file {path}: {e}") from e

            return yaml.safe_dump(content, allow_unicode=True)

    except OSError as e:
        raise VanillaReaderException(f"Failed to read file {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise VanillaReaderException(f"Encoding error reading {path}: {e}") from e


def _read_html_file(
    path: Union[str, Path], *, html_to_markdown: bool
) -> Tuple[str, str]:
    """Read an HTML file and optionally convert it to Markdown."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: str = fh.read()
    except OSError as e:
        raise VanillaReaderException(f"Failed to read HTML file {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise VanillaReaderException(f"Encoding error reading HTML {path}: {e}") from e

    if html_to_markdown:
        try:
            md: str = HtmlToMarkdown().convert(raw)
            return md, "md"
        except Exception as e:
            raise HtmlConversionError(
                f"Failed to convert HTML to Markdown for {path}: {e}"
            ) from e

    return raw, "html"


def _read_parquet(path: Union[str, Path], *, engine: Optional[str] = None) -> str:
    """Read a Parquet file using Pandas and return it as a CSV string."""
    try:
        if engine is None:
            df: DataFrame = pd.read_parquet(path)
        else:
            df: DataFrame = pd.read_parquet(path, engine=engine)
        return df.to_csv(index=False)
    except ImportError as e:
        raise ReaderConfigException(
            f"Parquet engine missing. Install pyarrow or fastparquet. Detail: {e}"
        ) from e
    except Exception as e:
        raise VanillaReaderException(f"Error reading Parquet file {path}: {e}") from e


def _read_excel(path: Union[str, Path], *, engine: str = "openpyxl") -> str:
    """Read an Excel file using Pandas and return the first sheet as a CSV string."""
    try:
        df: dict = pd.read_excel(path, engine=engine)
        return df.to_csv(index=False)
    except ImportError as e:
        raise ReaderConfigException(
            f"Excel engine '{engine}' missing. "
            f"Install openpyxl or check configuration. "
            f"Detail: {e}"
        ) from e
    except Exception as e:
        raise VanillaReaderException(f"Error reading Excel file {path}: {e}") from e


def _load_via_requests(url: str, *, html_to_markdown: bool = False) -> Tuple[Any, str]:
    """Fetch content from a URL and parse it based on the Content-Type header."""
    try:
        resp: Response = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise VanillaReaderException(f"HTTP request failed for {url}: {e}") from e

    ctype: str = (resp.headers.get("Content-Type", "") or "").lower()

    try:
        # JSON
        if "application/json" in ctype or url.endswith(".json"):
            return resp.json(), "json"

        # HTML
        if "text/html" in ctype or url.endswith((".html", ".htm")):
            raw_html = resp.text
            if html_to_markdown:
                try:
                    md: str = HtmlToMarkdown().convert(raw_html)
                    return md, "md"
                except Exception as e:
                    raise HtmlConversionError(
                        f"Failed to convert downloaded HTML from {url}: {e}"
                    ) from e
            return raw_html, "html"

        # YAML
        if "text/yaml" in ctype or url.endswith((".yaml", ".yml")):
            return yaml.safe_load(resp.text), "json"

        # Text
        return resp.text, "txt"

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise VanillaReaderException(
            f"Failed to parse content from {url} as {ctype}: {e}"
        ) from e


class SimpleHTMLTextExtractor(HTMLParser):
    """Legacy helper to extract raw text from HTML by concatenating data nodes."""

    def __init__(self):
        super().__init__()
        self.text_parts: list = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts).strip()
