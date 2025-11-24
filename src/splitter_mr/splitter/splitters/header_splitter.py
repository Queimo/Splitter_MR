import re
import warnings
from typing import List, Optional, Sequence, Tuple, cast

from bs4 import BeautifulSoup
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ...reader.utils import HtmlToMarkdown
from ...schema import (
    ALLOWED_HEADERS,
    FiletypeAmbiguityWarning,
    HeaderLevelOutOfRangeError,
    HtmlConversionError,
    InvalidHeaderNameError,
    NormalizationError,
    ReaderOutput,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
)
from ...schema.constants import ALLOWED_HEADERS_LITERAL as HeaderName
from ..base_splitter import BaseSplitter


class HeaderSplitter(BaseSplitter):
    """Split HTML or Markdown documents into chunks by header levels (H1–H6).

    - If the input looks like HTML, it is first converted to Markdown using the
      project's HtmlToMarkdown utility, which emits ATX-style headings (`#`, `##`, ...).
    - If the input is Markdown, Setext-style headings (underlines with `===` / `---`)
      are normalized to ATX so headers are reliably detected.
    - Splitting is performed with LangChain's MarkdownHeaderTextSplitter.
    - If no headers are detected after conversion/normalization, a safe fallback
      splitter (RecursiveCharacterTextSplitter) is used to avoid returning a single,
      excessively large chunk.

    Args:
        chunk_size: Size hint for fallback splitting; not used by header splitting itself.
        headers_to_split_on: Semantic header names like ``("Header 1", "Header 2")``.
            If ``None`` (default), all allowed headers are enabled (``ALLOWED_HEADERS``).
        group_header_with_content: If ``True`` (default), headers are kept with their
            following content (``strip_headers=False``). If ``False``, headers are removed
            from the chunks (``strip_headers=True``).

    Raises:
        InvalidHeaderNameError: If any header is not present in ``ALLOWED_HEADERS``.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        headers_to_split_on: Optional[Sequence[HeaderName]] = None,
        *,
        group_header_with_content: bool = True,
    ):
        super().__init__(chunk_size)

        # Use immutable default and validate any user-supplied values.
        if headers_to_split_on is None:
            safe_headers: Tuple[HeaderName, ...] = cast(
                Tuple[HeaderName, ...], ALLOWED_HEADERS
            )
        else:
            safe_headers = self._validate_headers(headers_to_split_on)

        self.headers_to_split_on: Tuple[HeaderName, ...] = safe_headers
        self.group_header_with_content = bool(group_header_with_content)

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Perform header-based splitting with HTML→Markdown conversion and safe fallback.

        Steps:
            1. Detect filetype (HTML/MD).
            2. If HTML, convert to Markdown with HtmlToMarkdown (emits ATX headings).
            3. If Markdown, normalize Setext headings to ATX.
            4. Split by headers via MarkdownHeaderTextSplitter.
            5. If no headers found, fallback to RecursiveCharacterTextSplitter.

        Args:
            reader_output: The reader output containing text and metadata.

        Returns:
            SplitterOutput: A populated splitter output with chunk contents and metadata.

        Warnings:
            SplitterInputWarning: if text field in ReaderOutput is missing or void.

        Raises:
            HtmlConversionError: if HTML Conversion fails.

        Example:
            Basic Markdown input with **default headers** (H1–H6), keeping headers with content:

            ```python
            from splitter_mr.splitter import HeaderSplitter
            from splitter_mr.schema.models import ReaderOutput

            md = (
                "# Title\\n"
                "Intro paragraph.\\n\\n"
                "## Section A\\n"
                "Content A.\\n\\n"
                "## Section B\\n"
                "Content B."
            )
            ro = ReaderOutput(text=md, document_name="example.md")

            splitter = HeaderSplitter(group_header_with_content=True)  # keep headers in chunks
            out = splitter.split(ro)
            print(out.chunks)
            ```
            ```python
            [
                "# Title\\nIntro paragraph.",
                "## Section A\\nContent A.",
                "## Section B\\nContent B."
            ]
            ```

            HTML input with a **restricted set of headers and stripping headers** from chunks:

            ```python
            html = (
                "<h1>Title</h1>"
                "<p>Intro paragraph.</p>"
                "<h2>Section A</h2>"
                "<p>Content A.</p>"
                "<h3>Sub A.1</h3>"
                "<p>Detail A.1</p>"
            )
            ro = ReaderOutput(text=html, document_name="example.html")

            # Only split on Header 1 and Header 2 (i.e., H1/H2)
            splitter = HeaderSplitter(
                headers_to_split_on=("Header 1", "Header 2"),
                group_header_with_content=False  # drop headers from chunks
            )
            out = splitter.split(ro)
            print(out.chunks)
            ```
            ```python
            [
                "Intro paragraph.",
                "Content A.\\nSub A.1\\nDetail A.1"
            ]
            ```
        """
        text: str = reader_output.text
        if text is None or not str(text).strip():
            warnings.warn(
                SplitterInputWarning(
                    "ReaderOutput.text is empty or whitespace-only. "
                    "Proceeding; this will yield a single empty chunk."
                )
            )
            chunks: list[str] = [""]
            return SplitterOutput(
                chunks=chunks,
                chunk_id=self._generate_chunk_ids(len(chunks)),
                document_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="header_splitter",
                split_params={
                    "headers_to_split_on": list(self.headers_to_split_on),
                    "group_header_with_content": self.group_header_with_content,
                },
                metadata=self._default_metadata(),
            )

        filetype: str = self._guess_filetype(reader_output)
        tuples: list[tuple] = self._make_tuples("md")

        text: str = reader_output.text

        # HTML → Markdown using the project's converter
        if filetype == "html":
            try:
                text: str = HtmlToMarkdown().convert(text)
            except Exception as e:
                raise HtmlConversionError(
                    f"HTML→Markdown failed for {reader_output.document_name!r}"
                ) from e
        else:
            text: str = self._normalize_setext(text)

        # Detect presence of ATX headers (after conversion/normalization)
        has_headers: bool = bool(re.search(r"(?m)^\s*#{1,6}\s+\S", text))

        # Configure header splitter. group_header_with_content -> strip_headers False
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=tuples,
            return_each_line=False,
            strip_headers=not self.group_header_with_content,
        )

        docs: list[str] = splitter.split_text(text) if has_headers else []
        # Fallback if no headers were found
        if not docs:
            rc = RecursiveCharacterTextSplitter(
                chunk_size=max(1, int(self.chunk_size) or 1000),
                chunk_overlap=min(200, max(0, int(self.chunk_size) // 10)),
            )
            docs: list = rc.create_documents([text])

        chunks: list[str] = [doc.page_content for doc in docs]

        return SplitterOutput(
            chunks=chunks,
            chunk_id=self._generate_chunk_ids(len(chunks)),
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="header_splitter",
            split_params={
                "headers_to_split_on": list(self.headers_to_split_on),
                "group_header_with_content": self.group_header_with_content,
            },
            metadata=self._default_metadata(),
        )

    # ---- Helpers ---- #

    @staticmethod
    def _validate_headers(headers: Sequence[str]) -> Tuple[HeaderName, ...]:
        """Validate that headers are a subset of ALLOWED_HEADERS and return an immutable tuple.

        Args:
            headers: Proposed list/tuple of header names.

        Returns:
            A tuple of validated header names.

        Raises:
            InvalidHeaderNameError: If any header is not present in ``ALLOWED_HEADERS``.
        """
        invalid: list = [h for h in headers if h not in ALLOWED_HEADERS]
        if invalid:
            allowed_display: str = ", ".join(ALLOWED_HEADERS)
            bad_display: str = ", ".join(invalid)
            raise InvalidHeaderNameError(
                f"Invalid headers: [{bad_display}]. "
                f"Allowed values are: [{allowed_display}]."
            )
        # Preserve caller order but store immutably.
        return cast(Tuple[HeaderName, ...], tuple(headers))

    def _make_tuples(self, filetype: str) -> List[Tuple[str, str]]:
        """Convert semantic header names (e.g., ``"Header 2"``) into Markdown tokens.

        Args:
            filetype: Only ``"md"`` is supported (HTML is converted to MD first).

        Returns:
            Tuples of ``(header_token, semantic_name)``, e.g., ``("##", "Header 2")``.

        Raises:
            SplitterConfigException: If an unsupported filetype is provided.
        """
        tuples: list[tuple[str, str]] = []
        for header in self.headers_to_split_on:
            lvl = self._header_level(header)
            if filetype == "md":
                tuples.append(("#" * lvl, header))
            else:
                raise SplitterConfigException(f"Unsupported filetype: {filetype!r}")
        return tuples

    @staticmethod
    def _header_level(header: str) -> int:
        """Extract numeric level from a header name like ``"Header 2"``.

        Args:
            header: The header label.

        Returns:
            The numeric level extracted from the header label.

        Raises:
            InvalidHeaderNameError: If the header string is not of the expected form.
            HeaderLevelOutOfRangeError: if header level is greater than 7 or lower than 0.
        """
        m = re.match(r"header\s*(\d+)", header.lower())
        if not m:
            raise InvalidHeaderNameError(f"Expected 'Header N', got: {header!r}")
        level = int(m.group(1))
        if not 1 <= level <= 7:
            raise HeaderLevelOutOfRangeError(
                f"Header level {level} out of range [1..7]"
            )
        return level

    @staticmethod
    def _guess_filetype(reader_output: ReaderOutput) -> str:
        """Heuristically determine whether the input is HTML or Markdown.

        The method first checks the filename extension, then uses lightweight HTML
        detection via BeautifulSoup as a fallback.

        Args:
            reader_output: The input document and metadata.

        Returns:
            ``"html"`` if the text appears to be HTML, otherwise ``"md"``.

        Warnings:
            FiletypeAmbiguityWarning: warned if file extension and suggested
            DOM shape does not match.
        """
        name: str = (reader_output.document_name or "").lower()
        md_ext: str = "md" if name.endswith((".md", ".markdown")) else None
        ext_hint: str = "html" if name.endswith((".html", ".htm")) else md_ext

        text: str = reader_output.text or ""
        soup = BeautifulSoup(text, "html.parser")
        dom_hint: str = (
            "html"
            if (
                soup.find("html")  # noqa: W503
                or soup.find(re.compile(r"^h[1-6]$"))  # noqa: W503
                or soup.find("div")  # noqa: W503
            )
            else "md"
        )

        if ext_hint and ext_hint != dom_hint:
            warnings.warn(
                FiletypeAmbiguityWarning(
                    f"Filetype heuristics disagree for {name!r}: "
                    f"extension suggests {ext_hint}, DOM suggests {dom_hint}. "
                    f"Proceeding with {dom_hint}."
                )
            )
        return dom_hint

    @staticmethod
    def _normalize_setext(md_text: str) -> str:
        """Normalize Setext-style headings to ATX so MarkdownHeaderTextSplitter
        can detect them.

        Transformations:
            - ``H1:  Title\\n====  →  # Title``
            - ``H2:  Title\\n----  →  ## Title``

        Args:
            md_text: Raw Markdown text possibly containing Setext headings.

        Returns:
            Markdown text with Setext headings rewritten as ATX headings.

        Raises:
            NormalizationError: if regular expression normalization fails.
        """
        fence: re.Pattern = re.compile(r"(^```.*?$)(.*?)(^```$)", flags=re.M | re.S)
        placeholders: list[str] = []

        def _stash(m: re.Match) -> str:
            placeholders.append(m.group(0))
            return f"__CODEFENCE_PLACEHOLDER_{len(placeholders) - 1}__"

        try:
            protected: str = re.sub(fence, _stash, md_text)
        except re.error as e:
            raise NormalizationError(f"Failed to scan code fences: {e}") from e

        # Normalize setext in the protected text only (outside fences)
        try:
            protected: str = re.sub(
                r"^(?P<t>[^\n]+)\n=+\s*$", r"# \g<t>", protected, flags=re.M
            )
            protected: str = re.sub(
                r"^(?P<t>[^\n]+)\n-+\s*$", r"## \g<t>", protected, flags=re.M
            )
        except re.error as e:
            raise NormalizationError(f"Setext→ATX normalization failed: {e}") from e

        # Restore code fences
        def _unstash(match: re.Match) -> str:
            idx: int = int(match.group(1))
            return placeholders[idx]

        try:
            normalized: str = re.sub(
                r"__CODEFENCE_PLACEHOLDER_(\d+)__", _unstash, protected
            )
        except Exception as e:
            raise NormalizationError(
                "Failed to restore code fences after normalization"
            ) from e

        if re.search(r"^[^\n]+\n[=-]{2,}\s*$", normalized, flags=re.M):
            raise NormalizationError(
                "Unnormalized Setext headings remain after normalization"
            )

        return normalized
