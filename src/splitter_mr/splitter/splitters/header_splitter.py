# header_splitter.py

import re
from typing import List, Optional, Sequence, Tuple, cast

from bs4 import BeautifulSoup
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from splitter_mr.schema.constants import ALLOWED_HEADERS
from splitter_mr.schema.constants import ALLOWED_HEADERS_LITERAL as HeaderName

from ...reader.utils import HtmlToMarkdown
from ...schema import ReaderOutput, SplitterOutput
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
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        headers_to_split_on: Optional[Sequence[HeaderName]] = None,
        *,
        group_header_with_content: bool = True,
    ):
        """Initialize the HeaderSplitter.

        Args:
            chunk_size: Used by the fallback character splitter if no headers are found.
            headers_to_split_on: Semantic headers, e.g., ``("Header 1", "Header 2")``.
                Defaults to all allowed levels defined in ``ALLOWED_HEADERS``.
            group_header_with_content: Keep headers attached to following content if ``True``.

        Raises:
            ValueError: If any provided header is not in ``ALLOWED_HEADERS``.
        """
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

    # ---- Helpers ---- #

    @staticmethod
    def _validate_headers(headers: Sequence[str]) -> Tuple[HeaderName, ...]:
        """Validate that headers are a subset of ALLOWED_HEADERS and return an immutable tuple.

        Args:
            headers: Proposed list/tuple of header names.

        Returns:
            A tuple of validated header names.

        Raises:
            ValueError: If any header is not present in ``ALLOWED_HEADERS``.
        """
        invalid: list = [h for h in headers if h not in ALLOWED_HEADERS]
        if invalid:
            allowed_display: str = ", ".join(ALLOWED_HEADERS)
            bad_display: str = ", ".join(invalid)
            raise ValueError(
                f"Invalid headers: [{bad_display}]. Allowed values are: [{allowed_display}]."
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
            ValueError: If an unsupported filetype is provided.
        """
        tuples: list[tuple[str, str]] = []
        for header in self.headers_to_split_on:
            lvl = self._header_level(header)
            if filetype == "md":
                tuples.append(("#" * lvl, header))
            else:
                raise ValueError(f"Unsupported filetype: {filetype!r}")
        return tuples

    @staticmethod
    def _header_level(header: str) -> int:
        """Extract numeric level from a header name like ``"Header 2"``.

        Args:
            header: The header label.

        Returns:
            The numeric level extracted from the header label.

        Raises:
            ValueError: If the header string is not of the expected form.
        """
        m: str | None = re.match(r"header\s*(\d+)", header.lower())
        if not m:
            raise ValueError(f"Invalid header: {header}")
        return int(m.group(1))

    @staticmethod
    def _guess_filetype(reader_output: ReaderOutput) -> str:
        """Heuristically determine whether the input is HTML or Markdown.

        The method first checks the filename extension, then uses lightweight HTML
        detection via BeautifulSoup as a fallback.

        Args:
            reader_output: The input document and metadata.

        Returns:
            ``"html"`` if the text appears to be HTML, otherwise ``"md"``.
        """
        name = (reader_output.document_name or "").lower()
        if name.endswith((".html", ".htm")):
            return "html"
        if name.endswith((".md", ".markdown")):
            return "md"

        soup = BeautifulSoup(reader_output.text or "", "html.parser")
        if soup.find("html") or soup.find(re.compile(r"^h[1-6]$")) or soup.find("div"):
            return "html"
        return "md"

    @staticmethod
    def _normalize_setext(md_text: str) -> str:
        """Normalize Setext-style headings to ATX so MarkdownHeaderTextSplitter can detect them.

        Transformations:
            - ``H1:  Title\\n====  →  # Title``
            - ``H2:  Title\\n----  →  ## Title``

        Args:
            md_text: Raw Markdown text possibly containing Setext headings.

        Returns:
            Markdown text with Setext headings rewritten as ATX headings.
        """
        # H1 underlines
        md_text = re.sub(r"^(?P<t>[^\n]+)\n=+\s*$", r"# \g<t>", md_text, flags=re.M)
        # H2 underlines
        md_text = re.sub(r"^(?P<t>[^\n]+)\n-+\s*$", r"## \g<t>", md_text, flags=re.M)
        return md_text

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Perform header-based splitting with HTML→Markdown conversion and safe fallback.

        Steps:
            1. Detect filetype (HTML/MD).
            2. If HTML, convert to Markdown with HtmlToMarkdown (emits ATX headings).
            3. If Markdown, normalize Setext headings to ATX.
            4. Split by headers via MarkdownHeaderTextSplitter.
            5. If no headers found, fallback to RecursiveCharacterTextSplitter.

        Args:
            reader_output: The reader output containing text and metadata.

        Returns:
            A :class:`SplitterOutput` with chunk contents and metadata.

        Raises:
            ValueError: If ``reader_output.text`` is empty.
        """
        if not reader_output.text:
            raise ValueError("reader_output.text is empty or None")

        filetype: str = self._guess_filetype(reader_output)
        tuples: list[tuple] = self._make_tuples("md")

        text: str = reader_output.text

        # HTML → Markdown using the project's converter
        if filetype == "html":
            text = HtmlToMarkdown().convert(text)
        else:
            # Normalize Setext headings if already Markdown
            text = self._normalize_setext(text)

        # Detect presence of ATX headers (after conversion/normalization)
        has_headers = bool(re.search(r"(?m)^\s*#{1,6}\s+\S", text))

        # Configure header splitter. group_header_with_content -> strip_headers False
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=tuples,
            return_each_line=False,
            strip_headers=not self.group_header_with_content,
        )

        docs = splitter.split_text(text) if has_headers else []
        # Fallback if no headers were found
        if not docs:
            rc = RecursiveCharacterTextSplitter(
                chunk_size=max(1, int(self.chunk_size) or 1000),
                chunk_overlap=min(200, max(0, int(self.chunk_size) // 10)),
            )
            docs = rc.create_documents([text])

        chunks: list = [doc.page_content for doc in docs]

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
