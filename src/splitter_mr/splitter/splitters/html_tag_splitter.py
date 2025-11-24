import copy
import warnings
from collections import Counter, defaultdict
from typing import Final, List, Optional, Sequence

import bs4
from bs4 import BeautifulSoup
from bs4.element import Tag

from ...reader.utils import html_to_markdown
from ...schema import (
    TABLE_CHILDREN,
    AutoTagFallbackWarning,
    BatchHtmlTableWarning,
    HtmlConversionError,
    InvalidHtmlTagError,
    ReaderOutput,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
    SplitterOutputException,
    SplitterOutputWarning,
)
from ..base_splitter import BaseSplitter

HTML_PARSER: Final[str] = "html.parser"
DEFAULT_HTML_TAG: Final[str] = "div"


class HTMLTagSplitter(BaseSplitter):
    """Split HTML content by tag, with optional batching and Markdown conversion.

    Behavior:
      - When `tag` is provided (e.g., `div`), split by all matching elements.
      - When `tag` is `None`, auto-detect the most frequent and shallowest tag.
      - Tables receive special handling to preserve header context when batching.

    Args:
        chunk_size: Maximum chunk size in characters for batching. If `0`, `1`,
            or `None`, batching groups all elements into a single chunk.
        tag: HTML tag to split on (e.g., `"div"`). If `None`, the tag is auto-detected.
        batch: If True, group elements up to `chunk_size`. If False, emit one chunk per element.
        to_markdown: If True, convert each emitted chunk from HTML to Markdown.

    Raises:
        SplitterConfigException: If `chunk_size` is negative or non-integer, or if
            `tag` is a non-string/empty string.
    """

    def __init__(
        self,
        chunk_size: int = 1,
        tag: Optional[str] = None,
        *,
        batch: bool = True,
        to_markdown: bool = True,
    ):
        super().__init__(chunk_size)
        if chunk_size is not None and (
            not isinstance(chunk_size, int) or chunk_size < 0
        ):
            raise SplitterConfigException(
                f"chunk_size must be a non-negative int or None, got {chunk_size!r}"
            )
        self.tag = tag
        if self.tag is not None and (
            not isinstance(self.tag, str) or not self.tag.strip()
        ):
            raise SplitterConfigException(f"Invalid tag: '{self.tag!r}'")
        self.batch = batch
        self.to_markdown = to_markdown

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Split HTML using the configured tag and batching, then optionally convert to Markdown.

        Semantics:
          - **Tables**
              - `batch=False`: one chunk per requested element. If splitting by a row-level tag
                (e.g., `tr`), emit a mini-table per row with `<thead>` once and that row in `<tbody>`.
              - `batch=True` and `chunk_size in (0, 1, None)`: all tables grouped into one chunk.
              - `batch=True` and `chunk_size > 1`: split each table into multiple chunks by batching
                `<tr>` rows; copy `<thead>` into every chunk and skip the header row from `<tbody>`.

          - **Non-table tags**
              - `batch=False`: one chunk per element.
              - `batch=True` and `chunk_size in (0, 1, None)`: all elements grouped into one chunk.
              -`batch=True` and `chunk_size > 1`: batch by total HTML length.

        Args:
          reader_output: Reader output containing at least `text`.

        Returns:
          SplitterOutput: The split result with chunks and metadata.

        Raises:
          HtmlConversionError: If parsing the HTML or converting chunks to Markdown fails.
          InvalidHtmlTagError: If the tag lookup (`find_all`) fails due to an invalid tag.
          SplitterOutputException: If building the final `SplitterOutput` fails.

        Example:
            **Basic usage** splitting **all `<div>` elements**:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters import HTMLTagSplitter

            html = '''
            <div>First block</div>
            <div>Second block</div>
            <div>Third block</div>
            '''

            ro = ReaderOutput(
                text=html,
                document_name="sample.html",
                document_path="/tmp/sample.html",
            )

            splitter = HTMLTagSplitter(chunk_size=10, tag="div", batch=False)
            output = splitter.split(ro)

            print(output.chunks)
            ```

            ```python
            ['<div>First block</div>','<div>Second block</div>','<div>Third block</div>']
            ```

            Example with **batching** (all `<p>` elements grouped into one chunk)::

            ```python
            html = "<p>A</p><p>B</p><p>C</p>"
            ro = ReaderOutput(text=html, document_name="demo.html")

            splitter = HTMLTagSplitter(chunk_size=1, tag="p", batch=True)
            out = splitter.split(ro)

            print(out.chunks[0])
            ```

            ```python
            '<p>A</p>\\n<p>B</p>\\n<p>C</p>'
            ```

            Example with **table batching** (each chunk contains a header and 2 rows):

            ```python
            html = '''
            <table>
                <thead><tr><th>H1</th><th>H2</th></tr></thead>
                <tbody>
                    <tr><td>A</td><td>1</td></tr>
                    <tr><td>B</td><td>2</td></tr>
                    <tr><td>C</td><td>3</td></tr>
                </tbody>
            </table>
            '''

            ro = ReaderOutput(text=html, document_name="table.html")

            splitter = HTMLTagSplitter(
                chunk_size=2,       # batch <tr> rows in groups of 2
                tag="tr",           # split by table rows
                batch=True,
            )
            out = splitter.split(ro)

            for i, c in enumerate(out.chunks, 1):
                print(f"--- CHUNK {i} ---")
                print(c)
            ```

            Example **enabling Markdown conversion**:

            ```python
            html = "<h1>Title</h1><p>Paragraph text</p>"
            ro = ReaderOutput(text=html)

            splitter = HTMLTagSplitter(
                chunk_size=5,
                tag=None,
                batch=False,
                to_markdown=True,
            )
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            ['# Title', 'Paragraph text']
            ```

        Notes:
          If the input text is empty/whitespace-only, a warning is emitted and
          a single empty chunk is returned.
        """
        html: str = getattr(reader_output, "text", "") or ""
        if not html.strip():
            warnings.warn(
                SplitterInputWarning(
                    "ReaderOutput.text is empty or whitespace-only. "
                    "Proceeding; this will yield a single empty chunk."
                )
            )
            return self._emit_result(
                chunks=[""],
                reader_output=reader_output,
                tag=self.tag or DEFAULT_HTML_TAG,
            )

        soup = self._parse_html(html)
        tag = self.tag or self._auto_tag(soup)

        elements, effective_tag = self._select_elements(soup, tag)

        chunks = self._dispatch_chunking(elements, effective_tag)
        if not chunks:
            warnings.warn(SplitterOutputWarning("Splitter has produced empty chunks"))
            chunks = [""]

        if self.to_markdown:
            chunks = self._convert_chunks_to_markdown(chunks)

        return self._emit_result(
            chunks=chunks,
            reader_output=reader_output,
            tag=effective_tag,
        )

    # ---- Helpers ---- #

    def _parse_html(self, html: str) -> bs4.BeautifulSoup:
        """Parse HTML into a BeautifulSoup document.

        Args:
          html: Raw HTML string.

        Returns:
          BeautifulSoup: Parsed document.

        Raises:
          HtmlConversionError: If parsing fails.
        """
        try:
            return bs4.BeautifulSoup(html, HTML_PARSER)
        except Exception as e:
            raise HtmlConversionError(f"BeautifulSoup failed to parse HTML: {e}") from e

    def _select_elements(self, soup: BeautifulSoup, tag: str) -> tuple[list, str]:
        """Select elements by tag and handle table escalation for batching.

        Args:
          soup: Parsed BeautifulSoup document.
          tag: Tag to search for.

        Returns:
          tuple[list, str]: `(elements, effective_tag)`. `effective_tag` may be
          `"table"` if row-level tags are escalated to tables for batching.

        Raises:
          InvalidHtmlTagError: If the selection fails (BeautifulSoup `find_all` error).
        """
        try:
            elements = soup.find_all(tag)
            if not elements:
                warnings.warn(
                    AutoTagFallbackWarning(f"No elements found for tag {tag!r}")
                )
        except Exception as e:
            raise InvalidHtmlTagError(
                f"find_all method has failed when locating {tag!r} on document."
            ) from e

        # Escalate row-level/table-children to tables when batching
        effective_tag = tag
        if self.batch and tag in TABLE_CHILDREN and elements:
            warnings.warn(
                BatchHtmlTableWarning(
                    "Batch process has been detected. "
                    "It will be split by elements in HTML table."
                )
            )
            seen = set()
            parent_tables = []
            for el in elements:
                table = el.find_parent("table")
                if table and id(table) not in seen:
                    seen.add(id(table))
                    parent_tables.append(table)
            if parent_tables:
                elements = parent_tables
                effective_tag = "table"

        return elements, effective_tag

    def _dispatch_chunking(self, elements: list, tag: str) -> List[str]:
        """Dispatch to table or non-table chunking based on tag.

        Args:
          elements: List of matched elements (or parent tables if escalated).
          tag: Effective tag name (possibly `"table"` after escalation).

        Returns:
          List[str]: HTML chunks.
        """
        if tag == "table":
            return self._chunk_tables(elements)
        return self._chunk_non_tables(elements, tag)

    def _chunk_tables(self, tables: list) -> List[str]:
        """Chunk table elements according to batching rules.

        Args:
          tables: List of `<table>` elements.

        Returns:
          List[str]: Table chunks as HTML strings.

        Raises:
          HtmlConversionError: Indirectly, if HTML-to-Markdown conversion is later applied.
        """
        if not self.batch:
            return [self._build_doc_with_children([el]) for el in tables]

        if self.chunk_size in (0, 1, None):
            return [self._build_doc_with_children(tables)] if tables else [""]

        # chunk_size > 1: batch rows within each table
        chunks: list = []
        for table_el in tables:
            _, rows, _ = self._extract_table_header_and_rows(table_el)
            if not rows:
                chunks.append(self._build_doc_with_children([table_el]))
                continue

            buf: list = []
            for row in rows:
                test_buf = buf + [row]
                test_html = self._build_table_chunk(table_el, test_buf)
                if len(test_html) > self.chunk_size and buf:
                    chunks.append(self._build_table_chunk(table_el, buf))
                    buf = [row]
                else:
                    buf = test_buf
            if buf:
                chunks.append(self._build_table_chunk(table_el, buf))

        return chunks

    def _convert_chunks_to_markdown(self, chunks: List[str]) -> List[str]:
        """Convert a list of HTML chunks to Markdown.

        Args:
          chunks: HTML chunks to convert.

        Returns:
          List[str]: Markdown strings.

        Raises:
          HtmlConversionError: If the conversion fails for any chunk.
        """
        try:
            converter = html_to_markdown.HtmlToMarkdown()
            return [converter.convert(c) for c in chunks]
        except Exception as e:
            raise HtmlConversionError("HTML to Markdown conversion failed") from e

    def _emit_result(
        self, chunks: List[str], reader_output: ReaderOutput, tag: str
    ) -> SplitterOutput:
        """Assemble the SplitterOutput with common metadata.

        Args:
          chunks: Final list of chunks (HTML or Markdown).
          reader_output: Original reader output.
          tag: Effective tag used for splitting (may differ from configured tag).

        Returns:
          SplitterOutput: Structured output including ids and metadata.

        Raises:
          SplitterOutputException: If building the `SplitterOutput` object fails.
        """
        try:
            return SplitterOutput(
                chunks=chunks,
                chunk_id=self._generate_chunk_ids(len(chunks)),
                document_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="html_tag_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "tag": tag,
                    "batch": self.batch,
                    "to_markdown": self.to_markdown,
                },
                metadata=self._default_metadata(),
            )
        except Exception as e:
            raise SplitterOutputException(f"Failed to build SplitterOutput: {e}") from e

    # ---- HTML / Table helpers ---- #

    def _build_doc_with_children(self, children: List) -> str:
        """Wrap top-level nodes into a minimal HTML document.

        Args:
          children: Nodes to append under `<body>`.

        Returns:
          str: Serialized HTML document containing the provided children.
        """
        doc = bs4.BeautifulSoup("", HTML_PARSER)
        html_tag: Tag = doc.new_tag("html")
        body_tag: Tag = doc.new_tag("body")
        html_tag.append(body_tag)
        doc.append(html_tag)
        for c in children:
            body_tag.append(copy.deepcopy(c))
        return str(doc)

    def _extract_table_header_and_rows(self, table_tag):
        """Extract table header and data rows.

        Args:
          table_tag: A `<table>` BeautifulSoup element.

        Returns:
          tuple: `(header_thead, data_rows, header_row_src)` where:
            * `header_thead`: a deep-copied `<thead>` or `None`.
            * `data_rows`: list of original `<tr>` nodes not in `<thead>`.
            * `header_row_src`: original `<tr>` used to synthesize `<thead>` (if any).
        """
        header = table_tag.find("thead")
        header_row_src: None = None

        if header is not None:
            data_rows = []
            for tr in table_tag.find_all("tr"):
                if tr.find_parent("thead") is not None:
                    continue
                data_rows.append(tr)
            return copy.deepcopy(header), data_rows, None

        first_tr = table_tag.find("tr")
        header_thead: None = None
        if first_tr is not None:
            tmp = bs4.BeautifulSoup("", HTML_PARSER)
            thead = tmp.new_tag("thead")
            thead.append(copy.deepcopy(first_tr))
            header_thead = thead
            header_row_src = first_tr

        data_rows: list = []
        for tr in table_tag.find_all("tr"):
            if header_row_src is not None and tr is header_row_src:
                continue
            if tr.find_parent("thead") is not None:
                continue
            data_rows.append(tr)

        return header_thead, data_rows, header_row_src

    def _build_table_chunk(self, table_tag, rows_subset: List) -> str:
        """Build a minimal document containing a single table with a subset of rows.

        Args:
          table_tag: The source `<table>` element (attributes are copied).
          rows_subset: The `<tr>` rows to include under `<tbody>`.

        Returns:
          str: Serialized HTML document with `<table>` containing the subset.
        """
        header_thead, _, _ = self._extract_table_header_and_rows(table_tag)
        doc = BeautifulSoup("", HTML_PARSER)
        html_tag: Tag = doc.new_tag("html")
        body_tag: Tag = doc.new_tag("body")
        html_tag.append(body_tag)
        doc.append(html_tag)

        new_table: Tag = doc.new_tag("table", **table_tag.attrs)
        if header_thead is not None:
            new_table.append(copy.deepcopy(header_thead))

        tbody: Tag = doc.new_tag("tbody")
        for r in rows_subset:
            tbody.append(copy.deepcopy(r))
        new_table.append(tbody)

        body_tag.append(new_table)
        return str(doc)

    def _chunk_non_tables(self, elements: list, tag: str) -> List[str]:
        """Chunk non-table elements according to batching rules.

        Args:
          elements: List of non-table elements to chunk.
          tag: Effective tag name (not `"table"`).

        Returns:
          List[str]: HTML chunks for non-table content.
        """
        if not self.batch:
            return self._non_tables_unbatched(elements, tag)

        if self.chunk_size in (0, 1, None):
            return self._single_group_or_empty(elements)

        # chunk_size > 1: batch by total HTML length
        return self._chunk_by_total_length(elements, self.chunk_size)

    def _build_doc(self, els: Sequence) -> str:
        """Build a minimal HTML document from nodes.

        Args:
          els: Sequence of nodes to be wrapped under `<body>`.

        Returns:
          str: Serialized HTML document containing the nodes.
        """
        return self._build_doc_with_children(list(els))

    def _single_group_or_empty(self, elements: Sequence) -> List[str]:
        """Return a single grouped chunk or an explicit empty chunk.

        Args:
          elements: Sequence of elements to group.

        Returns:
          List[str]: A single combined chunk, or `[""]` if there are no elements.
        """
        return [self._build_doc(elements)] if elements else [""]

    def _non_tables_unbatched(self, elements: list, tag: str) -> List[str]:
        """Unbatched emission for non-table tags (and table-children).

        Args:
          elements: List of elements to emit individually.
          tag: Effective tag name (may be a table-child, e.g., `"tr"`).

        Returns:
          List[str]: One HTML chunk per element (with special handling for table-children).
        """
        # Simple one-per-element when not dealing with table-children
        if tag not in TABLE_CHILDREN:
            return [self._build_doc_with_children([el]) for el in elements]

        # Row-like: keep header context if parent table exists
        chunks: List[str] = []
        for el in elements:
            table_el = el.find_parent("table")
            if not table_el:
                chunks.append(self._build_doc_with_children([el]))
                continue

            # Skip header-only rows or header tags
            if (el.name == "tr" and el.find_parent("thead") is not None) or el.name in {
                "thead",
                "th",
            }:
                continue

            chunks.append(self._build_table_chunk(table_el, [el]))
        return chunks

    def _chunk_by_total_length(self, elements: Sequence, max_len: int) -> List[str]:
        """Batch arbitrary elements by aggregated HTML length.

        Args:
          elements: Sequence of elements to batch.
          max_len: Maximum allowed length (in characters) per chunk.

        Returns:
          List[str]: HTML chunks whose serialized length does not exceed `max_len`,
          except when a single element alone exceeds `max_len` (in which case it is
          emitted as an oversized chunk).
        """
        chunks: list[str] = []
        buffer: list = []
        for el in elements:
            candidate = buffer + [el]
            candidate_str = self._build_doc(candidate)
            if len(candidate_str) > max_len and buffer:
                chunks.append(self._build_doc(buffer))
                buffer = [el]
            else:
                buffer = candidate
        if buffer:
            chunks.append(self._build_doc(buffer))
        return chunks

    # ---- Auto Tagging logic ---- #

    def _auto_tag(self, soup: BeautifulSoup) -> str:
        """Auto-detect the most frequent and shallowest tag within `<body>`.

        If no repeated tags are found, return the first tag found in `<body>`,
        otherwise fallback to `'div'`. Emits an `AutoTagFallbackWarning` when
        `<body>` is missing or when a fallback is used.

        Args:
          soup: Parsed BeautifulSoup document.

        Returns:
          str: Chosen tag name.
        """
        body = soup.find("body")
        if not body:
            warnings.warn(
                AutoTagFallbackWarning(
                    f"No body tag has been found in the provided input. "
                    f"Defaulting to '{DEFAULT_HTML_TAG}' tag"
                )
            )
            return DEFAULT_HTML_TAG

        # Traverse all tags in body, tracking tag: (count, min_depth)
        tag_counter = Counter()
        tag_min_depth = defaultdict(lambda: float("inf"))

        def traverse(el, depth=0):
            for child in el.children:
                if getattr(child, "name", None):
                    tag_counter[child.name] += 1
                    tag_min_depth[child.name] = min(tag_min_depth[child.name], depth)
                    traverse(child, depth + 1)

        traverse(body)

        if not tag_counter:
            for t in body.find_all(True, recursive=True):
                return t.name
            warnings.warn(
                AutoTagFallbackWarning(f"Defaulting to '{DEFAULT_HTML_TAG}' tag")
            )
            return DEFAULT_HTML_TAG

        max_count: int = max(tag_counter.values())
        candidates: list = [t for t, cnt in tag_counter.items() if cnt == max_count]
        chosen: int = min(candidates, key=lambda t: tag_min_depth[t])
        return chosen
