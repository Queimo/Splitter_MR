import warnings
from typing import List, Tuple

from ...schema import (
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
    SplitterOutputException,
    SplitterOutputWarning,
)
from ..base_splitter import BaseSplitter


class PagedSplitter(BaseSplitter):
    """
    Splits a multi-page document into page-based or multi-page chunks using a placeholder marker.

    This splitter uses the ``page_placeholder`` field of :class:`ReaderOutput` to break
    the text into logical "pages" and then groups those pages into chunks. It can also
    introduce character-based overlap between consecutive chunks.

    Args:
        chunk_size (int): Number of pages per chunk.
        chunk_overlap (int): Number of overlapping characters to include from the end
            of the previous chunk.

    Raises:
        SplitterConfigException:
            If ``chunk_size`` is less than 1 or ``chunk_overlap`` is negative.

    Warnings:
        SplitterInputWarning:
            When the input text is empty or whitespace-only.
        SplitterOutputWarning:
            When no non-empty pages are found after splitting on the placeholder and
            the splitter falls back to a single empty chunk.
    """

    def __init__(self, chunk_size: int = 1, chunk_overlap: int = 0):
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise SplitterConfigException(
                "chunk_size must be greater a positive number greater than 1"
            )
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise SplitterConfigException(
                "chunk_overlap must be a positive number greater or equal than 0"
            )

        # Note: PagedSplitter uses `chunk_size` as pages-per-chunk, not characters.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---- Main method --- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split the input text into page-based chunks using the page placeholder.

        The splitting process is:

        1. Validate and normalise the :class:`ReaderOutput` and extract
           ``text`` / ``page_placeholder``.
        2. Split the text into pages using ``page_placeholder``.
        3. Group pages into chunks (with optional character-based overlap).
        4. Build the final :class:`SplitterOutput`.

        Args:
            reader_output (ReaderOutput): The output from a reader containing text,
                metadata, and a ``page_placeholder`` string.

        Returns:
            SplitterOutput: The result with chunks and related metadata.

        Raises:
            ReaderOutputException:
                If ``reader_output`` does not contain a valid ``text`` or
                ``page_placeholder`` field.
            InvalidChunkException:
                If the number of generated ``chunk_id`` values does not match the
                number of chunks.
            SplitterOutputException:
                If constructing :class:`SplitterOutput` fails unexpectedly.

        Warnings:
            SplitterInputWarning:
                When the input text is empty or whitespace-only.
            SplitterOutputWarning:
                When no non-empty pages are found after splitting on the placeholder
                and the splitter falls back to a single empty chunk.

        Example:
            **Basic usage** with a simple placeholder:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters import PagedSplitter

            text = "<!-- page -->Page 1<!-- page -->Page 2<!-- page -->Page 3"
            ro = ReaderOutput(
                text=text,
                page_placeholder="<!-- page -->",
                document_name="demo.txt",
                document_path="/tmp/demo.txt",
            )

            splitter = PagedSplitter(chunk_size=1, chunk_overlap=0)
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            ['Page 1', 'Page 2', 'Page 3']
            ```

            Grouping **multiple pages** into a single chunk:

            ```python
            splitter = PagedSplitter(chunk_size=2)
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            ['Page 1\\nPage 2', 'Page 3']
            ```

            Applying **character-based overlap** between chunks:

            ```python
            text = "<p>One</p><!-- page --><p>Two</p><!-- page --><p>Three</p>"
            ro = ReaderOutput(text=text, page_placeholder="<!-- page -->")

            # Overlap last 5 characters from each previous chunk
            splitter = PagedSplitter(chunk_size=1, chunk_overlap=5)
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            ['<p>One</p>', 'ne</p><p>Two</p>', 'o</p><p>Three</p>']
            ```

            **Metadata propagation**:

            ```python
            ro = ReaderOutput(
                text="<!-- page -->A<!-- page -->B",
                page_placeholder="<!-- page -->",
                document_name="source.txt",
                document_path="/tmp/source.txt",
                document_id="abc123",
            )

            splitter = PagedSplitter(chunk_size=1)
            out = splitter.split(ro)

            print(out.document_name)
            ```
            ```python
            'source.txt'
            ```
            ```python
            print(out.split_method)
            ```
            ```python
            'paged_splitter'
            ```
            ```python
            print(out.split_params)
            ```
            ```python
            {'chunk_size': 1, 'chunk_overlap': 0}
            ```
        """
        text, page_placeholder = self._validate_reader_output(reader_output)
        pages = self._split_into_pages(text, page_placeholder)
        chunks = self._build_chunks(pages)

        try:
            return self._build_output(reader_output, chunks)
        except InvalidChunkException:
            raise
        except (TypeError, ValueError) as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in PagedSplitter: {exc}"
            ) from exc

    # ---- Helpers ---- #

    def _validate_reader_output(self, reader_output: ReaderOutput) -> Tuple[str, str]:
        """
        Validate and normalise the incoming ReaderOutput.

        Ensures that ``page_placeholder`` and ``text`` are present and of the right
        type, and emits input-level warnings when appropriate.

        Raises:
            ReaderOutputException: On missing/invalid fields.
        """
        if not hasattr(reader_output, "page_placeholder"):
            raise ReaderOutputException(
                "ReaderOutput object must expose a 'page_placeholder' attribute."
            )

        page_placeholder = reader_output.page_placeholder
        if not isinstance(page_placeholder, str) or not page_placeholder.strip():
            raise ReaderOutputException(
                "ReaderOutput.page_placeholder must be a non-empty string."
            )

        if not hasattr(reader_output, "text"):
            raise ReaderOutputException(
                "ReaderOutput object must expose a 'text' attribute."
            )

        text = reader_output.text

        if not text.strip():
            warnings.warn(
                "PagedSplitter received empty or whitespace-only text; "
                "resulting chunks will be empty.",
                SplitterInputWarning,
                stacklevel=3,
            )

        return text, page_placeholder

    def _split_into_pages(self, text: str, page_placeholder: str) -> List[str]:
        """
        Split the document text into normalised pages using the placeholder.

        Emits an output-level warning and returns an empty list if no pages could be
        derived.

        Warnings:
            SplitterOutputWarning: When no non-empty pages are found.
        """
        pages: List[str] = [
            page.strip() for page in text.split(page_placeholder) if page.strip()
        ]

        if not pages:
            warnings.warn(
                "PagedSplitter did not find any non-empty pages after splitting; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=3,
            )

        return pages

    def _build_chunks(self, pages: List[str]) -> List[str]:
        """
        Group pages into chunks, applying character-based overlap if configured.

        Guarantees that the returned list is never empty (fallback to ``['']``).
        """
        chunks: List[str] = []

        for i in range(0, len(pages), self.chunk_size):
            chunk = "\n".join(pages[i : i + self.chunk_size])

            if self.chunk_overlap > 0 and i > 0 and chunks:
                overlap_text = chunks[-1][-self.chunk_overlap :]
                chunk = overlap_text + chunk

            if chunk:
                chunks.append(chunk)

        if not chunks:
            chunks = [""]

        return chunks

    def _build_output(
        self,
        reader_output: ReaderOutput,
        chunks: List[str],
    ) -> SplitterOutput:
        """
        Assemble the final :class:`SplitterOutput` and perform consistency checks.

        Raises:
            InvalidChunkException:
                If the number of generated chunk IDs does not match the number of chunks.
        """
        chunk_ids = self._generate_chunk_ids(len(chunks))

        if len(chunk_ids) != len(chunks):
            raise InvalidChunkException(
                "Number of chunk IDs does not match number of chunks "
                f"(chunk_ids={len(chunk_ids)}, chunks={len(chunks)})."
            )

        metadata = self._default_metadata()

        try:
            return SplitterOutput(
                chunks=chunks,
                chunk_id=chunk_ids,
                document_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="paged_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
                metadata=metadata,
            )
        except Exception as exc:
            raise SplitterOutputException(f"Error trying to build response: {exc}")
