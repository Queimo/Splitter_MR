import re
import warnings
from typing import List, Union

from ...schema import (
    DEFAULT_PARAGRAPH_SEPARATORS,
    ChunkUnderflowWarning,
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


class ParagraphSplitter(BaseSplitter):
    """
    ParagraphSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk contains a specified number of paragraphs, and overlap is defined
    by a number or percentage of words from the end of the previous chunk.

    Args:
        chunk_size (int): Maximum number of paragraphs per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping words
            between chunks. If a float in [0, 1), it is treated as a fraction of the
            maximum paragraph length (in words); otherwise it is interpreted as an
            absolute number of words.
        line_break (Union[str, List[str]]): Character(s) used to split text into
            paragraphs. A single string or a list of strings.

    Raises:
        SplitterConfigException:
            If ``chunk_size`` is less than 1, ``chunk_overlap`` is negative or not
            numeric, or ``line_break`` is neither a non-empty string nor a list of
            non-empty strings.
    """

    def __init__(
        self,
        chunk_size: int = 3,
        chunk_overlap: Union[int, float] = 0,
        line_break: Union[str, List[str]] = DEFAULT_PARAGRAPH_SEPARATORS,
    ):
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise SplitterConfigException(
                "chunk_size must be a positive number greater than 1"
            )

        if not isinstance(chunk_overlap, (int, float)) or chunk_overlap < 0:
            raise SplitterConfigException(
                "chunk_overlap must be a non-negative int or float"
            )

        # Normalise line_break to a list of strings and validate
        if isinstance(line_break, str):
            line_break_list = [line_break]
        elif isinstance(line_break, list):
            line_break_list = line_break
        else:
            raise SplitterConfigException(
                "line_break must be a string or a list of strings"
            )

        if not line_break_list or any(
            not isinstance(lb, str) or not lb for lb in line_break_list
        ):
            raise SplitterConfigException(
                "line_break must contain at least one non-empty string"
            )

        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.line_break = line_break_list

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split the text in ``reader_output.text`` into paragraph-based chunks.

        Pipeline:

        1. Validate and normalise ``reader_output.text``.
        2. Split into paragraphs.
        3. Compute word overlap.
        4. Build chunks (with overlap).
        5. Build the final :class:`SplitterOutput`.

        Args:
            reader_output (ReaderOutput): Dataclass containing at least a ``text`` field
                (str or None) and optional document metadata.

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ReaderOutputException:
                If ``reader_output.text`` is missing or not ``str``/``None``.
            InvalidChunkException:
                If the number of generated chunk IDs does not match the number of chunks.
            SplitterOutputException:
                If constructing :class:`SplitterOutput` fails unexpectedly.

        Warnings:
            SplitterInputWarning:
                When the input text is empty or whitespace-only.
            SplitterOutputWarning:
                When no non-empty paragraphs are found, causing the splitter to fall
                back to a single empty chunk.

        Example:
            **Basic usage** with default line breaks and no overlap:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters import ParagraphSplitter

            text = (
                "First paragraph.\\n\\n"
                "Second paragraph with more text.\\n\\n"
                "Third paragraph."
            )

            ro = ReaderOutput(
                text=text,
                document_name="example.txt",
                document_path="/tmp/example.txt",
                document_id="doc-1",
                conversion_method="text",
                reader_method="plain",
                ocr_method=None,
                metadata={},
            )

            splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
            output = splitter.split(ro)

            print(output.chunks)
            ```

            ```python
            ['First paragraph.\\n\\nSecond paragraph with more text.', 'Third paragraph.']
            ```

            Example with **custom line breaks** and **word overlap** between chunks:

            ```python
            text = (
                "Intro paragraph.@@"
                "Details paragraph one.@@"
                "Details paragraph two.@@"
                "Conclusion paragraph."
            )

            ro = ReaderOutput(text=text, document_name="custom_sep.txt")

            splitter = ParagraphSplitter(
                chunk_size=2,
                chunk_overlap=3, # reuse last 3 words from previous chunk
                line_break="@@", # custom paragraph separator
            )
            output = splitter.split(ro)

            for chunk in output.chunks:
                print("--- CHUNK ---")
                print(chunk)
            ```
        """
        text = self._validate_reader_output(reader_output)
        paragraphs = self._split_into_paragraphs(text)
        overlap = self._compute_overlap(paragraphs)
        chunks = self._build_chunks(paragraphs, overlap)
        return self._build_output(reader_output, chunks)

    # ---- Internal helpers ---- #

    def _validate_reader_output(self, reader_output: ReaderOutput) -> str:
        """
        Validate and normalise ReaderOutput.text.

        Raises:
            ReaderOutputException: On missing or invalid text.
        """
        if not hasattr(reader_output, "text"):
            raise ReaderOutputException(
                "ReaderOutput object must expose a 'text' attribute."
            )

        text = reader_output.text
        if text is None:
            text = ""
        elif not isinstance(text, str):
            raise ReaderOutputException(
                f"ReaderOutput.text must be of type 'str' or None, got "
                f"{type(text).__name__!r}"
            )

        if not text.strip():
            warnings.warn(
                "ParagraphSplitter received empty or whitespace-only text; "
                "resulting chunks will be empty.",
                SplitterInputWarning,
                stacklevel=3,
            )

        return text

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split the input text into normalised paragraphs.

        Warnings:
            SplitterOutputWarning:
                When no non-empty paragraphs are found and the splitter falls back
                to a single empty paragraph.
        """
        pattern = "|".join(map(re.escape, self.line_break))
        paragraphs = [p for p in re.split(pattern, text) if p.strip()]

        if not paragraphs:
            warnings.warn(
                "ParagraphSplitter did not find any non-empty paragraphs; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=3,
            )
            paragraphs = [""]

        return paragraphs

    def _compute_overlap(self, paragraphs: List[str]) -> int:
        """
        Compute the number of overlapping words between chunks based on the
        configured ``chunk_overlap``.
        """
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            max_para_words = max((len(p.split()) for p in paragraphs), default=0)
            return int(max_para_words * self.chunk_overlap)
        return int(self.chunk_overlap)

    def _build_chunks(self, paragraphs: List[str], overlap: int) -> List[str]:
        """
        Build paragraph-based chunks, applying word overlap from the previous chunk
        when requested.

        Warnings:
            SplitterOutputWarning: If splitter produces empty chunks.
            ChunkUnderflowWarning: If fewer chunks than ``chunk_size`` are produced
                because the input has too few paragraphs.
        """
        chunks: List[str] = []
        num_paragraphs = len(paragraphs)
        start = 0

        while start < num_paragraphs:
            end = min(start + self.chunk_size, num_paragraphs)
            chunk_paragraphs = paragraphs[start:end]
            chunk_text = self.line_break[0].join(chunk_paragraphs)

            if overlap > 0 and chunks:
                prev_words = chunks[-1].split()
                overlap_words = (
                    prev_words[-overlap:] if overlap <= len(prev_words) else prev_words
                )
                chunk_text = (
                    self.line_break[0]
                    .join([" ".join(overlap_words), chunk_text])
                    .strip()
                )

            chunks.append(chunk_text)
            start += self.chunk_size

        if not chunks:
            chunks = [""]
            warnings.warn(
                "ParagraphSplitter did not find any non-empty paragraphs; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=3,
            )
            return chunks

        if len(chunks) < self.chunk_size:
            warnings.warn(
                f"ParagraphSplitter produced fewer chunks ({len(chunks)}) than "
                f"the configured chunk_size ({self.chunk_size}) because the input "
                f"contains only {num_paragraphs} paragraph(s).",
                ChunkUnderflowWarning,
                stacklevel=3,
            )

        return chunks

    def _build_output(
        self, reader_output: ReaderOutput, chunks: List[str]
    ) -> SplitterOutput:
        """
        Assemble and return the final SplitterOutput.

        Raises:
            InvalidChunkException: If #chunk_ids != #chunks.
            SplitterOutputException: If SplitterOutput construction fails.
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
                split_method="paragraph_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "line_break": self.line_break,
                },
                metadata=metadata,
            )
        except Exception as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in ParagraphSplitter: {exc}"
            ) from exc
