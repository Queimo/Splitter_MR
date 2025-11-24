import warnings
from typing import Union

from ...schema import ReaderOutput, SplitterOutput
from ...schema.exceptions import InvalidChunkException, SplitterConfigException
from ...schema.warnings import ChunkUnderflowWarning, SplitterInputWarning
from ..base_splitter import BaseSplitter


class WordSplitter(BaseSplitter):
    """Split text into overlapping or non-overlapping word-based chunks.

    This splitter is configurable with a maximum chunk size (``chunk_size`` in
    words) and an overlap between consecutive chunks (``chunk_overlap``). The
    overlap can be specified either as an integer (number of words) or as a
    float between 0 and 1 (fraction of chunk size). It is useful for NLP tasks
    where word-based boundaries are important for context preservation.

    Args:
        chunk_size: Maximum number of words per chunk. Must be a positive
            integer.
        chunk_overlap: Number or percentage of overlapping words between
            chunks. If a float is provided, it must satisfy
            ``0 <= chunk_overlap < 1``.

    Raises:
        SplitterConfigException: If ``chunk_size`` is not positive, if
            ``chunk_overlap`` is invalid (negative, too large, or wrong type),
            or if it is greater than or equal to ``chunk_size``.
    """

    def __init__(self, chunk_size: int = 5, chunk_overlap: Union[int, float] = 0):
        if chunk_size <= 0:
            raise SplitterConfigException(
                f"chunk_size must be a positive integer, got {chunk_size!r}."
            )

        if not isinstance(chunk_overlap, (int, float)):
            raise SplitterConfigException(
                "chunk_overlap must be an int or float, "
                f"got {type(chunk_overlap).__name__!r}."
            )

        # Validate float overlap range (as fraction)
        if isinstance(chunk_overlap, float) and not (0 <= chunk_overlap < 1):
            raise SplitterConfigException(
                "When chunk_overlap is a float, it must be between 0 and 1."
            )

        # Validate integer overlap is not negative
        if isinstance(chunk_overlap, int) and chunk_overlap < 0:
            raise SplitterConfigException(
                "chunk_overlap cannot be negative when provided as an integer."
            )

        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap

    def _compute_overlap(self) -> int:
        """Compute overlap in words from ``self.chunk_overlap`` and ``chunk_size``.

        Returns:
            The overlap as a non-negative integer number of words.

        Raises:
            SplitterConfigException: If the resulting overlap is invalid or
                greater than or equal to ``chunk_size``.
        """
        chunk_size = self.chunk_size

        if isinstance(self.chunk_overlap, float):
            # At this point we already know 0 <= chunk_overlap < 1
            overlap = int(chunk_size * self.chunk_overlap)
        else:
            overlap = int(self.chunk_overlap)

        if overlap < 0:
            # Defensive; should be caught earlier, but keep for safety
            raise SplitterConfigException("chunk_overlap cannot be negative.")

        if overlap >= chunk_size:
            raise SplitterConfigException(
                "chunk_overlap must be smaller than chunk_size."
            )

        return overlap

    # ---- Main logic ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Split the input text into word-based chunks.

        The splitter uses simple whitespace tokenization and supports either
        integer or fractional overlap between consecutive chunks.

        Args:
            reader_output: Input text and associated metadata.

        Returns:
            A ``SplitterOutput`` instance containing:

            * ``chunks``: List of word-based chunks.
            * ``chunk_id``: Corresponding unique identifiers for each chunk.
            * Document metadata and splitter configuration parameters.

        Raises:
            SplitterConfigException:
                If the configuration is invalid (for example, overlap is too
                large).
            InvalidChunkException:
                If the internal chunk ID generation does not match the number
                of produced chunks.

        Warns:
            SplitterInputWarning:
                If the input text is empty or whitespace-only.
            ChunkUnderflowWarning:
                If no chunks are produced (for example, due to empty input or
                aggressive filtering).

        Example:
            ```python
            from splitter_mr.splitter import WordSplitter

            reader_output = ReaderOutput(
                text: "My Wonderful Family\\nI live in a house near the mountains.I have two brothers and one sister, and I was born last...",
                document_name: "my_wonderful_family.txt",
                document_path: "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt",
            )

            # Split into chunks of 5 words, overlapping by 2 words
            splitter = WordSplitter(chunk_size=5, chunk_overlap=2)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['My Wonderful Family\\nI live','I live in a house near','house near the mountains.I', ...]
            ```
        """
        # Initialize variables
        text = reader_output.text or ""
        chunk_size = self.chunk_size

        if not text.strip():
            warnings.warn(
                "WordSplitter received empty or whitespace-only text; "
                "no chunks will be produced.",
                SplitterInputWarning,
            )

        # Split text into words (using simple whitespace tokenization)
        words = text.split()
        total_words = len(words)

        # Determine overlap in words
        overlap = self._compute_overlap()
        step = chunk_size - overlap
        if step <= 0:
            raise SplitterConfigException(
                "Invalid step size computed for WordSplitter; "
                "check chunk_size and chunk_overlap configuration."
            )

        # Split into chunks
        chunks: list[str] = []
        start = 0
        while start < total_words:
            end = start + chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            start += step

        if not chunks:
            warnings.warn(
                "WordSplitter produced no chunks for the given input.",
                ChunkUnderflowWarning,
            )

        # Generate chunk_id and append metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        if len(chunk_ids) != len(chunks):
            raise InvalidChunkException(
                "Chunk ID generation mismatch: number of chunk_ids does not "
                "match number of chunks."
            )

        metadata = self._default_metadata()

        # Return output (SplitterOutput may still validate and reject empty chunks)
        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="word_splitter",
            split_params={
                "chunk_size": chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            metadata=metadata,
        )
        return output
