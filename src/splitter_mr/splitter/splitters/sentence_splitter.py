import re
import warnings
from typing import List, Union

from ...schema import (
    DEFAULT_SENTENCE_SEPARATORS,
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


class SentenceSplitter(BaseSplitter):
    """
    SentenceSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk contains a specified number of sentences, and overlap is defined
    by a number or percentage of words from the end of the previous chunk.

    Args:
        chunk_size (int): Maximum number of sentences per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping words
            between chunks. If a float in [0, 1), it is treated as a fraction of the
            maximum sentence length (in words); otherwise it is interpreted as an
            absolute number of words.
        separators (Union[str, List[str]]): Sentence boundary separators. If a list,
            it is normalised into a regex pattern (legacy path). If a string, it is
            treated as a full regex pattern.

    Raises:
        SplitterConfigException:
            If ``chunk_size`` is less than 1 or not an int, ``chunk_overlap`` is
            negative or not numeric, or ``separators`` is neither a non-empty string
            nor a list of non-empty strings.
    """

    def __init__(
        self,
        chunk_size: int = 5,
        chunk_overlap: Union[int, float] = 0,
        separators: Union[str, List[str]] = DEFAULT_SENTENCE_SEPARATORS,
    ):
        # ---- Config validation ---- #
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise SplitterConfigException(
                "chunk_size must be a positive integer greater than or equal to 1"
            )

        if not isinstance(chunk_overlap, (int, float)) or chunk_overlap < 0:
            raise SplitterConfigException(
                "chunk_overlap must be a non-negative int or float"
            )

        # Normalise and validate separators
        if isinstance(separators, list):
            if not separators or any(
                not isinstance(s, str) or not s for s in separators
            ):
                raise SplitterConfigException(
                    "separators list must contain at least one non-empty string"
                )
            # Legacy path (NOT recommended): join list with alternation, ensure "..." before "."
            parts = sorted({*separators}, key=lambda s: (s != "...", s))
            sep_pattern = "|".join(re.escape(s) for s in parts)
            # Attach trailing quotes/brackets if user insisted on a list
            separators_pattern = rf'(?:{sep_pattern})(?:["”’\'\)\]\}}»]*)\s*'
        elif isinstance(separators, str):
            if not separators:
                raise SplitterConfigException(
                    "separators must be a non-empty regex/string or a list of strings"
                )
            # Recommended path: already a full regex pattern
            separators_pattern = separators
        else:
            raise SplitterConfigException(
                "separators must be a string or a list of strings"
            )

        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.separators = separators_pattern
        self._sep_re = re.compile(f"({self.separators})")

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text from ``reader_output`` into sentence-based chunks,
        allowing for overlap at the word level.

        Pipeline:

        1. Validate and normalise ``reader_output.text``.
        2. Split into sentences.
        3. Compute word overlap.
        4. Build chunks (with overlap).
        5. Build the final :class:`SplitterOutput`.

        Args:
            reader_output (ReaderOutput): Dataclass containing at least a ``text``
                attribute (str or None) and optional document metadata.

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
                When no non-empty sentences are found, causing the splitter to fall
                back to a single empty chunk.
            ChunkUnderflowWarning:
                When fewer chunks than ``chunk_size`` are produced because the input
                has too few sentences.

        Example:
            ```python
            from splitter_mr.splitter import SentenceSplitter

            reader_output = ReaderOutput(
                text: "My Wonderful Family\\nI live in a house near the mountains.I have two brothers and one sister, and I was born last...",
                document_name: "my_wonderful_family.txt",
                document_path: "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt",
            )

            # Split into chunks of 2 sentences, no overlapping
            splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['My Wonderful Family. I live in a house near the mountains.', 'I have two brothers and one sister, and I was born last...', ...]
            ```
        """
        text = self._validate_reader_output(reader_output)
        sentences = self._split_into_sentences(text)
        overlap = self._compute_overlap(sentences)
        chunks = self._build_chunks(sentences, overlap)
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
                "SentenceSplitter received empty or whitespace-only text; "
                "resulting chunks will be empty.",
                SplitterInputWarning,
                stacklevel=3,
            )

        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split the input text into normalised sentences.

        Warnings:
            SplitterOutputWarning:
                When no non-empty sentences are found and the splitter falls back
                to a single empty sentence.
        """
        if not text.strip():
            # Already warned in _validate_reader_output; just normalise.
            sentences = []
        else:
            parts = self._sep_re.split(text)  # [text, sep, text, sep, ...]
            sentences: List[str] = []
            i = 0
            while i < len(parts):
                segment = (parts[i] or "").strip()
                if i + 1 < len(parts):
                    # we have a separator that belongs to this sentence
                    sep = parts[i + 1] or ""
                    sentence = (segment + sep).strip()
                    if sentence:
                        sentences.append(sentence)
                    i += 2
                else:
                    # tail without terminator
                    if segment:
                        sentences.append(segment)
                    i += 1

        # Fallback when no sentences were found
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            warnings.warn(
                "SentenceSplitter did not find any non-empty sentences; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=3,
            )
            sentences = [""]

        return sentences

    def _compute_overlap(self, sentences: List[str]) -> int:
        """
        Compute the number of overlapping words between chunks based on the
        configured ``chunk_overlap``.
        """
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            max_sent_words = max((len(s.split()) for s in sentences), default=0)
            return int(max_sent_words * self.chunk_overlap)
        return int(self.chunk_overlap)

    def _build_chunks(self, sentences: List[str], overlap: int) -> List[str]:
        """
        Build sentence-based chunks, applying word overlap from the previous chunk
        when requested.

        Warnings:
            SplitterOutputWarning: If splitter produces empty chunks.
            ChunkUnderflowWarning: If fewer chunks than ``chunk_size`` are produced
                because the input has too few sentences.
        """
        chunks: List[str] = []
        num_sentences = len(sentences)
        start = 0

        while start < num_sentences:
            end = min(start + self.chunk_size, num_sentences)
            chunk_sents = sentences[start:end]
            chunk_text = " ".join(chunk_sents)

            if overlap > 0 and chunks:
                prev_words = chunks[-1].split()
                overlap_words = (
                    prev_words[-overlap:] if overlap <= len(prev_words) else prev_words
                )
                chunk_text = " ".join([" ".join(overlap_words), chunk_text]).strip()

            chunks.append(chunk_text)
            start += self.chunk_size

        if not chunks:
            chunks = [""]
            warnings.warn(
                "SentenceSplitter did not produce any chunks; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=3,
            )
            return chunks

        if len(chunks) < self.chunk_size:
            warnings.warn(
                f"SentenceSplitter produced fewer chunks ({len(chunks)}) than "
                f"the configured chunk_size ({self.chunk_size}) because the input "
                f"contains only {num_sentences} sentence(s).",
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
                split_method="sentence_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "separators": self.separators,
                },
                metadata=metadata,
            )
        except Exception as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in SentenceSplitter: {exc}"
            ) from exc
