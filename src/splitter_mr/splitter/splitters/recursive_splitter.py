import warnings
from typing import List, Tuple, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
from ...schema.constants import DEFAULT_RECURSIVE_SEPARATORS
from ..base_splitter import BaseSplitter


class RecursiveCharacterSplitter(BaseSplitter):
    """
    RecursiveCharacterSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk is created by repeatedly breaking down the text until it reaches the
    desired chunk size. This splitter is backed by LangChain's
    :class:`RecursiveCharacterTextSplitter`.

    Args:
        chunk_size (int): the number of characters per chunks (approximately).
        chunk_overlap (int | float): the number of characters which matches between
            contiguous chunks, or a fraction of chunk_size when 0 <= value < 1.
        separators (str | List[str]): the list of characters or regex patterns which
            defines how text is split.

    Raises:
        SplitterConfigException:
            If ``chunk_size`` is less than 1, ``chunk_overlap`` is negative or
            effectively greater than or equal to ``chunk_size``, or ``separators`` is
            neither a non-empty string nor a sequence of strings with at least one
            non-empty entry.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: Union[int, float] = 0.1,
        separators: Union[str, List[str], Tuple[str]] = DEFAULT_RECURSIVE_SEPARATORS,
    ):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise SplitterConfigException(
                "chunk_size must be a positive integer greater than or equal to 1"
            )

        if not isinstance(chunk_overlap, (int, float)) or chunk_overlap < 0:
            raise SplitterConfigException(
                "chunk_overlap must be a non-negative int or float"
            )

        if isinstance(separators, str):
            separators_list = [separators]
        elif isinstance(separators, (list, tuple)):
            separators_list = list(separators)
        else:
            raise SplitterConfigException(
                "separators must be a string or a list of strings"
            )

        if not separators_list or any(not isinstance(s, str) for s in separators_list):
            raise SplitterConfigException("separators must contain only string values")

        is_default_separators = (
            isinstance(separators, tuple) and separators == DEFAULT_RECURSIVE_SEPARATORS
        )

        if not is_default_separators and any(s == "" for s in separators_list):
            raise SplitterConfigException(
                "separators must contain at least one non-empty string"
            )

        if isinstance(chunk_overlap, float):
            eff_overlap = int(chunk_size * chunk_overlap)
        else:
            eff_overlap = int(chunk_overlap)

        if eff_overlap >= chunk_size:
            raise SplitterConfigException(
                "chunk_overlap (effective characters) must be smaller than chunk_size"
            )

        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.separators = separators_list

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text into character-based chunks using a recursive splitting strategy
        (via LangChain's :class:`RecursiveCharacterTextSplitter`), supporting configurable
        separators, chunk size, and overlap.

        Args:
            reader_output (ReaderOutput): Dataclass containing at least a ``text`` field (str
                or None) and optional document metadata (e.g., ``document_name``,
                ``document_path``, etc.).

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ReaderOutputException:
                If ``reader_output.text`` is missing or not ``str``/``None``.
            SplitterConfigException:
                If (effective) ``chunk_overlap`` is greater than or equal to ``chunk_size``.
            InvalidChunkException:
                If the number of generated chunk IDs does not match the number of chunks.
            SplitterOutputException:
                If constructing :class:`SplitterOutput` fails unexpectedly, or if
                LangChain's splitter raises an unexpected error.

        Warnings:
            SplitterInputWarning:
                When the input text is empty or whitespace-only.
            SplitterOutputWarning:
                When no chunks are produced and the splitter falls back to a single
                empty chunk.


        Example:
            **Basic usage** with a simple text string:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter import RecursiveCharacterSplitter

            # Sample text (short for demonstration)
            text = (
                "LangChain makes it easy to build LLM-powered applications. "
                "Recursive splitting helps maintain semantic coherence while "
                "still enforcing chunk-size limits."
            )

            reader_output = ReaderOutput(
                text=text,
                document_name="example.txt",
                document_path="/tmp/example.txt",
                document_id="abc123",
                conversion_method="text",
                metadata={}
            )

            splitter = RecursiveCharacterSplitter(
                chunk_size=50,
                chunk_overlap=0.2,  # 20% of chunk_size overlap
                separators=["\\n\\n", ".", " "]  # recursive fallback separators
            )

            output = splitter.split(reader_output)

            # Inspect results
            print(output.chunks)
            print(output.chunk_id)
            print(output.split_params)
            ```
        """
        # Validate input
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
                "RecursiveCharacterSplitter received empty or whitespace-only text; "
                "resulting chunks will be empty.",
                SplitterInputWarning,
                stacklevel=2,
            )

        chunk_size = self.chunk_size

        # Determine overlap in characters (effective value used by LangChain)
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            overlap = int(chunk_size * self.chunk_overlap)
        else:
            overlap = int(self.chunk_overlap)

        if overlap >= chunk_size:
            # Config is invalid relative to this chunk_size
            raise SplitterConfigException(
                "chunk_overlap (effective characters) must be smaller than chunk_size"
            )

        # Generate chunks
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=self.separators,
            )
            texts = splitter.create_documents([text])
        except Exception as exc:
            # Wrap any unexpected LangChain behaviour
            raise SplitterOutputException(
                f"RecursiveCharacterTextSplitter failed during split: {exc}"
            ) from exc

        chunks = [doc.page_content for doc in texts] if texts else []

        # -> If no chunks at all, warn and fall back
        if not chunks:
            warnings.warn(
                "RecursiveCharacterSplitter did not produce any chunks; "
                "returning a single empty chunk.",
                SplitterOutputWarning,
                stacklevel=2,
            )
            chunks = [""]

        # Generate chunk_ids and append metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        if len(chunk_ids) != len(chunks):
            raise InvalidChunkException(
                "Number of chunk IDs does not match number of chunks "
                f"(chunk_ids={len(chunk_ids)}, chunks={len(chunks)})."
            )

        metadata = self._default_metadata()

        # Build output
        try:
            output = SplitterOutput(
                chunks=chunks,
                chunk_id=chunk_ids,
                document_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="recursive_character_splitter",
                split_params={
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "separators": self.separators,
                },
                metadata=metadata,
            )
        except Exception as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in RecursiveCharacterSplitter: {exc}"
            ) from exc

        return output
