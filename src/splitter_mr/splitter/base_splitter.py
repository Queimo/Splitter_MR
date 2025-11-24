import uuid
from abc import ABC, abstractmethod
from typing import List

from ..schema import ReaderOutput, SplitterOutput


class BaseSplitter(ABC):
    """
    Abstract base class for all splitter implementations.

    This class defines the common interface and utility methods for splitters that
    divide text or data into smaller chunks, typically for downstream natural language
    processing tasks or information retrieval. Subclasses should implement the `split`
    method, which takes a :class:`ReaderOutput` and returns a :class:`SplitterOutput`
    containing the resulting chunks and associated metadata.

    Attributes:
        chunk_size (int):
            The maximum number of units (characters, sentences, rows, etc.)
            that a derived splitter is allowed to place in a chunk (semantic
            meaning depends on the subclass).

    Methods:
        split(reader_output):
            Abstract method. Must be implemented by subclasses to perform the
            actual domain-specific splitting logic.

        _generate_chunk_ids(num_chunks):
            Utility to generate a list of unique UUID4-based identifiers for
            chunk tracking.

        _default_metadata():
            Returns a default (empty) metadata dictionary. Subclasses may
            override or extend this to attach extra information to the final
            :class:`SplitterOutput`.

    Example:
        **Creating a simple custom splitter** that breaks text every ``N`` characters:

        ```python
        from splitter_mr.schema import ReaderOutput, SplitterOutput
        from splitter_mr.splitter.base_splitter import BaseSplitter

        class FixedCharSplitter(BaseSplitter):
            def split(self, reader_output: ReaderOutput) -> SplitterOutput:
                text = reader_output.text or ""
                chunks = [
                    text[i : i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)
                ]

                chunk_ids = self._generate_chunk_ids(len(chunks))

                return SplitterOutput(
                    chunks=chunks,
                    chunk_id=chunk_ids,
                    document_name=reader_output.document_name,
                    document_path=reader_output.document_path,
                    document_id=reader_output.document_id,
                    conversion_method=reader_output.conversion_method,
                    reader_method=reader_output.reader_method,
                    ocr_method=reader_output.ocr_method,
                    split_method="fixed_char_splitter",
                    split_params={"chunk_size": self.chunk_size},
                    metadata=self._default_metadata(),
                )

        ro = ReaderOutput(text="abcdefghijklmnopqrstuvwxyz")
        splitter = FixedCharSplitter(chunk_size=5)
        out = splitter.split(ro)

        print(out.chunks)
        ```
        ```python
        ['abcde', 'fghij', 'klmno', 'pqrst', 'uvwxy', 'z']
        ```
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initializer method for BaseSplitter classes
        """
        self.chunk_size = chunk_size

    @abstractmethod
    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Abstract method to split input data into chunks.

        Args:
            reader_output (ReaderOutput): Input data, typically from a document reader,
                including the text to split and any relevant metadata.

        Returns:
            SplitterOutput: A dictionary containing split chunks and associated metadata.
        """

    def _generate_chunk_ids(self, num_chunks: int) -> List[str]:
        """
        Generate a list of unique chunk identifiers.

        Args:
            num_chunks (int): Number of chunk IDs to generate.

        Returns:
            List[str]: List of unique string IDs (UUID4).
        """
        return [str(uuid.uuid4()) for _ in range(num_chunks)]

    def _default_metadata(self) -> dict:
        """
        Return a default metadata dictionary.

        Returns:
            dict: An empty dictionary; subclasses may override to provide additional metadata.
        """
        return {}
