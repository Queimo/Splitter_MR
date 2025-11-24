import json

from langchain_text_splitters.json import RecursiveJsonSplitter

from ...schema import (
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterOutput,
    SplitterOutputException,
)
from ..base_splitter import BaseSplitter


class RecursiveJSONSplitter(BaseSplitter):
    """
    Split a JSON string or structure into overlapping or non-overlapping chunks,
    using the Langchain RecursiveJsonSplitter. This splitter is designed to recursively
    break down JSON data (including nested objects and arrays) into manageable pieces based
    on keys, arrays, or other separators, until the desired chunk size is reached.

    Args:
        chunk_size (int): Maximum chunk size, measured in the number of characters per chunk.
        min_chunk_size (int): Minimum chunk size, in characters.

    Raises:
        SplitterConfigException: if parameters are not provided with the expected type.

    Notes:
        See [Langchain Docs on RecursiveJsonSplitter](https://python.langchain.com/api_reference/text_splitters/json/langchain_text_splitters.json.RecursiveJsonSplitter.html#langchain_text_splitters.json.RecursiveJsonSplitter).
    """

    def __init__(self, chunk_size: int = 1000, min_chunk_size: int = 200):
        super().__init__(chunk_size)

        if not isinstance(chunk_size, int):
            raise SplitterConfigException(
                "Parameter `chunk_size` must be an integer number"
            )
        if not isinstance(min_chunk_size, int):
            raise SplitterConfigException(
                "Parameter `min_chunk_size` must be an integer number"
            )

        self.min_chunk_size = min_chunk_size

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input JSON text from the reader_output dictionary into recursively chunked pieces,
        allowing for overlap by number or percentage of characters.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary containing at least a 'text' key (str) and optional document metadata
                (e.g., 'document_name', 'document_path', etc.).

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If the 'text' field is missing from reader_output.
            json.JSONDecodeError: If the 'text' field contains invalid JSON.

        Example:
            ```python
            from splitter_mr.splitter import RecursiveJSONSplitter

            # This dictionary has been obtained from `VanillaReader`
            reader_output = ReaderOutput(
                text: '{"company": {"name": "TechCorp", "employees": [{"name": "Alice"}, {"name": "Bob"}]}}'
                document_name: "company_data.json",
                document_path: "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/company_data.json",
                document_id: "doc123",
                conversion_method: "vanilla",
                ocr_method: None
            )
            splitter = RecursiveJSONSplitter(chunk_size=100, min_chunk_size=20)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['{"company": {"name": "TechCorp"}}', '{"employees": [{"name": "Alice"}, {"name": "Bob"}]}']
            ```

        Raises:
            ReaderOutputException: if input does not contain a valid JSON.
            InvalidChunkException: if returned chunks are not in a valid format.
            SplitterOutputException: if response has not been generated as expected
        """
        # Initialize variables
        try:
            text = json.loads(reader_output.text)
        except json.JSONDecodeError as e:
            raise ReaderOutputException(f"Input does not contain a valid JSON: {e}")

        # Split text into smaller JSON chunks
        try:
            splitter = RecursiveJsonSplitter(
                max_chunk_size=self.chunk_size,
                min_chunk_size=int(self.chunk_size - self.min_chunk_size),
            )
            chunks = splitter.split_text(json_data=text, convert_lists=True)
        except Exception as e:
            raise InvalidChunkException(
                f"There was an error trying to split the JSON text: {e}"
            )

        if chunks is None or chunks == []:
            raise InvalidChunkException("Splitter has produced void or missing chunks")

        # Generate chunk_ids and metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        # Return output
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
                split_method="recursive_json_splitter",
                split_params={
                    "max_chunk_size": self.chunk_size,
                    "min_chunk_size": self.min_chunk_size,
                },
                metadata=metadata,
            )
        except Exception as exc:
            raise SplitterOutputException(
                f"There was an error trying to build SplitterOutput response: {exc}"
            )
        return output
