import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import nltk
import spacy
import tiktoken
from langchain_text_splitters import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)

from ...schema import ReaderOutput, SplitterOutput
from ...schema.constants import (
    DEFAULT_NLTK,
    DEFAULT_TOKEN_LANGUAGE,
    DEFAULT_TOKENIZER,
    SPACY_DEFAULTS,
    SUPPORTED_TOKENIZERS,
    TIKTOKEN_DEFAULTS,
)
from ...schema.exceptions import InvalidChunkException, SplitterConfigException
from ...schema.warnings import ChunkUnderflowWarning, SplitterInputWarning
from ..base_splitter import BaseSplitter


class TokenSplitter(BaseSplitter):
    """Split text into token-based chunks using multiple tokenizer backends.

    TokenSplitter splits a given text into chunks based on token counts
    derived from different tokenization models or libraries.

    This splitter supports tokenization via `tiktoken` (OpenAI tokenizer),
    `spacy` (spaCy tokenizer), and `nltk` (NLTK tokenizer). It allows splitting
    text into chunks of a maximum number of tokens (`chunk_size`), using the
    specified tokenizer model.

    Args:
        chunk_size: Maximum number of tokens per chunk.
        model_name: Tokenizer and model in the format ``tokenizer/model``.
            Supported tokenizers include:

            * ``tiktoken/cl100k_base`` (OpenAI tokenizer via tiktoken)
            * ``spacy/en_core_web_sm`` (spaCy English model)
            * ``nltk/punkt_tab`` (NLTK Punkt tokenizer variant)
        language: Language code for the NLTK tokenizer (for example, ``"english"``).

    Raises:
        SplitterConfigException: If ``chunk_size`` is not a positive integer.

    Notes:
        See the LangChain documentation for more details about splitting
        by tokens:
        https://python.langchain.com/docs/how_to/split_by_token/
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        model_name: str = DEFAULT_TOKENIZER,
        language: str = DEFAULT_TOKEN_LANGUAGE,
    ):
        if chunk_size <= 0:
            raise SplitterConfigException(
                f"chunk_size must be a positive integer, got {chunk_size!r}."
            )

        super().__init__(chunk_size)
        self.model_name = model_name or DEFAULT_TOKENIZER
        self.language = language or DEFAULT_TOKEN_LANGUAGE

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Split the input text into token-based chunks.

        The splitter uses the backend specified by ``model_name`` and
        delegates to a tokenizer-specific implementation:

        * tiktoken: Uses OpenAI encodings via
          ``RecursiveCharacterTextSplitter``.
        * spaCy: Uses the specified pipeline via ``SpacyTextSplitter``.
        * NLTK: Uses the Punkt sentence tokenizer via ``NLTKTextSplitter``.

        Models or language data are downloaded automatically if missing.

        Args:
            reader_output: Input text and associated metadata to be split.

        Returns:
            A ``SplitterOutput`` instance containing:

            * ``chunks``: List of token-based text chunks.
            * ``chunk_id``: Corresponding unique identifiers for each chunk.
            * Document metadata and splitter configuration parameters.

        Raises:
            SplitterConfigException:
                If ``model_name`` is malformed, the tokenizer backend is
                unsupported, or the requested model or language resources are
                unavailable.
            InvalidChunkException:
                If the underlying splitter returns an invalid chunks structure.

        Warns:
            SplitterInputWarning:
                If the input text is empty or whitespace-only.
            ChunkUnderflowWarning:
                If no chunks are produced from a non-empty input.

        Example:
            Basic usage with **tiktoken**:

            ```python
            from splitter_mr.splitter import TokenSplitter
            from splitter_mr.schema.models import ReaderOutput

            text = (
                "This is a demonstration of the TokenSplitter. "
                "It splits text into chunks based on token counts."
            )

            ro = ReaderOutput(text=text, document_name="demo.txt")
            splitter = TokenSplitter(
                chunk_size=20,
                model_name="tiktoken/cl100k_base",
            )
            output = splitter.split(ro)
            print(output.chunks)
            ```

            Using **spaCy**:

            ```python
            splitter = TokenSplitter(
                chunk_size=50,
                model_name="spacy/en_core_web_sm",
            )
            output = splitter.split(ro)
            print(output.chunks)
            ```

            Using **NLTK**:

            ```python
            splitter = TokenSplitter(
                chunk_size=40,
                model_name="nltk/punkt_tab",
                language="english",
            )
            output = splitter.split(ro)
            print(output.chunks)
            ```
        """
        text = reader_output.text or ""

        if not text.strip():
            warnings.warn(
                "TokenSplitter received empty or whitespace-only text; "
                "no chunks will be produced.",
                SplitterInputWarning,
            )

        tokenizer, model = self._parse_model()
        factory = self._get_splitter_factory(tokenizer)
        splitter = factory(model)

        chunks = splitter.split_text(text)

        if chunks is None:
            raise InvalidChunkException(
                "The underlying text splitter returned None instead of a list of chunks."
            )

        if not chunks:
            warnings.warn(
                "TokenSplitter produced no chunks for the given input.",
                ChunkUnderflowWarning,
            )

        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="token_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "model_name": self.model_name,
                "language": self.language,
            },
            metadata=metadata,
        )

    # ---- Internal helpers ---- #

    @staticmethod
    def list_nltk_punkt_languages() -> List[str]:
        """Return a sorted list of available NLTK Punkt models.

        Returns:
            model_list(List[str]): A sorted list of language codes corresponding to available
            Punkt sentence tokenizer models in the local NLTK data path.
        """
        models = set()
        for base in map(Path, nltk.data.path):
            punkt_dir = base / "tokenizers" / "punkt"
            if punkt_dir.exists():
                models.update(f.stem for f in punkt_dir.glob("*.pickle"))
        model_list = sorted(models)
        return model_list

    def _parse_model(self) -> Tuple[str, str]:
        """Parse and validate the ``tokenizer/model`` string.

        Returns:
            A tuple ``(tokenizer, model)`` where ``tokenizer`` is the backend
            name (for example, ``"tiktoken"``, ``"spacy"``, or ``"nltk"``) and
            ``model`` is the corresponding model identifier.

        Raises:
            SplitterConfigException: If ``model_name`` is not in the
                ``tokenizer/model`` format.
        """
        if "/" not in self.model_name:
            raise SplitterConfigException(
                "model_name must be in the format 'tokenizer/model', "
                f"e.g. '{DEFAULT_TOKENIZER}'. Got: {self.model_name!r}"
            )
        tokenizer, model = self.model_name.split("/", 1)
        return tokenizer, model

    # ---- Tokenizer builders ---- #

    def _build_tiktoken_splitter(self, model: str) -> RecursiveCharacterTextSplitter:
        """Build a tiktoken-based text splitter.

        Args:
            model: The tiktoken encoding name (for example, ``"cl100k_base"``).

        Returns:
            A configured ``RecursiveCharacterTextSplitter`` that uses the
            specified tiktoken encoding.

        Raises:
            SplitterConfigException:
                If tiktoken encodings cannot be listed or if the requested
                encoding is not available.
        """
        try:
            available_models = tiktoken.list_encoding_names()
        except Exception as exc:  # defensive, backend failure
            raise SplitterConfigException(
                "Failed to list tiktoken encodings. "
                "Please ensure tiktoken is correctly installed."
            ) from exc

        if model not in available_models:
            raise SplitterConfigException(
                f"tiktoken encoding {model!r} is not available. "
                f"Available defaults include: {TIKTOKEN_DEFAULTS}. "
                f"Full list from tiktoken: {available_models}"
            )

        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=model,
            chunk_size=self.chunk_size,
            chunk_overlap=0,
        )

    def _build_spacy_splitter(self, model: str) -> SpacyTextSplitter:
        """Build a spaCy-based text splitter.

        Args:
            model: The spaCy pipeline name (for example, ``"en_core_web_sm"``).

        Returns:
            A configured ``SpacyTextSplitter`` that uses the specified spaCy
            model.

        Raises:
            SplitterConfigException:
                If the spaCy model cannot be downloaded or loaded.

        Warns:
            SplitterInputWarning: If ``chunk_size`` is so large that spaCy
                may require excessive memory.
        """
        if not spacy.util.is_package(model):
            try:
                spacy.cli.download(model)
            except Exception as exc:
                raise SplitterConfigException(
                    f"spaCy model {model!r} is not available for download. "
                    f"Common models include: {SPACY_DEFAULTS}"
                ) from exc

        try:
            spacy.load(model)
        except Exception as exc:
            raise SplitterConfigException(
                f"spaCy model {model!r} could not be loaded. "
                "Please verify that the installation is not corrupted."
            ) from exc

        MAX_SAFE_LENGTH = 1_000_000
        if self.chunk_size > MAX_SAFE_LENGTH:
            warnings.warn(
                "Configured chunk_size is very large; spaCy v2.x parser and NER "
                "models may require ~1GB of temporary memory per 100,000 characters.",
                SplitterInputWarning,
            )

        return SpacyTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            max_length=MAX_SAFE_LENGTH,
            pipeline=model,
        )

    def _build_nltk_splitter(self, _model: str) -> NLTKTextSplitter:
        """Build an NLTK-based text splitter.

        The ``_model`` argument is currently unused because the NLTK backend
        is controlled by ``language`` rather than by an explicit model name.
        It is kept for uniformity with other builder methods.

        Args:
            _model: Unused placeholder for tokenizer-specific model ID.

        Returns:
            A configured ``NLTKTextSplitter`` that uses the configured
            ``language`` for sentence tokenization.

        Raises:
            SplitterConfigException:
                If NLTK Punkt data cannot be found or downloaded.
        """
        punkt_relpath = Path("tokenizers") / "punkt" / f"{self.language}.pickle"
        try:
            nltk.data.find(str(punkt_relpath))
        except LookupError:
            try:
                nltk.download(DEFAULT_NLTK[0])
            except Exception as exc:
                raise SplitterConfigException(
                    "NLTK Punkt data could not be downloaded. "
                    f"Tried language {self.language!r} and default resource "
                    f"{DEFAULT_NLTK[0]!r}."
                ) from exc

        return NLTKTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            language=self.language,
        )

    def _get_splitter_factory(self, tokenizer: str) -> Callable[[str], object]:
        """Return the factory function for the given tokenizer backend.

        Args:
            tokenizer: The tokenizer backend name, such as ``"tiktoken"``,
                ``"spacy"``, or ``"nltk"``.

        Returns:
            A callable that accepts a model string and returns a configured
            text splitter instance.

        Raises:
            SplitterConfigException: If the tokenizer backend is not supported.
        """
        factories: dict[str, Callable[[str], object]] = {
            "tiktoken": self._build_tiktoken_splitter,
            "spacy": self._build_spacy_splitter,
            "nltk": self._build_nltk_splitter,
        }

        try:
            return factories[tokenizer]
        except KeyError:
            raise SplitterConfigException(
                f"Unsupported tokenizer {tokenizer!r}. "
                f"Supported tokenizers: {SUPPORTED_TOKENIZERS}"
            )
