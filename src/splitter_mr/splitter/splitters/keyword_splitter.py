import re
import uuid
import warnings
from typing import Dict, Iterable, List, Pattern, Tuple, Union

from ...schema import (
    DEFAULT_KEYWORD_DELIMITER_POS,
    SUPPORTED_KEYWORD_DELIMITERS,
    InvalidChunkException,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
    SplitterOutputException,
)
from ..base_splitter import BaseSplitter


class KeywordSplitter(BaseSplitter):
    """
    Splitter that chunks text around *keyword* boundaries using regular expressions.

    This splitter searches the input text for one or more *keyword patterns* (regex)
    and creates chunks at each match boundary. You can control how the matched
    delimiter is attached to the resulting chunks (before/after/both/none) and apply a
    secondary, size-based re-chunking to respect ``chunk_size``.

    Notes:
        - All regexes are compiled into **one** alternation with *named groups* when
          ``patterns`` is a dict. This simplifies per-keyword accounting.
        - If the input text is empty or no matches are found, the entire text
          becomes a single chunk (subject to size-based re-chunking).

    Args:
        patterns (Union[List[str], Dict[str, str]]): A list of regex pattern strings **or** a mapping of
            ``name -> regex pattern``. When a dict is provided, the keys are used in
            the metadata counts. When a list is provided, synthetic names are
            generated (``k0``, ``k1``, ...).
        flags (int): Standard ``re`` flags combined with ``|`` (e.g., ``re.IGNORECASE``).
        include_delimiters (str): Where to attach the matched keyword delimiter.
            One of ``"none"``, ``"before"``, ``"after"``, ``"both"``.
            - ``before`` (default) appends the match to the *preceding* chunk.
            - ``after`` prepends the match to the *following* chunk.
            - ``both`` duplicates the match on both sides.
            - ``none`` omits the delimiter from both sides.
        chunk_size (int): Target maximum size (in characters) for each chunk. When a
            produced chunk exceeds this value, it is *soft*-wrapped by whitespace
            using a greedy strategy.

    Raises:
        SplitterConfigException: If ``patterns``, ``include_delimiters`` or ``chunk_size``
            are invalid or regex compilation fails.
        ReaderOutputException: If ``reader_output`` does not expose a valid ``text`` field.
        InvalidChunkException: If internal chunk accounting becomes inconsistent.
        SplitterOutputException: If building :class:`SplitterOutput` fails unexpectedly.
    """

    def __init__(
        self,
        patterns: Union[List[str], Dict[str, str]],
        *,
        flags: int = 0,
        include_delimiters: str = DEFAULT_KEYWORD_DELIMITER_POS,
        chunk_size: int = 100000,
    ) -> None:
        # Basic config validation at construction time
        if chunk_size <= 0 or not isinstance(chunk_size, int):
            raise SplitterConfigException(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )

        super().__init__(chunk_size=chunk_size)
        self.include_delimiters = self._validate_include_delimiters(include_delimiters)

        # Validate patterns type early for clearer errors
        if not isinstance(patterns, (list, dict)):
            raise SplitterConfigException(
                "patterns must be a list of regex strings or a dict[name -> pattern], "
                f"got {type(patterns).__name__!r}"
            )

        try:
            self.pattern_names, self.compiled = self._compile_patterns(patterns, flags)
        except re.error as exc:  # invalid regex, bad group name, etc.
            raise SplitterConfigException(
                f"Failed to compile keyword patterns: {exc}"
            ) from exc

        self.flags = flags

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split ReaderOutput into keyword-delimited chunks and build structured output.

        The method first splits around regex keyword matches (respecting
        ``include_delimiters``), then performs a secondary size-based soft wrap to
        respect ``chunk_size``. It returns a fully populated :class:`SplitterOutput`.

        Args:
            reader_output (ReaderOutput): Input document and metadata.

        Returns:
            SplitterOutput: Output structure with chunked text and metadata.

        Raises:
            ReaderOutputException: If ``reader_output`` has an invalid structure.
            InvalidChunkException: If the number of chunks and chunk IDs diverge.
            SplitterOutputException: If constructing the output object fails.

        Example:
            **Basic usage** with a **list** of patterns:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters import KeywordSplitter

            text = "Alpha KEY Beta KEY Gamma"
            ro = ReaderOutput(
                text=text,
                document_name="demo.txt",
                document_path="/tmp/demo.txt",
            )

            splitter = KeywordSplitter(patterns=[r"KEY"])
            out = splitter.split(ro)

            print(out.chunks)
            ```

            ```python
            ['Alpha KEY', 'Beta KEY', 'Gamma']
            ```

            Using a **`dict` of named patterns** (names appear in metadata):

            ```python
            patterns = {
                "plus": r"\\+",
                "minus": r"-",
            }
            text = "A + B - C + D"
            ro = ReaderOutput(text=text)

            splitter = KeywordSplitter(patterns=patterns)
            out = splitter.split(ro)

            print(out.chunks)
            ```

            ```python
            ['A +', 'B -', 'C +', 'D']
            ```

            ```python
            print(out.metadata["keyword_matches"]["counts"])
            ```

            ```json
            {'plus': 2, 'minus': 1}
            ```

            Demonstrating ``include_delimiters`` modes:

            ```python
            text = "A#B#C"

            splitter = KeywordSplitter(patterns=[r"#"], include_delimiters="after")
            out = splitter.split(ReaderOutput(text=text))
            print(out.chunks)
            ```

            ```python
            ['A#', 'B#', 'C']
            ```

            ```python
            splitter = KeywordSplitter(patterns=[r"#"], include_delimiters="none")
            out = splitter.split(ReaderOutput(text=text))
            print(out.chunks)
            ```

            ```python
            ['A', 'B', 'C']
            ```

            Example showing **size-based soft wrapping** (`chunk_size=5`):

            ```python
            text = "abcdefghijklmnopqrstuvwxyz"
            splitter = KeywordSplitter(patterns=[r"x"], chunk_size=5)
            ```

            ```python
            out = splitter.split(ReaderOutput(text=text))
            print(out.chunks)
            ```

            ```python
            ['abcde', 'fghij', 'klmno', 'pqrst', 'uvwxy', 'z']
            ```

            Example with **multiple patterns and mixed text**:

            ```python
            splitter = KeywordSplitter(
                patterns=[r"ERROR", r"WARNING"],
                include_delimiters="after",
            )

            log = "INFO Start\\nERROR Failure occurred\\nWARNING Low RAM\\nINFO End"
            out = splitter.split(ReaderOutput(text=log))

            print(out.chunks)
            ```

            ```python
            ['INFO Start\\nERROR', 'Failure occurred\\nWARNING', 'Low RAM\\nINFO End']
            ```
        """
        if not hasattr(reader_output, "text"):
            raise ReaderOutputException(
                "ReaderOutput object must expose a 'text' attribute."
            )

        text = reader_output.text
        if text is None:
            text: str = ""
        elif not isinstance(text, str):
            raise ReaderOutputException(
                f"ReaderOutput.text must be of type 'str' or None, got "
                f"{type(text).__name__!r}"
            )

        # Warn on suspiciously empty input
        if not text.strip():
            warnings.warn(
                "KeywordSplitter received empty or whitespace-only text; "
                "output will contain a single empty chunk.",
                SplitterInputWarning,
                stacklevel=2,
            )

        # Ensure document_id is present so it propagates (fixes metadata test)
        if not getattr(reader_output, "document_id", None):
            reader_output.document_id = str(uuid.uuid4())

        # Primary split by keyword matches (names used for counts)
        raw_chunks, match_spans, match_names = self._split_by_keywords(text)

        # Secondary size-based re-chunking to respect chunk_size
        sized_chunks: list[str] = []
        for ch in raw_chunks:
            sized_chunks.extend(self._soft_wrap(ch, self.chunk_size))
        if not sized_chunks:
            sized_chunks: list[str] = [""]

        # Generate IDs
        chunk_ids = self._generate_chunk_ids(len(sized_chunks))

        # Extra sanity check: chunks vs IDs
        if len(chunk_ids) != len(sized_chunks):
            raise InvalidChunkException(
                "Number of chunk IDs does not match number of chunks "
                f"(chunk_ids={len(chunk_ids)}, chunks={len(sized_chunks)})."
            )

        # Build metadata (ensure counts/spans are always present)
        matches_meta: dict[str, any] = {
            "counts": self._count_by_name(match_names),
            "spans": match_spans,
            "include_delimiters": self.include_delimiters,
            "flags": self.flags,
            "pattern_names": self.pattern_names,
            "chunk_size": self.chunk_size,
        }

        try:
            return self._build_output(
                reader_output=reader_output,
                chunks=sized_chunks,
                chunk_ids=chunk_ids,
                matches_meta=matches_meta,
            )
        except (TypeError, ValueError) as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in KeywordSplitter: {exc}"
            ) from exc

    # ---- Helpers ---- #

    @staticmethod
    def _validate_include_delimiters(value: str) -> str:
        """
        Validate and normalize include_delimiters argument.

        Args:
            value (str): One of {"none", "before", "after", "both"}.

        Returns:
            str: Normalized delimiter mode.

        Raises:
            SplitterConfigException: If the mode is invalid.
        """
        if not isinstance(value, str):
            raise SplitterConfigException(
                f"include_delimiters must be a string, got {type(value).__name__!r}"
            )

        v: str = value.lower().strip()
        if v not in SUPPORTED_KEYWORD_DELIMITERS:
            raise SplitterConfigException(
                "include_delimiters must be one of "
                f"{sorted(SUPPORTED_KEYWORD_DELIMITERS)}, got {value!r}"
            )
        return v

    @staticmethod
    def _compile_patterns(
        patterns: Union[List[str], Dict[str, str]], flags: int
    ) -> Tuple[List[str], Pattern[str]]:
        """
        Compile patterns into a single alternation regex.

        If a dict is given, build a pattern with **named** groups to preserve the
        provided names. If a list is given, synthesize names (k0, k1, ...).

        Args:
            patterns (Union[List[str], Dict[str, str]]): Patterns or mapping.
            flags (int): Regex flags.

        Returns:
            Tuple[List[str], Pattern[str]]: Names and compiled regex.

        Raises:
            SplitterConfigException: If patterns have an unsupported type.
            re.error: If regex compilation fails (caught in __init__).
        """
        if isinstance(patterns, dict):
            names: list = list(patterns.keys())
            parts: list = [f"(?P<{name}>{pat})" for name, pat in patterns.items()]
        elif isinstance(patterns, list):
            names: list = [f"k{i}" for i in range(len(patterns))]
            parts: list = [f"(?P<{n}>{pat})" for n, pat in zip(names, patterns)]
        else:
            # Should be prevented by __init__, but keep as guardrail.
            raise SplitterConfigException(
                "patterns must be a list of regex strings or a dict[name -> pattern]"
            )

        combined: str = (
            "|".join(parts) if parts else r"(?!x)x"
        )  # never matches if empty
        compiled: re.Pattern = re.compile(combined, flags)
        return names, compiled

    def _split_by_keywords(
        self, text: str
    ) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
        """
        Split ``text`` around matches of ``self.compiled``.

        Respects include_delimiters in {"before", "after", "both", "none"}.

        Args:
            text (str): The text to split.

        Returns:
            Tuple[List[str], List[Tuple[int, int]], List[str]]:
                (chunks, spans, names) where `chunks` are before size re-wrapping,
                spans are (start, end) tuples, and names are group names for each match.
        """

        def _append_chunk(acc: List[str], chunk: str) -> None:
            if chunk and chunk.strip():
                acc.append(chunk)

        chunks: list[str] = []
        spans: list[tuple[int, int]] = []
        names: list[str] = []

        matches: list = list(self.compiled.finditer(text))
        last_idx: int = 0
        pending_prefix: str = ""  # used when include_delimiters is "after" or "both"

        for m in matches:
            start, end = m.span()
            match_txt: str = text[start:end]
            group_name: str = m.lastgroup or "unknown"

            spans.append((start, end))
            names.append(group_name)

            # Build the piece between last match end and this match start,
            # prefixing any pending delimiter
            before_piece: str = pending_prefix + text[last_idx:start]
            pending_prefix: str = ""

            # Attach delimiter to the left side if requested
            if self.include_delimiters in ("before", "both"):
                before_piece += match_txt

            _append_chunk(chunks, before_piece)

            # If delimiter should be on the right, carry it
            # forward to prefix next chunk
            if self.include_delimiters in ("after", "both"):
                pending_prefix = match_txt

            last_idx: int = end

        # Remainder after the last match (may contain pending_prefix)
        remainder: str = pending_prefix + text[last_idx:]
        _append_chunk(chunks, remainder)

        if not chunks:
            return [""], spans, names

        # normalize whitespace trimming for each chunk
        chunks: list[str] = [c.strip() for c in chunks if c and c.strip()]

        if not chunks:
            return [""], spans, names

        return chunks, spans, names

    @staticmethod
    def _soft_wrap(text: str, max_size: int) -> List[str]:
        """
        Greedy soft-wrap by whitespace to respect ``max_size``.

        - If ``len(text) <= max_size``: return ``[text]``.
        - Else: split on whitespace and rebuild lines greedily.
        - If a single token is longer than ``max_size``, it is hard-split.

        Args:
            text (str): Text to wrap.
            max_size (int): Maximum chunk size.

        Returns:
            List[str]: List of size-constrained chunks.
        """
        if max_size <= 0 or len(text) <= max_size:
            return [text] if text else []

        tokens = re.findall(r"\S+|\s+", text)
        out: List[str] = []
        buf = ""
        for tok in tokens:
            if len(buf) + len(tok) <= max_size:
                buf += tok
                continue
            if buf:
                out.append(buf)
                buf = ""
            # token alone is too big -> hard split
            while len(tok) > max_size:
                out.append(tok[:max_size])
                tok = tok[max_size:]
            buf = tok
        if buf:
            out.append(buf)
        return [c for c in (s.strip() for s in out) if c]

    @staticmethod
    def _count_by_name(names: Iterable[str]) -> Dict[str, int]:
        """
        Aggregate match counts by group name (k0/k1/... for list patterns,
        custom names for dict).

        Args:
            names (Iterable[str]): Group names.

        Returns:
            Dict[str, int]: Count of matches per group name.
        """
        counts: Dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        return counts

    def _build_output(
        self,
        reader_output: ReaderOutput,
        chunks: List[str],
        chunk_ids: List[str],
        matches_meta: Dict[str, object],
    ) -> SplitterOutput:
        """
        Assemble a :class:`SplitterOutput` carrying over reader metadata.

        Args:
            reader_output (ReaderOutput): Input document and metadata.
            chunks (List[str]): Final list of chunks.
            chunk_ids (List[str]): Unique chunk IDs.
            matches_meta (Dict[str, object]): Keyword matches metadata.

        Returns:
            SplitterOutput: Populated output object.
        """
        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="keyword",
            split_params={
                "include_delimiters": self.include_delimiters,
                "flags": self.flags,
                "chunk_size": self.chunk_size,
                "pattern_names": self.pattern_names,
            },
            metadata={
                **(reader_output.metadata or {}),
                "keyword_matches": matches_meta,
            },
        )
