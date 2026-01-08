import io
import json
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame

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


class RowColumnSplitter(BaseSplitter):
    """Split tabular data by rows, columns, or character-based chunk size.

    RowColumnSplitter splits tabular data (such as CSV, TSV, Markdown tables,
    or JSON tables) into smaller tables based on rows, columns, or by total
    character size while preserving row integrity.

    This splitter supports several modes:

    * **By rows**: Split the table into chunks with a fixed number of rows,
      with optional overlapping rows between chunks.
    * **By columns**: Split the table into chunks by columns, with optional
      overlapping columns between chunks.
    * **By chunk size**: Split the table into markdown-formatted table chunks,
      where each chunk contains as many complete rows as fit under the specified
      character limit, optionally overlapping a fixed number of rows between
      chunks.

    Supported formats for the input text are:

    * CSV / TSV / TXT (comma- or tab-separated values).
    * Markdown tables.
    * JSON in tabular shape (list of dicts or dict of lists).

    Args:
        chunk_size (int, optional):
            Maximum number of characters per chunk when using character-based
            splitting. Defaults to ``1000``.
        num_rows (int, optional):
            Number of rows per chunk when splitting by rows. Mutually
            exclusive with ``num_cols``. Defaults to ``0`` (disabled).
        num_cols (int, optional):
            Number of columns per chunk when splitting by columns. Mutually
            exclusive with ``num_rows``. Defaults to ``0`` (disabled).
        chunk_overlap (int | float, optional):
            Overlap between chunks. Interpretation depends on the mode:

            * When splitting by rows or columns, if an ``int``, it is the
              number of overlapping rows/columns. If a ``float`` in
              ``\\[0, 1)``, it is interpreted as a fraction of the rows/columns
              per chunk.
            * When splitting by ``chunk_size``, it represents the number or
              fraction of overlapping **rows** (not characters).

            Defaults to ``0``.

    Raises:
        SplitterConfigException:
            If configuration is invalid, e.g.:

            * ``num_rows`` and ``num_cols`` are both non-zero.
            * ``chunk_overlap`` as ``float`` is not in ``\\[0, 1)``.
            * ``chunk_overlap`` as ``int`` is negative.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        num_rows: int = 0,
        num_cols: int = 0,
        chunk_overlap: Union[int, float] = 0,
    ):
        super().__init__(chunk_size)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.chunk_overlap = chunk_overlap
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate splitter configuration.

        Performs basic sanity checks on the configuration and raises
        splitter-specific errors when invalid.

        Raises:
            SplitterConfigException:
                If any of the following holds:

                * ``num_rows`` and ``num_cols`` are both non-zero.
                * ``num_rows`` or ``num_cols`` is negative.
                * ``chunk_overlap`` is negative.
                * ``chunk_overlap`` is a ``float`` outside ``\\[0, 1)``.
        """
        if self.num_rows and self.num_cols:
            raise SplitterConfigException(
                "num_rows and num_cols are mutually exclusive."
            )

        if self.num_rows < 0 or self.num_cols < 0:
            raise SplitterConfigException(
                "num_rows and num_cols must be non-negative integers."
            )

        if not isinstance(self.chunk_overlap, (int, float)):
            raise SplitterConfigException("chunk_overlap must be an int or a float.")

        if self.chunk_overlap < 0:
            raise SplitterConfigException("chunk_overlap must be non-negative.")

        if isinstance(self.chunk_overlap, float) and not (0 <= self.chunk_overlap < 1):
            raise SplitterConfigException(
                "chunk_overlap as float must be in the range [0, 1)."
            )

    # ---- Main logic ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split the input tabular data into chunks.

        The splitting strategy is determined by the configuration:

        - If ``num_rows > 0``: split by rows.
        - Else if ``num_cols > 0``: split by columns.
        - Else: split by character-based chunk size in markdown format,
          preserving a header row and never cutting data rows.

        Args:
            reader_output (ReaderOutput):
                Reader output containing at least ``text`` (tabular data as a
                string) and optionally:

                * ``conversion_method``: format hint (``"markdown"``, ``"csv"``,
                  ``"tsv"``, ``"txt"``, ``"json"`` or custom).
                * ``document_name``, ``document_path``, ``document_id``,
                  ``conversion_method``, ``reader_method``, ``ocr_method`` for
                  metadata propagation.

        Returns:
            SplitterOutput:
                Populated splitter output with:

                * ``chunks``: list of chunked tables.
                * ``chunk_id``: generated chunk identifiers.
                * document metadata carried over from ``reader_output``.
                * ``split_method="row_column_splitter"``.
                * ``split_params`` describing the configuration.
                * ``metadata`` containing extra information.

        Raises:
            ReaderOutputException:
                If ``reader_output.text`` is missing or not of type
                ``str``/``None``.
            InvalidChunkException:
                If the number of generated chunk IDs does not match the number
                of chunks.
            SplitterOutputException:
                If constructing :class:`SplitterOutput` fails unexpectedly.

        Warnings:
            SplitterInputWarning:
                If the input text is empty/whitespace-only or if the
                ``conversion_method`` is unknown and a fallback parser is used.
            SplitterOutputWarning:
                If non-empty text produces an empty DataFrame, which may
                indicate malformed input.

        Example:
            Splitting a **CSV table** by **rows** with **overlap**:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters import RowColumnSplitter

            csv_text = (
                "id,name,amount\\n"
                "1,A,10\\n"
                "2,B,20\\n"
                "3,C,30\\n"
                "4,D,40\\n"
            )

            ro = ReaderOutput(
                text=csv_text,
                conversion_method="csv",
                document_name="payments.csv",
                document_path="/tmp/payments.csv",
                document_id="payments-1",
            )

            splitter = RowColumnSplitter(
                num_rows=2,          # 2 rows per chunk
                chunk_overlap=1,     # reuse last 1 row in the next chunk
            )
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            [
              'id,name,amount\\n1,A,10\\n2,B,20',
              'id,name,amount\\n2,B,20\\n3,C,30',
              'id,name,amount\\n3,C,30\\n4,D,40',
            ]
            ```

            ```python
            print(out.metadata["chunks"][0])
            ```
            ```python
            {'rows': [0, 1], 'type': 'row'}
            ```

            Splitting a CSV table by **columns**::

            ```python
            splitter = RowColumnSplitter(
                num_cols=2,          # 2 columns per chunk
                chunk_overlap=1,     # reuse 1 column in the next chunk
            )
            out = splitter.split(ro)

            print(out.chunks)
            ```
            ```python
            [['id', 1, 2, 3, 4], ['name', 'A', 'B', 'C', 'D']]
            ```
            ```python
            print(out.metadata["chunks"][0])
            ```

            ```python
            {'cols': ['id', 'name'], 'type': 'column'}
            ```

            Splitting by **character-based chunk size** (markdown output)::

            ```python
            md_text = '''
            | id | name | amount |
            |----|------|--------|
            | 1  | A    | 10     |
            | 2  | B    | 20     |
            | 3  | C    | 30     |
            | 4  | D    | 40     |
            '''.strip()

            ro = ReaderOutput(
                text=md_text,
                conversion_method="markdown",
                document_name="table.md",
            )

            splitter = RowColumnSplitter(
                chunk_size=80,        # max ~80 chars per chunk
                chunk_overlap=0.25,   # 25% row overlap between chunks
            )
            out = splitter.split(ro)

            for i, (chunk, meta) in enumerate(
                zip(out.chunks, out.metadata["chunks"]), start=1
            ):
                print(f"--- CHUNK {i} ---")
                print(chunk)
                print("rows:", meta["rows"])   # original row indices
            ```

            Handling **unknown conversion_method** with JSON/CSV fallback::

            ```python
            json_text = '''
            [
                {"id": 1, "name": "A", "amount": 10},
                {"id": 2, "name": "B", "amount": 20}
            ]
            '''.strip()

            ro = ReaderOutput(
                text=json_text,
                conversion_method="unknown",   # triggers JSON → CSV fallback logic
            )

            splitter = RowColumnSplitter(num_rows=1)
            out = splitter.split(ro)
            print(out.chunks)
            ```
        """
        # Minimal ReaderOutput validation (type-level issues)
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

        # Load tabular data into a DataFrame
        df = self._load_tabular(reader_output)
        orig_method = (reader_output.conversion_method or "").lower()
        col_names = df.columns.tolist()

        # If text is non-empty but we got no rows/columns, warn
        if text.strip() and df.empty:
            warnings.warn(
                "RowColumnSplitter produced an empty DataFrame from non-empty "
                "input text; this may indicate malformed or unsupported table "
                "format.",
                SplitterOutputWarning,
            )

        # Dispatch to splitting strategy
        if self.num_rows > 0:
            chunks, meta_per_chunk = self._split_by_rows(df, orig_method)
        elif self.num_cols > 0:
            chunks, meta_per_chunk = self._split_by_columns(df, orig_method, col_names)
        else:
            chunks, meta_per_chunk = self._split_by_chunk_size(df)

        # Generate chunk IDs and validate
        chunk_ids = self._generate_chunk_ids(len(chunks))
        if len(chunk_ids) != len(chunks):
            raise InvalidChunkException(
                "Number of chunk IDs does not match number of chunks "
                f"(chunk_ids={len(chunk_ids)}, chunks={len(chunks)})."
            )

        # Build SplitterOutput, wrapping any unexpected issues
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
                split_method="row_column_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "num_rows": self.num_rows,
                    "num_cols": self.num_cols,
                    "chunk_overlap": self.chunk_overlap,
                },
                metadata={"chunks": meta_per_chunk},
            )
        except Exception as exc:
            raise SplitterOutputException(
                f"Failed to build SplitterOutput in RowColumnSplitter: {exc}"
            ) from exc

    # ---- Internal helpers ---- #

    # Splitting strategies

    def _split_by_rows(
        self,
        df: pd.DataFrame,
        method: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split the DataFrame into chunks by rows.

        Uses ``num_rows`` and ``chunk_overlap`` to build overlapping row-based
        chunks. Each chunk contains full rows; rows are never split.

        Args:
            df (pd.DataFrame):
                Input table as a DataFrame.
            method (str):
                Original conversion method (e.g., ``"markdown"`` or ``"csv"``);
            used to decide the output string format.

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]:
                A tuple ``(chunks, metadata)`` where:

                * ``chunks`` is a list of stringified table chunks.
                * ``metadata`` is a list of per-chunk metadata dicts, each
                  containing:

                  * ``"rows"``: list of DataFrame indices included.
                  * ``"type"``: ``"row"``.

        """
        chunks: List[str] = []
        meta_per_chunk: List[Dict[str, Any]] = []

        overlap = self._get_overlap(self.num_rows)
        step = self.num_rows - overlap if (self.num_rows - overlap) > 0 else 1

        for i in range(0, len(df), step):
            chunk_df = df.iloc[i : i + self.num_rows]
            if chunk_df.empty:
                continue
            chunk_str = self._to_str(chunk_df, method)
            chunks.append(chunk_str)
            meta_per_chunk.append(
                {
                    "rows": chunk_df.index.tolist(),
                    "type": "row",
                }
            )

        return chunks, meta_per_chunk

    def _split_by_columns(
        self,
        df: pd.DataFrame,
        method: str,
        col_names: List[str],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split the DataFrame into chunks by columns.

        Uses ``num_cols`` and ``chunk_overlap`` to build overlapping
        column-based chunks. Each chunk preserves all rows but only a subset
        of columns.

        Args:
            df (pd.DataFrame):
                Input table as a DataFrame.
            method (str):
                Original conversion method (e.g., ``"markdown"`` or ``"csv"``);
                used to decide the output string format.
            col_names (List[str]):
                List of column names in the order used for slicing.

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]:
                A tuple ``(chunks, metadata)`` where:

                * ``chunks`` is a list of stringified table chunks.
                * ``metadata`` is a list of per-chunk metadata dicts, each
                  containing:

                  * ``"cols"``: list of column names included.
                  * ``"type"``: ``"column"``.
        """
        chunks: List[str] = []
        meta_per_chunk: List[Dict[str, Any]] = []

        overlap = self._get_overlap(self.num_cols)
        step = self.num_cols - overlap if (self.num_cols - overlap) > 0 else 1
        total_cols = len(col_names)

        for i in range(0, total_cols, step):
            sel_cols = col_names[i : i + self.num_cols]
            if not sel_cols:
                continue
            chunk_df = df[sel_cols]
            chunk_str = self._to_str(chunk_df, method, colwise=True)
            chunks.append(chunk_str)
            meta_per_chunk.append(
                {
                    "cols": sel_cols,
                    "type": "column",
                }
            )

        return chunks, meta_per_chunk

    def _split_by_chunk_size(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split the DataFrame into markdown chunks constrained by ``chunk_size``.

        The header is always preserved, rows are never cut, and overlap is
        applied in terms of full rows (not characters). Each chunk is rendered
        as a markdown table string.

        Args:
            df (pd.DataFrame):
                Input table as a DataFrame.

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]:
                A tuple ``(chunks, metadata)`` where:

                * ``chunks`` is a list of markdown-formatted tables.
                * ``metadata`` is a list of per-chunk metadata dicts, each
                  containing:

                  * ``"rows"``: list of row indices in the original table.
                  * ``"type"``: ``"char_row"``.

        Raises:
            SplitterConfigException:
                If ``chunk_size`` is too small to fit the header and at
                least one row.
        """
        chunks: List[str] = []
        meta_per_chunk: List[Dict[str, Any]] = []

        # Build header
        header_lines = self._get_markdown_header(df)
        header_length = len(header_lines)

        # Build per-row markdown representations
        row_md_list = [self._get_markdown_row(df, i) for i in range(len(df))]
        row_len_list = [len(r) + 1 for r in row_md_list]  # +1 for newline

        # Input validation
        if row_md_list:
            min_required = header_length + max(row_len_list)
            if self.chunk_size < min_required:
                raise SplitterConfigException(
                    "chunk_size is too small to fit the header and at least one row; "
                    f"minimum required is {min_required}, got {self.chunk_size}."
                )

        i = 0
        n = len(row_md_list)

        while i < n:
            curr_chunk: List[str] = []
            curr_len = header_length
            j = i

            # Accumulate rows while there is space
            while j < n and curr_len + row_len_list[j] <= self.chunk_size:
                curr_chunk.append(row_md_list[j])
                curr_len += row_len_list[j]
                j += 1

            rows_in_chunk = j - i
            chunk_str = header_lines + "\n".join(curr_chunk)
            chunks.append(chunk_str)
            meta_per_chunk.append(
                {
                    "rows": list(range(i, j)),
                    "type": "char_row",
                }
            )

            # --- compute overlap AFTER we know rows_in_chunk ---
            if isinstance(self.chunk_overlap, float):
                overlap_rows = int(rows_in_chunk * self.chunk_overlap)
            else:
                overlap_rows = int(self.chunk_overlap)

            # Avoid infinite loops when overlap >= rows_in_chunk
            overlap_rows = min(overlap_rows, max(rows_in_chunk - 1, 0))
            i = j - overlap_rows

        return chunks, meta_per_chunk

    # ---- Internal helpers ---- #

    def _get_overlap(self, base: int) -> int:
        """Compute integer overlap from ``chunk_overlap`` configuration.

        Args:
            base (int):
                Base number (rows or columns) from which to compute the
                overlap when ``chunk_overlap`` is a float.

        Returns:
            int:
                Overlap expressed as an integer count of rows or columns.
        """
        if isinstance(self.chunk_overlap, float):
            return int(base * self.chunk_overlap)
        return int(self.chunk_overlap)

    def _load_tabular(self, reader_output: ReaderOutput) -> pd.DataFrame:
        """Load and parse input tabular data into a DataFrame.

        The parsing strategy is driven by ``reader_output.conversion_method``:

        * ``"markdown"`` → parse markdown table.
        * ``"csv"`` / ``"txt"`` → parse as CSV.
        * ``"tsv"`` → parse as TSV.
        * ``"json"`` → parse as tabular JSON (list-of-dicts or dict-of-lists).
        * Any other value (including ``None``) triggers a fallback:
          try tabular JSON, then CSV, with warnings.

        Args:
            reader_output (ReaderOutput):
                Reader output containing the raw tabular text and
                ``conversion_method`` hint.

        Returns:
            pd.DataFrame:
                DataFrame representation of the table. May be empty if the
                input text is empty or contains no parsable rows.

        Raises:
            pandas.errors.ParserError:
                If a known format (e.g. ``"json"`` or markdown) is declared
                but the content is malformed and cannot be parsed.

        Warnings:
            SplitterInputWarning:
                If:

                * The input text is empty or whitespace-only.
                * The ``conversion_method`` is unknown and a fallback parser
                  is used (JSON or CSV).
        """
        text = reader_output.text or ""
        if not text.strip():
            warnings.warn(
                "RowColumnSplitter received empty or whitespace-only text; "
                "resulting chunks will be empty.",
                SplitterInputWarning,
                stacklevel=3,
            )
            return pd.DataFrame()

        method = (reader_output.conversion_method or "").lower()

        # Local helpers / factory
        def _read_csv(src: str, **kwargs: Any) -> DataFrame:
            return pd.read_csv(io.StringIO(src), **kwargs)

        def _read_markdown(src: str) -> DataFrame:
            return self._parse_markdown_table(src)

        def _read_tsv(src: str) -> DataFrame:
            return _read_csv(src, sep="\t")

        def _read_json(src: str) -> DataFrame:
            df = self._try_json_tabular(src)
            if df is None:
                # If 'json' is declared but the content is not tabular JSON,
                # let this be an error rather than silently guessing.
                raise pd.errors.ParserError("Input is not tabular JSON")
            return df

        parser_map: Dict[str, Callable[[str], DataFrame]] = {
            "markdown": _read_markdown,
            "csv": _read_csv,
            "txt": _read_csv,
            "tsv": _read_tsv,
            "json": _read_json,
        }

        parser = parser_map.get(method)
        if parser is not None:
            return parser(text)

        # Unknown / missing method:
        # Try tabular JSON first, then fall back to CSV, and warn.
        json_df = self._try_json_tabular(text)
        if json_df is not None:
            warnings.warn(
                (
                    f"Unknown conversion_method '{method}', but input parsed as "
                    "tabular JSON. Treating as JSON table."
                ),
                SplitterInputWarning,
            )
            return json_df

        warnings.warn(
            (
                f"Unknown conversion_method '{method}', falling back to CSV parser. "
                "Check that the input is comma-separated."
            ),
            SplitterInputWarning,
        )
        return _read_csv(text)

    def _try_json_tabular(self, text: str) -> Optional[pd.DataFrame]:
        """Try to interpret text as tabular JSON.

        Accepted shapes:

        * ``List[Dict[str, Any]]``: rows as dicts.
        * ``Dict[str, List[Any]]``: columns as lists.

        Args:
            text (str):
                Raw JSON string.

        Returns:
            Optional[pd.DataFrame]:
                A DataFrame if parsing succeeds and a tabular shape is
                detected, otherwise ``None``.
        """
        try:
            js = json.loads(text)
        except Exception:
            return None

        if isinstance(js, list) and js and all(isinstance(row, dict) for row in js):
            return pd.DataFrame(js)

        if isinstance(js, dict):
            return pd.DataFrame(js)

        return None

    def _parse_markdown_table(self, md: str) -> pd.DataFrame:
        """Parse a markdown table string into a DataFrame.

        Ignores non-table lines and trims markdown-specific formatting.
        Also handles the separator line (e.g. ``---``) in the header.

        Args:
            md (str):
                Markdown text that may contain a table.

        Returns:
            pd.DataFrame:
                Parsed table as a DataFrame.

        Raises:
            pandas.errors.ParserError:
                If the markdown table is malformed and cannot be parsed.
        """
        table_lines: List[str] = []
        started = False
        for line in md.splitlines():
            if re.match(r"^\s*\|.*\|\s*$", line):
                started = True
                table_lines.append(line.strip())
            elif started and not line.strip():
                break
        table_md = "\n".join(table_lines)
        table_io = io.StringIO(
            re.sub(
                r"^\s*\|",
                "",
                re.sub(r"\|\s*$", "", table_md, flags=re.MULTILINE),
                flags=re.MULTILINE,
            )
        )
        try:
            df = pd.read_csv(table_io, sep="|").rename(
                lambda x: x.strip(), axis="columns"
            )
        except pd.errors.EmptyDataError:
            # No actual table content (e.g., markdown text with no table lines)
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            # Real markdown table that is malformed → surface as ParserError
            raise pd.errors.ParserError(f"Malformed markdown table: {e}") from e

        if not df.empty and all(re.match(r"^-+$", str(x).strip()) for x in df.iloc[0]):
            df = df.drop(df.index[0]).reset_index(drop=True)
        return df

    def _to_str(self, df: pd.DataFrame, method: str, colwise: bool = False) -> str:
        """Convert a chunk DataFrame to string representation.

        Args:
            df (pd.DataFrame):
                Chunk DataFrame to convert.
            method (str):
                Original conversion method (e.g., ``"markdown"``, ``"csv"``).
            colwise (bool, optional):
                If ``True``, output a list-of-lists representation for
                column-based chunks. If ``False``, output a table-like string
                (markdown or CSV). Defaults to ``False``.

        Returns:
            str:
                String representation of the chunk.
        """
        if colwise:
            return (
                "["
                + ", ".join(str([col] + df[col].tolist()) for col in df.columns)  # noqa: W503
                + "]"  # noqa: W503
            )
        if method in ("markdown", "md"):
            return df.to_markdown(index=False)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().strip("\n")

    @staticmethod
    def _get_markdown_header(df: pd.DataFrame) -> str:
        """Return markdown header + separator with trailing newline.

        Args:
            df (pd.DataFrame):
                DataFrame whose columns define the header.

        Returns:
            str:
                Markdown-formatted header (two lines) followed by a newline.
        """
        lines = df.head(0).to_markdown(index=False).splitlines()
        return "\n".join(lines[:2]) + "\n"

    @staticmethod
    def _get_markdown_row(df: pd.DataFrame, row_idx: int) -> str:
        """Return a single markdown-formatted row from the DataFrame.

        Args:
            df (pd.DataFrame):
                DataFrame containing the table.
            row_idx (int):
                Index of the row to extract.

        Returns:
            str:
                Markdown-formatted row string (data row only).
        """
        row = df.iloc[[row_idx]]
        md = row.to_markdown(index=False).splitlines()
        return md[-1]
