import warnings
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from ...embedding import BaseEmbedding
from ...schema import (
    DEFAULT_BREAKPOINTS,
    BreakpointThresholdType,
    ReaderOutput,
    ReaderOutputException,
    SplitterConfigException,
    SplitterInputWarning,
    SplitterOutput,
    SplitterOutputException,
    SplitterOutputWarning,
)
from ...splitter import BaseSplitter
from .sentence_splitter import SentenceSplitter

# ---- External helpers ---- #


def _cosine_similaritynp(a: List[float], b: List[float], eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors using NumPy for speed.

    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.
        eps (float): Numerical stability epsilon.

    Returns:
        float: Cosine similarity in [-1, 1].
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.maximum(np.linalg.norm(va) * np.linalg.norm(vb), eps))
    return float(np.dot(va, vb) / denom)


def _combine_sentences(
    sentences: List[Dict[str, Any]], buffer_size: int
) -> List[Dict[str, Any]]:
    """Create a sliding window string around each sentence using NumPy helpers.

    For each sentence i, concatenates up to `buffer_size` neighbors on both sides.

    Args:
        sentences (List[Dict[str, Any]]): Items with {"sentence": str, "index": int}.
        buffer_size (int): Number of neighbors on each side.

    Returns:
        List[Dict[str, Any]]: In-place augmented with "combined_sentence".
    """
    n = len(sentences)
    for i in range(n):
        left = int(np.maximum(0, i - buffer_size))
        right = int(np.minimum(n, i + 1 + buffer_size))
        parts = [sentences[j]["sentence"] for j in range(left, right)]
        sentences[i]["combined_sentence"] = " ".join(parts).strip()
    return sentences


class SemanticSplitter(BaseSplitter):
    """
    Split text into semantically coherent chunks using embedding similarity.

    **Pipeline:**

    - Split text into sentences via `SentenceSplitter` (one sentence chunks).
    - Build a sliding window around each sentence (`buffer_size`).
    - Embed each window with `BaseEmbedding` (batched).
    - Compute cosine *distances* between consecutive windows (1 - cosine_sim).
    - Pick breakpoints using a thresholding strategy, or aim for `number_of_chunks`.
    - Join sentences between breakpoints; enforce minimum size via `chunk_size`.

    Args:
        embedding: Embedding backend implementing an ``embed_documents(texts: List[str])``
            method. Typically wraps a model from OpenAI, Azure, or a local
            embedding model.
        buffer_size: Number of neighbouring sentences to include on each side
            when building the contextual window for each sentence. A value of
            ``1`` means "current sentence plus one sentence to the left and
            one to the right" (where available).
        breakpoint_threshold_type: Strategy used to decide where to place
            breakpoints. Supported values are:

            * ``"percentile"`` – cut where distances exceed a percentile
              of the distance distribution.
            * ``"standard_deviation"`` – cut where distances exceed
              ``mean + k * std``.
            * ``"interquartile"`` – cut where distances exceed
              ``mean + k * IQR``.
            * ``"gradient"`` – cut where the *gradient* of distances
              exceeds a percentile threshold.
        breakpoint_threshold_amount: Strength of the threshold for the
            chosen strategy. Meaning depends on ``breakpoint_threshold_type``:

            * For ``"percentile"`` / ``"gradient"``:
              value in ``[0, 100]`` interpreted as a percentile, or a
              value in ``(0, 1]`` interpreted as a ratio and automatically
              scaled to ``[0, 100]``.
            * For ``"standard_deviation"`` / ``"interquartile"``:
              finite multiplier ``k`` applied to the deviation term
              (std or IQR).
            If ``None``, a default from ``DEFAULT_BREAKPOINTS`` is used.
        number_of_chunks: Desired number of output chunks. When provided,
            the splitter selects the largest distances to approximate this
            target (subject to document length and `chunk_size`). Must be a
            positive, finite value; non-integers are allowed but will be
            truncated internally.
        chunk_size: Minimum allowed chunk size in characters. Short segments
            below this size are merged forward to avoid excessively small,
            fragmented chunks.

    Raises:
        SplitterConfigException:
            - If `embedding` does not provide an `embed_documents` method.
            - If `buffer_size < 0`.
            - If `breakpoint_threshold_type` is not supported.
            - If `breakpoint_threshold_amount` is invalid for the chosen strategy.
            - If `number_of_chunks` is non-positive or non-finite.

    Warnings:
        SplitterInputWarning:
            - If `breakpoint_threshold_amount` in (0, 1] is auto-scaled as
              a ratio to a percentile in [0, 100].
            - If `number_of_chunks` is not an integer; it will be truncated
              when used internally.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        *,
        buffer_size: int = 1,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__(chunk_size=chunk_size)

        # Validate embedding backend
        if embedding is None or not hasattr(embedding, "embed_documents"):
            raise SplitterConfigException(
                "SemanticSplitter requires an embedding backend with an "
                "'embed_documents' method."
            )
        self.embedding = embedding

        # Validate buffer size
        if buffer_size < 0:
            raise SplitterConfigException("buffer_size must be >= 0.")
        self.buffer_size = int(buffer_size)

        # Validate breakpoint strategy
        valid_types = set(DEFAULT_BREAKPOINTS.keys())
        if breakpoint_threshold_type not in valid_types:
            raise SplitterConfigException(
                f"Invalid breakpoint_threshold_type={breakpoint_threshold_type!r}. "
                f"Expected one of {sorted(valid_types)}."
            )

        self.breakpoint_threshold_type = cast(
            BreakpointThresholdType, breakpoint_threshold_type
        )

        # Resolve threshold amount
        raw_amount = (
            DEFAULT_BREAKPOINTS[self.breakpoint_threshold_type]
            if breakpoint_threshold_amount is None
            else float(breakpoint_threshold_amount)
        )

        # Normalise / validate threshold amount per strategy
        if self.breakpoint_threshold_type in ("percentile", "gradient"):
            amt = float(raw_amount)
            if 0.0 < amt <= 1.0:
                # interpret as ratio -> scale to [0, 100]
                warnings.warn(
                    "SemanticSplitter: breakpoint_threshold_amount given in (0, 1]; "
                    "interpreting as a ratio and scaling to [0, 100] percent.",
                    SplitterInputWarning,
                )
                amt *= 100.0
            if not 0.0 <= amt <= 100.0:
                raise SplitterConfigException(
                    "For 'percentile' and 'gradient' strategies, "
                    "breakpoint_threshold_amount must be in [0, 100] "
                    "(or (0, 1] to be interpreted as a ratio)."
                )
            self.breakpoint_threshold_amount = amt
        else:
            # std-dev / IQR strategies: just require finite value
            if not np.isfinite(raw_amount):
                raise SplitterConfigException(
                    "breakpoint_threshold_amount must be finite."
                )
            self.breakpoint_threshold_amount = float(raw_amount)

        # Validate number_of_chunks
        if number_of_chunks is not None:
            if not np.isfinite(number_of_chunks) or number_of_chunks <= 0:
                raise SplitterConfigException(
                    "number_of_chunks must be a positive finite integer when provided."
                )
            if not float(number_of_chunks).is_integer():
                warnings.warn(
                    f"SemanticSplitter: number_of_chunks={number_of_chunks!r} is not "
                    "an integer; it will be truncated when used internally.",
                    SplitterInputWarning,
                )
        self.number_of_chunks = number_of_chunks

        self._sentence_splitter = SentenceSplitter(
            chunk_size=1, chunk_overlap=0, separators=[".", "!", "?"]
        )

    # ---- Main method ---- #

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split the document text into semantically coherent chunks.

        This method uses sentence embeddings to find semantic breakpoints.
        Sentences are embedded in overlapping windows (controlled by `buffer_size`),
        then cosine distances between consecutive windows are used to detect topic
        shifts. Breakpoints are determined using either a threshold strategy
        (percentile, std-dev, IQR, gradient) or by targeting a number of chunks.

        Args:
            reader_output (ReaderOutput): Input text and associated metadata.

        Returns:
            SplitterOutput: Structured splitter output containing:
                * ``chunks`` — list of semantically grouped text segments.
                * ``chunk_id`` — corresponding unique identifiers.
                * document metadata and splitter parameters.

        Raises:
            ReaderOutputException:
                If the provided text is empty, None, or otherwise invalid.
            SplitterConfigException:
                If an invalid configuration is detected at runtime
                (defensive re-checks).
            SplitterOutputException:
                - If sentence splitting fails unexpectedly.
                - If the embedding backend fails or returns invalid shapes.
                - If non-finite distances/gradients are produced.
                - If post-processing of distances fails.

        Warnings:
            SplitterInputWarning:
                - If certain configuration values are auto-normalised (see __init__).
            SplitterOutputWarning:
                - If no semantic breakpoints are detected and a single chunk is
                  returned for multi-sentence input.
                - If the requested `number_of_chunks` is larger than the maximum
                  achievable for the given document.
                - If all candidate cuts are rejected due to `chunk_size`, resulting
                  in a single merged chunk.

        Notes:
            - With a single sentence (or 2 in gradient mode), returns text as-is.
            - ``chunk_size`` acts as the *minimum* allowed chunk size; small
              segments are merged forward.
            - The `buffer_size` defines how much contextual overlap each sentence
              has for embedding (e.g., 1 = one sentence on either side).

        Example:
            **Basic usage** with a **custom embedding backend**:

            ```python
            from splitter_mr.schema import ReaderOutput
            from splitter_mr.splitter.splitters.semantic_splitter import SemanticSplitter
            from splitter_mr.embedding import BaseEmbedding

            class DummyEmbedding(BaseEmbedding):
                \"\"\"Minimal embedding backend for demonstration purposes.\"\"\"
                model_name = "dummy-semantic-model"

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    # Return a simple fixed-length vector per text
                    dim = 8
                    return [[float(i) for i in range(dim)] for _ in texts]

            text = (
                "Cats like to sleep in the sun. "
                "They often chase laser pointers. "
                "Neural networks can classify animal images. "
                "Transformers are widely used in NLP."
            )

            ro = ReaderOutput(text=text, document_name="semantic_demo.txt")

            splitter = SemanticSplitter(
                embedding=DummyEmbedding(),
                buffer_size=1,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75.0,
                chunk_size=50,
            )

            output = splitter.split(ro)

            print(output.chunks)
            ```

            Targeting a **specific number of chunks**:

            ```python
            splitter = SemanticSplitter(
                embedding=DummyEmbedding(),
                buffer_size=1,
                number_of_chunks=3,
                chunk_size=40,
            )

            output = splitter.split(ro)
            print(output.chunks)          # ~3 semantic chunks (subject to document length)
            print(output.split_method)    # "semantic_splitter"
            print(output.split_params)    # includes threshold config and model name
            ```
        """
        text: str = reader_output.text
        if text is None or text.strip() == "":
            raise ReaderOutputException("ReaderOutput.text is empty or None.")

        sentences: list[str] = self._split_into_sentences(reader_output)

        # Edge cases where thresholds aren't meaningful
        if len(sentences) <= 1:
            chunks = sentences if sentences else [text]
        elif self.breakpoint_threshold_type == "gradient" and len(sentences) == 2:
            chunks = sentences
        else:
            distances, sentence_dicts = self._calculate_sentence_distances(sentences)

            indices_above: list[int]

            if self.number_of_chunks is not None and distances:
                # Warn if target number_of_chunks is unattainable
                max_possible = len(distances) + 1
                if self.number_of_chunks > max_possible:
                    warnings.warn(
                        "SemanticSplitter: requested number_of_chunks="
                        f"{self.number_of_chunks} is larger than the maximum "
                        f"possible ({max_possible}); using {max_possible} instead.",
                        SplitterOutputWarning,
                    )

                # Pick top (k-1) distances as breakpoints
                k = int(self.number_of_chunks)
                m = max(0, min(k - 1, len(distances)))  # number of cuts to make
                if m == 0:
                    indices_above = []  # single chunk
                else:
                    # indices of the m largest distances (breaks), sorted in ascending order
                    idxs = np.argsort(np.asarray(distances))[-m:]
                    indices_above = sorted(int(i) for i in idxs.tolist())
            else:
                threshold, ref_array = self._calculate_breakpoint_threshold(distances)
                indices_above = [
                    i for i, val in enumerate(ref_array) if val > threshold
                ]

            # Warn if no breakpoints found (but only when >1 chunk requested)
            if (
                not indices_above  # noqa: W503
                and len(sentences) > 1  # noqa: W503
                and (self.number_of_chunks is None or self.number_of_chunks > 1)  # noqa: W503
            ):
                warnings.warn(
                    "SemanticSplitter did not detect any semantic breakpoints; "
                    "returning a single chunk.",
                    SplitterOutputWarning,
                )

            chunks = []
            start_idx = 0

            for idx in indices_above:
                end = idx + 1  # inclusive slice end
                candidate = " ".join(
                    d["sentence"] for d in sentence_dicts[start_idx:end]
                ).strip()
                if len(candidate) < self.chunk_size:
                    # too small: keep accumulating (do NOT move start_idx)
                    continue
                chunks.append(candidate)
                start_idx = end

            # Tail (always emit whatever remains)
            if start_idx < len(sentence_dicts):
                tail = " ".join(
                    d["sentence"] for d in sentence_dicts[start_idx:]
                ).strip()
                if tail:
                    chunks.append(tail)

            if not chunks:
                chunks = [" ".join(sentences).strip() or (reader_output.text or "")]

        # Warn if everything got merged into a single chunk due to chunk_size
        if (
            len(chunks) == 1  # noqa: W503
            and len(" ".join(sentences)) >= self.chunk_size  # noqa: W503
            and len(sentences) > 1  # noqa: W503
        ):
            warnings.warn(
                "SemanticSplitter merged all sentences into a single chunk because "
                "no candidate segments met the minimum chunk_size.",
                SplitterOutputWarning,
            )

        # Append chunk_ids and metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()
        model_name = getattr(self.embedding, "model_name", None)

        # Produce output
        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="semantic_splitter",
            split_params={
                "buffer_size": self.buffer_size,
                "breakpoint_threshold_type": self.breakpoint_threshold_type,
                "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
                "number_of_chunks": self.number_of_chunks,
                "chunk_size": self.chunk_size,
                "model_name": model_name,
            },
            metadata=metadata,
        )

    # ---- Internal helpers ---- #

    def _split_into_sentences(self, reader_output: ReaderOutput) -> List[str]:
        """Split the input text into sentences using `SentenceSplitter` (no overlap).

        Args:
            reader_output (ReaderOutput): The document to split.

        Returns:
            List[str]: List of sentences preserving punctuation.

        Raises:
            SplitterOutputException: If the underlying sentence splitter fails
                in an unexpected way.
        """
        try:
            sent_out = self._sentence_splitter.split(reader_output)
        except SplitterOutputException:
            # Propagate domain-specific splitter failures as-is
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise SplitterOutputException(
                f"Sentence splitting failed in SemanticSplitter: {exc}"
            ) from exc
        return sent_out.chunks

    def _calculate_sentence_distances(
        self, single_sentences: List[str]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Embed sentence windows (batch) and compute consecutive cosine distances.

        Args:
            single_sentences (List[str]): Sentences in order.

        Returns:
            Tuple[List[float], List[Dict[str, Any]]]:
                - distances between consecutive windows (len = n-1)
                - sentence dicts enriched with combined text and embeddings

        Raises:
            SplitterOutputException:
                - If the embedding backend fails during `embed_documents`.
                - If the number of returned embeddings does not match the number
                  of windows.
                - If non-finite (NaN/inf) distances are produced.
        """
        # Prepare sentence dicts and combine with buffer
        sentences = [
            {"sentence": s, "index": i} for i, s in enumerate(single_sentences)
        ]
        sentences = _combine_sentences(sentences, self.buffer_size)

        # Batch embed all combined sentences
        windows = [item["combined_sentence"] for item in sentences]
        try:
            embeddings = self.embedding.embed_documents(windows)
        except Exception as exc:  # pragma: no cover - defensive
            raise SplitterOutputException(
                f"Embedding backend failed during SemanticSplitter: {exc}"
            ) from exc

        if len(embeddings) != len(sentences):
            raise SplitterOutputException(
                "Embedding backend returned a number of vectors that does not match "
                f"the number of windows in SemanticSplitter "
                f"({len(embeddings)} embeddings for {len(sentences)} windows)."
            )

        for item, emb in zip(sentences, embeddings):
            item["combined_sentence_embedding"] = emb

        # Distances (1 - cosine similarity) between consecutive windows
        n = len(sentences)
        if n <= 1:
            return [], sentences

        distances: List[float] = []
        for i in range(n - 1):
            sim = _cosine_similaritynp(
                sentences[i]["combined_sentence_embedding"],
                sentences[i + 1]["combined_sentence_embedding"],
            )
            dist = 1.0 - sim
            distances.append(dist)
            sentences[i]["distance_to_next"] = dist

        distances_arr = np.asarray(distances, dtype=np.float64)
        if not np.all(np.isfinite(distances_arr)):
            raise SplitterOutputException(
                "Non-finite values (NaN/inf) encountered in semantic distances; "
                "embedding backend produced invalid vectors."
            )

        return distances_arr.tolist(), sentences

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """Estimate a percentile threshold to reach `number_of_chunks`.

        Maps desired chunks x∈[1, len(distances)] to percentile y∈[100, 0].

        Args:
            distances (List[float]): Consecutive distances.

        Returns:
            float: Threshold value as a percentile over `distances`.
        """
        assert self.number_of_chunks is not None
        x1, y1 = float(len(distances)), 0.0
        x2, y2 = 1.0, 100.0
        x = max(min(float(self.number_of_chunks), x1), x2)
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1) if x2 != x1 else y2
        y = float(np.clip(y, 0.0, 100.0))
        return float(np.percentile(distances, y)) if distances else 0.0

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        """Compute the breakpoint threshold and reference array per selected strategy.

        Args:
            distances (List[float]): Consecutive distances between windows.

        Returns:
            Tuple[float, List[float]]: (threshold, reference_array)
                If strategy == "gradient", reference_array is the gradient;
                otherwise it's `distances`.

        Raises:
            SplitterOutputException:
                If non-finite values are detected in the distance or gradient arrays.
            SplitterConfigException:
                If an unexpected `breakpoint_threshold_type` is encountered.
        """
        if not distances:
            return 0.0, distances

        arr = np.asarray(distances, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise SplitterOutputException(
                "Non-finite values (NaN/inf) encountered in distances when "
                "computing breakpoint threshold."
            )

        if self.breakpoint_threshold_type == "percentile":
            return (
                float(np.percentile(arr, self.breakpoint_threshold_amount)),
                arr.tolist(),
            )

        if self.breakpoint_threshold_type == "standard_deviation":
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            return mu + self.breakpoint_threshold_amount * sd, arr.tolist()

        if self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(arr, [25.0, 75.0])
            iqr = float(q3 - q1)
            mu = float(np.mean(arr))
            return mu + self.breakpoint_threshold_amount * iqr, arr.tolist()

        if self.breakpoint_threshold_type == "gradient":
            grads_arr = np.gradient(arr)
            if not np.all(np.isfinite(grads_arr)):
                raise SplitterOutputException(
                    "Non-finite values (NaN/inf) encountered in gradient distances."
                )
            grads = grads_arr.tolist()
            thr = float(np.percentile(grads_arr, self.breakpoint_threshold_amount))
            return thr, grads  # use gradient array as the reference

        # Should be prevented by __init__, but keep as a defensive guard
        raise SplitterConfigException(
            f"Unexpected breakpoint_threshold_type: {self.breakpoint_threshold_type}"
        )
