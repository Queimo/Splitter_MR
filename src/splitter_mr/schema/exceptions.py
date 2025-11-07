# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #

# ---- Base Exception ---- #


class SplitterException(Exception):
    """Base exception for splitter-related errors."""

    pass


class ReaderException(Exception):
    """Base exception for reader-related errors."""

    pass


# ---- General exceptions ---- #


class InvalidChunkException(SplitterException):
    """Raised when chunks cannot be constructed correctly."""


class SplitterOutputException(SplitterException):
    """Raised when SplitterOutput cannot be built or validated."""


# ---- CodeSplitter ---- #


class UnsupportedCodeLanguage(Exception):
    """Raised when the requested code language is not supported by the splitter."""


# ---- HeaderSplitter ---- #


class InvalidHeaderNameError(SplitterException):
    """Raised when a header string isn't of the expected 'Header N' form."""


class HeaderLevelOutOfRangeError(SplitterException):
    """Raised when the parsed header level is outside 1..6."""


class HtmlConversionError(ReaderException):
    """Raised when HTML→Markdown conversion fails."""


class NormalizationError(SplitterException):
    """Raised when Setext→ATX normalization can't be safely applied."""
