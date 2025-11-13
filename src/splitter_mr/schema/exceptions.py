# -------------------------------- #
# ------------ Reader ------------ #
# -------------------------------- #

# ---- Base Exception ---- #


class ReaderException(Exception):
    """Base exception for reader-related errors."""

    pass


# ---- Conversion Exceptions ---- #


class HtmlConversionError(ReaderException):
    """Raised when HTML→Markdown conversion fails."""


# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #

# ---- Base Exception ---- #


class SplitterException(Exception):
    """Base exception for splitter-related errors."""

    pass


# ---- General exceptions ---- #


class InvalidChunkException(SplitterException):
    """Raised when chunks cannot be constructed correctly."""


class SplitterConfigException(SplitterException):
    """Raised when the configuration provided to the Splitter class is not correct"""


class SplitterOutputException(SplitterException):
    """Raised when SplitterOutput cannot be built or validated."""


# ---- HeaderSplitter ---- #


class InvalidHeaderNameError(SplitterConfigException):
    """Raised when a header string isn't of the expected 'Header N' form."""


class HeaderLevelOutOfRangeError(SplitterConfigException):
    """Raised when the parsed header level is outside 1..6."""


class NormalizationError(ReaderException):
    """Raised when Setext→ATX normalization can't be safely applied."""


# ---- HTMLTagSplitter ---- #


class InvalidHtmlTagError(ReaderException):
    """
    Raised when an invalid HTML Tag is provided or when it is missing
    in the document.
    """
