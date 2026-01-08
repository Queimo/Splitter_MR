# -------------------------------- #
# ------------ Reader ------------ #
# -------------------------------- #

# ---- Base Warning ---- #


class BaseReaderWarning(UserWarning):
    """
    Base Warning class to all Reader exceptions
    """

    pass


class FiletypeAmbiguityWarning(BaseReaderWarning):
    """
    Warned when filetype heuristics disagree (extension vs DOM sniff).
    """


# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #

# ---- Base Warning ---- #


class BaseSplitterWarning(UserWarning):
    """
    Base Warning class to all Reader exceptions
    """

    pass


# ---- General Warnings ---- #


class SplitterInputWarning(BaseSplitterWarning):
    """
    Warning raised when the splitter input is suspicious (e.g., empty text or
    text expected to be JSON but not parseable as JSON).
    """


class SplitterOutputWarning(BaseSplitterWarning):
    """
    Warning raised when the splitter output present suspicious elements (e.g.,
    empty text or text expected to be JSON but not parseable as JSON).
    """


class ChunkUnderflowWarning(SplitterOutputWarning):
    """
    Warned when fewer chunks are produced than expected from the configured
    chunk_size due to the number of paragraphs being insufficient.
    """


class ChunkOverflowWarning(SplitterOutputWarning):
    """
    Warned when fewer chunks are produced than expected from the configured
    chunk_size due to the number of paragraphs being insufficient.
    """


# ---- HTMLTagSplitter ---- #


class AutoTagFallbackWarning(SplitterInputWarning):
    """
    Warned when HTML Tag Splitter performs auto tagging, e.g., when
    not finding a tag or when no tag is provided.
    """


class BatchHtmlTableWarning(SplitterInputWarning):
    """
    Warned when a tag is presented in a table and the splitting process is being
    produced on batch. In that case, it is splitted by table.
    """
