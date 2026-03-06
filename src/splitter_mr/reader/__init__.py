from typing import TYPE_CHECKING, Any

from .base_reader import BaseReader

if TYPE_CHECKING:
    from .readers import DoclingReader, ElsevierXmlReader, MarkItDownReader, VanillaReader

__all__ = ["BaseReader", "VanillaReader", "MarkItDownReader", "DoclingReader", "ElsevierXmlReader"]


def __getattr__(name: str) -> Any:
    if name in {"VanillaReader", "MarkItDownReader", "DoclingReader", "ElsevierXmlReader"}:
        from . import readers  # this module is lazy too

        return getattr(readers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
