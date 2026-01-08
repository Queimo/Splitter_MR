import base64
import logging
from collections import defaultdict
from io import BytesIO
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from ...model import BaseVisionModel
from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DEFAULT_IMAGE_PLACEHOLDER,
    DEFAULT_PAGE_PLACEHOLDER,
)

logger = logging.getLogger(__name__)


class PDFPlumberReaderException(RuntimeError):
    """Raised when PDFPlumberReader cannot read/convert/process a PDF reliably."""


class PDFPlumberReader:
    """Extract structured content from PDFs using pdfplumber and return Markdown output."""

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        """Validate that file_path is a non-empty string."""
        if not isinstance(file_path, str) or not file_path.strip():
            raise PDFPlumberReaderException("file_path must be a non-empty string.")

    @staticmethod
    def _validate_tolerance(tolerance: float) -> None:
        """Validate that tolerance is a positive number."""
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise PDFPlumberReaderException("tolerance must be a positive number.")

    @staticmethod
    def _validate_resolution(resolution: int) -> None:
        """Validate that resolution is an int within a safe range [72, 600]."""
        if not isinstance(resolution, int) or resolution < 72 or resolution > 600:
            raise PDFPlumberReaderException(
                "resolution must be an int in the range [72, 600]."
            )

    @staticmethod
    def _validate_image_format(image_format: str) -> None:
        """Validate that image_format is a non-empty string."""
        if not isinstance(image_format, str) or not image_format.strip():
            raise PDFPlumberReaderException("image_format must be a non-empty string.")

    def group_by_lines(
        self, words: List[Dict[str, Any]], tolerance: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Group OCR word dictionaries into text lines using their vertical positions."""
        self._validate_tolerance(tolerance)
        if words is None:
            return []
        if not isinstance(words, list):
            raise PDFPlumberReaderException(
                "words must be a list of word dictionaries."
            )

        lines = defaultdict(list)
        for word in words:
            if not isinstance(word, dict):
                continue
            if any(k not in word for k in ("top", "text", "x0", "bottom")):
                continue
            try:
                top = round(word["top"] / tolerance) * tolerance
            except Exception:
                continue
            lines[top].append(word)

        sorted_lines: list[dict[str, Any]] = []
        for top in sorted(lines):
            sorted_words = sorted(lines[top], key=lambda w: w["x0"])
            if not sorted_words:
                continue
            try:
                line_text = " ".join([w["text"] for w in sorted_words])
                bottom = max(w["bottom"] for w in sorted_words)
            except Exception:
                continue
            sorted_lines.append(
                {"type": "text", "top": top, "bottom": bottom, "content": line_text}
            )
        return sorted_lines

    def is_real_table(self, table: List[List[Any]]) -> bool:
        """Return True if the extracted table looks meaningful based on simple heuristics."""
        if not table or len(table) < 2:
            return False
        col_counts = [len(row) for row in table if row]
        if not col_counts:
            return False
        if col_counts.count(1) > len(col_counts) * 0.7:
            return False
        if max(col_counts) < 2:
            return False
        return True

    def extract_tables(
        self, page, page_num: int
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float, float]]]:
        """Extract tables from a page and return both table blocks and their bounding boxes."""
        tables: list[dict[str, Any]] = []
        table_bboxes: list[Tuple[float, float, float, float]] = []
        try:
            found = page.find_tables()
        except Exception as e:
            logger.warning("Failed to find tables on page %s: %s", page_num, e)
            return tables, table_bboxes

        for table in found:
            try:
                bbox = table.bbox
                extracted = table.extract() or []
                cleaned = [
                    [cell if cell is not None else "" for cell in row]
                    for row in extracted
                    if row and any(cell not in (None, "", " ") for cell in row)
                ]
                if self.is_real_table(cleaned):
                    table_bboxes.append(bbox)
                    tables.append(
                        {
                            "type": "table",
                            "top": bbox[1],
                            "bottom": bbox[3],
                            "content": cleaned,
                            "page": page_num,
                        }
                    )
            except Exception as e:
                logger.warning("Failed to extract a table on page %s: %s", page_num, e)
                continue

        return tables, table_bboxes

    def extract_images(
        self,
        page,
        page_num: int,
        prompt: Optional[str] = None,
        model: Optional[BaseVisionModel] = None,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
    ) -> List[Dict[str, Any]]:
        """Extract images from a page, embedding them as base64 URIs and optionally annotating them."""
        images: list[dict[str, Any]] = []
        page_images = getattr(page, "images", None) or []
        for idx, img in enumerate(page_images):
            try:
                x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                bbox = (x0, top, x1, bottom)
                cropped = page.within_bbox(bbox).to_image(resolution=150)

                buf = BytesIO()
                cropped.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode()
                img_uri = f"data:image/png;base64,{img_b64}"

                image_description = image_placeholder
                if model:
                    try:
                        annotation = model.analyze_content(file=img_b64, prompt=prompt)
                        image_description += f"\n{annotation}"
                    except Exception as e:
                        logger.warning(
                            "Model annotation failed for image %s on page %s: %s",
                            idx,
                            page_num,
                            e,
                        )
                        image_description += f"\n**Annotation error:** {e}"

                images.append(
                    {
                        "type": "image",
                        "top": top,
                        "bottom": bottom,
                        "content": img_uri,
                        "annotation": image_description,
                        "page": page_num,
                    }
                )
            except Exception as e:
                logger.warning(
                    "Error extracting/encoding image %s on page %s: %s",
                    idx,
                    page_num,
                    e,
                )
                continue
        return images

    def analyze_content(
        self, page, page_num: int, table_bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Dict[str, Any]]:
        """Extract text lines from a page, excluding those that overlap detected table regions."""
        try:
            words = page.extract_words()
        except Exception as e:
            logger.warning("Failed to extract words on page %s: %s", page_num, e)
            return []

        lines = self.group_by_lines(words)
        texts: list[dict[str, Any]] = []
        for line in lines:
            try:
                mid_y = (line["top"] + line["bottom"]) / 2
                in_table = any(b[1] <= mid_y <= b[3] for b in table_bboxes)
                if not in_table:
                    line["page"] = page_num
                    texts.append(line)
            except Exception:
                continue
        return texts

    def extract_page_blocks(
        self,
        page,
        page_num: int,
        prompt: Optional[str] = None,
        model: Optional[BaseVisionModel] = None,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
    ) -> List[Dict[str, Any]]:
        """Extract tables, images, and text blocks from a single page and return them sorted by position."""
        tables, table_bboxes = self.extract_tables(page, page_num)
        images = self.extract_images(
            page,
            page_num=page_num,
            model=model,
            prompt=prompt,
            image_placeholder=image_placeholder,
        )
        texts = self.analyze_content(
            page=page, page_num=page_num, table_bboxes=table_bboxes
        )
        blocks = tables + images + texts
        return sorted(blocks, key=lambda x: x.get("top", 0))

    def extract_pages_as_images(
        self,
        file_path: str,
        resolution: int = 300,
        image_format: str = "PNG",
        return_base64: bool = True,
    ) -> List[str]:
        """Render each page of a PDF into an image and return either base64 strings or raw bytes."""
        self._validate_file_path(file_path)
        self._validate_resolution(resolution)
        self._validate_image_format(image_format)

        pages_data: list = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        page_img = page.to_image(resolution=resolution)
                        buf = BytesIO()
                        page_img.save(buf, format=image_format)
                        data_bytes = buf.getvalue()
                        pages_data.append(
                            base64.b64encode(data_bytes).decode()
                            if return_base64
                            else data_bytes
                        )
                    except Exception as e:
                        raise PDFPlumberReaderException(
                            f"Failed to rasterize page {i} from '{file_path}': {e}"
                        ) from e
        except PDFPlumberReaderException:
            raise
        except Exception as e:
            raise PDFPlumberReaderException(
                f"Failed to open/read PDF '{file_path}': {e}"
            ) from e

        return pages_data

    def table_to_markdown(self, table: List[List[Any]]) -> str:
        """Convert a 2D list table into GitHub-flavored Markdown."""
        if not table or not isinstance(table, list) or not table[0]:
            return ""
        try:
            max_cols = max(len(row) for row in table)
            padded = [row + [""] * (max_cols - len(row)) for row in table]
        except Exception:
            return ""

        header = (
            "| "  # noqa: W503
            + " | ".join(  # noqa: W503
                str(cell).strip().replace("\n", " ")
                for cell in padded[0]  # noqa: W503
            )  # noqa: W503
            + " |"  # noqa: W503
        )
        separator = "| " + " | ".join(["---"] * max_cols) + " |"

        fmt_row = (  # noqa: E731
            lambda r: "| "  # noqa: W503
            + " | ".join(str(cell).strip().replace("\n", " ") for cell in r)  # noqa: W503
            + " |"  # noqa: W503
        )

        rows = [fmt_row(row) for row in padded[1:]]
        return "\n".join([header, separator] + rows)

    def blocks_to_markdown(
        self,
        all_blocks: List[Dict[str, Any]],
        show_base64_images: bool = True,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
        page_placeholder: str = DEFAULT_PAGE_PLACEHOLDER,
    ) -> str:
        """Convert extracted blocks (text/images/tables) into a single Markdown document."""
        if all_blocks is None:
            return ""

        md_lines: list[str] = [""]
        try:
            all_blocks.sort(key=lambda x: (x.get("page", 0), x.get("top", 0)))
        except Exception:
            pass

        for page, blocks in groupby(all_blocks, key=lambda x: x.get("page", 0)):
            md_lines += [page_placeholder + "\n"]
            last_type: Optional[str] = None
            paragraph: list[str] = []

            for item in blocks:
                item_type = item.get("type")
                if item_type == "text":
                    if last_type not in (None, "text") and paragraph:
                        md_lines.append("\n".join(paragraph))
                        md_lines.append("")
                        paragraph = []
                    paragraph.append(str(item.get("content", "")))
                    last_type = "text"
                    continue

                if paragraph:
                    md_lines.append("\n".join(paragraph))
                    md_lines.append("")
                    paragraph = []

                if item_type == "image":
                    if show_base64_images and item.get("content"):
                        md_lines.append(
                            f"![Image page {item.get('page', '?')}]({item['content']})\n"
                        )
                    elif item.get("annotation"):
                        md_lines.append(f"{item['annotation']}\n")
                    else:
                        md_lines.append(f"\n{image_placeholder}\n")
                elif item_type == "table":
                    md_lines.append(self.table_to_markdown(item.get("content", [])))
                    md_lines.append("")

                last_type = item_type

            if paragraph:
                md_lines.append("\n".join(paragraph))
                md_lines.append("")

        clean_lines: list[str] = []
        for line in md_lines:
            if line != "" or (clean_lines and clean_lines[-1] != ""):
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def describe_pages(
        self,
        file_path: str,
        model: BaseVisionModel,
        prompt: Optional[str] = None,
        resolution: Optional[int] = 300,
        **parameters,
    ) -> List[str]:
        """Rasterize each page and ask a VLM to describe it, returning one description per page."""
        self._validate_file_path(file_path)
        if model is None:
            raise PDFPlumberReaderException("model is required for describe_pages().")
        if resolution is None:
            resolution: int = 300
        self._validate_resolution(resolution)

        page_images_b64: list[str] = self.extract_pages_as_images(
            file_path=file_path, resolution=resolution, return_base64=True
        )

        descriptions: list[str] = []
        for i, img_b64 in enumerate(page_images_b64, start=1):
            try:
                description = model.analyze_content(
                    file=img_b64, prompt=prompt, **parameters
                )
            except Exception as e:
                description = f"**Error on page {i}:** {e}"
            descriptions.append(description)
        return descriptions

    def read(
        self,
        file_path: str,
        prompt: Optional[str] = None,
        model: Optional[BaseVisionModel] = None,
        show_base64_images: bool = False,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
        page_placeholder: str = DEFAULT_PAGE_PLACEHOLDER,
    ) -> str:
        """
        Read a PDF file and return extracted content as Markdown.

        This method uses `pdfplumber` to iterate through each page in the PDF, extracting:
        - Text lines (grouped by vertical alignment and excluding lines overlapping tables)
        - Tables (rendered as GitHub-flavored Markdown tables)
        - Images (optionally embedded as base64 data URIs and optionally annotated via a VLM)

        The output is a single Markdown string, with `page_placeholder` inserted at the start
        of each page section.

        Args:
            file_path (str): Path to a local PDF file.
            prompt (Optional[str]): Prompt used for annotating images with `model`. If not provided,
                defaults to `DEFAULT_IMAGE_CAPTION_PROMPT`.
            model (Optional[BaseVisionModel]): Optional vision-capable model to annotate extracted
                images. If provided, each extracted image may include an annotation string.
            show_base64_images (bool): If True, embed extracted images directly in Markdown as
                `data:image/png;base64,...` URIs. If False, images are not embedded; instead:
                - if an annotation is present, the annotation text is emitted
                - otherwise `image_placeholder` is emitted
            image_placeholder (str): Placeholder inserted for images when images are omitted and no
                annotation is available. Defaults to `DEFAULT_IMAGE_PLACEHOLDER`.
            page_placeholder (str): Placeholder inserted at each page boundary in the Markdown output.
                Defaults to `DEFAULT_PAGE_PLACEHOLDER`.

        Returns:
            str: Markdown document representing the extracted PDF content. The Markdown contains
            page separators, paragraphs of text, Markdown tables, and optionally embedded images.

        Raises:
            PDFPlumberReaderException: If `file_path` is invalid, the PDF cannot be opened/read,
                or extraction fails for a page in a way that prevents completing the output.

        Warns:
            UserWarning (via logging): Individual non-fatal extraction issues are logged (e.g.,
                failure to extract tables on a page, failure to encode a single image, or failure
                to annotate an image). These warnings do not necessarily abort extraction.

        Notes:
            - This reader is designed to be resilient: table and image extraction failures are
              treated as non-fatal and logged, while critical failures (unable to open the PDF,
              unrecoverable page processing errors) raise an exception.
            - Image extraction depends on `pdfplumber`'s ability to detect images and may not work
              for all PDFs.
        """
        self._validate_file_path(file_path)

        all_blocks: list[Dict[str, Any]] = []
        prompt: str = prompt or DEFAULT_IMAGE_CAPTION_PROMPT

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        all_blocks.extend(
                            self.extract_page_blocks(
                                page=page,
                                page_num=i,
                                model=model,
                                prompt=prompt,
                                image_placeholder=image_placeholder,
                            )
                        )
                    except Exception as e:
                        raise PDFPlumberReaderException(
                            f"Failed extracting content blocks on page {i} of '{file_path}': {e}"
                        ) from e
        except PDFPlumberReaderException:
            raise
        except Exception as e:
            raise PDFPlumberReaderException(
                f"Failed to open/read PDF '{file_path}': {e}"
            ) from e

        markdown_text = self.blocks_to_markdown(
            all_blocks,
            show_base64_images=show_base64_images,
            image_placeholder=image_placeholder,
            page_placeholder=page_placeholder,
        )
        return markdown_text
