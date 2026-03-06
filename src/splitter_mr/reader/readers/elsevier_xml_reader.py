import os
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ...schema import ReaderOutput
from ...schema.exceptions import ReaderConfigException, VanillaReaderException
from .vanilla_reader import VanillaReader


class ElsevierXmlReader(VanillaReader):
    """Parse Elsevier full-text XML into clean Markdown.

    This reader subclasses :class:`VanillaReader` and specializes XML handling for
    Elsevier API responses (e.g., files rooted at ``full-text-retrieval-response``).
    Non-XML inputs are delegated to ``VanillaReader``.
    """

    def read(self, file_path: str | Path = None, **kwargs: Any) -> ReaderOutput:
        if file_path is None:
            raise ReaderConfigException("file_path must be provided.")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise VanillaReaderException(f"File not found: {file_path}")

        ext = file_path_obj.suffix.lower().lstrip(".")
        if ext != "xml":
            return super().read(file_path=file_path, **kwargs)

        try:
            xml_text = file_path_obj.read_text(encoding="utf-8")
            markdown_text = self._xml_to_markdown(xml_text)
        except ET.ParseError as exc:
            raise VanillaReaderException(f"Invalid XML file {file_path}: {exc}") from exc
        except OSError as exc:
            raise VanillaReaderException(f"Could not read XML file {file_path}: {exc}") from exc

        return ReaderOutput(
            text=markdown_text,
            document_name=file_path_obj.name,
            document_path=os.fspath(file_path_obj),
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method="md",
            reader_method="elsevier_xml",
            ocr_method=None,
            page_placeholder=None,
            metadata=kwargs.get("metadata", {}),
        )

    def _xml_to_markdown(self, xml_text: str) -> str:
        root = ET.fromstring(xml_text)

        sections: list[str] = []
        title = self._first_non_empty(
            self._find_text(root, ".//{*}coredata/{*}title"),
            self._find_text(root, ".//{*}title"),
        )
        if title:
            sections.append(f"# {title}\n")

        abstract_paras = [
            self._clean_text("".join(para.itertext()))
            for para in root.findall(".//{*}abstract//{*}para")
        ]
        abstract_paras = [text for text in abstract_paras if text]
        if abstract_paras:
            sections.append("## Abstract\n")
            sections.extend(f"{paragraph}\n" for paragraph in abstract_paras)

        keywords = [
            self._clean_text("".join(node.itertext()))
            for node in root.findall(".//{*}authkeywords//{*}author-keyword")
        ]
        keywords = [kw for kw in keywords if kw]
        if keywords:
            sections.append("## Keywords\n")
            sections.append(", ".join(dict.fromkeys(keywords)) + "\n")

        body_nodes = root.findall(".//{*}body//{*}section")
        if not body_nodes:
            body_nodes = root.findall(".//{*}sections/{*}section")
        if body_nodes:
            sections.append("## Body\n")
            for node in body_nodes:
                md_section = self._section_to_markdown(node)
                if md_section:
                    sections.append(md_section)

        table_blocks: list[str] = []
        seen_table_blocks: set[str] = set()
        for table_node in root.findall(".//{*}table"):
            table_markdown = self._table_to_markdown(table_node)
            if table_markdown and table_markdown not in seen_table_blocks:
                seen_table_blocks.add(table_markdown)
                table_blocks.append(table_markdown)
        if table_blocks:
            sections.append("## Tables\n")
            sections.extend(f"{table}\n" for table in table_blocks)

        references = [
            self._clean_text("".join(node.itertext()))
            for node in root.findall(".//{*}bibliography//{*}reference")
        ]
        references = [ref for ref in references if ref]
        if references:
            sections.append("## References\n")
            sections.extend(f"- {ref}\n" for ref in references)

        if not sections:
            return self._clean_text(" ".join(root.itertext()))

        return "\n".join(sections).strip()

    def _section_to_markdown(self, section_node: ET.Element, level: int = 3) -> str:
        chunks: list[str] = []

        title = self._first_non_empty(
            self._find_text(section_node, "./{*}title"),
            self._find_text(section_node, "./{*}section-title"),
            self._find_text(section_node, "./{*}label"),
        )
        if title:
            chunks.append(f"{'#' * min(level, 6)} {title}\n")

        for para in section_node.findall("./{*}para"):
            paragraph = self._clean_text("".join(para.itertext()))
            if paragraph:
                chunks.append(f"{paragraph}\n")

        for list_item in section_node.findall(".//{*}list-item"):
            item_text = self._clean_text("".join(list_item.itertext()))
            if item_text:
                chunks.append(f"- {item_text}\n")

        for child_section in section_node.findall("./{*}section"):
            nested = self._section_to_markdown(child_section, level=level + 1)
            if nested:
                chunks.append(nested)

        return "\n".join(chunks).strip()

    def _table_to_markdown(self, table_node: ET.Element) -> str:
        chunks: list[str] = []

        table_label = self._first_non_empty(
            self._find_text(table_node, "./{*}label"),
            self._find_text(table_node, "./{*}alt-text"),
        )
        table_caption = self._first_non_empty(
            self._find_text(table_node, "./{*}caption/{*}simple-para"),
            self._find_text(table_node, "./{*}caption"),
        )

        heading_parts = [part for part in (table_label, table_caption) if part]
        if heading_parts:
            chunks.append(f"**{' — '.join(heading_parts)}**\n")

        rows: list[list[str]] = []
        header_row = table_node.find(".//{*}thead/{*}row")
        if header_row is not None:
            header_cells = [
                self._clean_text("".join(entry.itertext()))
                for entry in header_row.findall("./{*}entry")
            ]
            header_cells = [cell if cell else " " for cell in header_cells]
            if header_cells:
                rows.append(header_cells)

        for body_row in table_node.findall(".//{*}tbody/{*}row"):
            cells = [
                self._clean_text("".join(entry.itertext()))
                for entry in body_row.findall("./{*}entry")
            ]
            if any(cells):
                rows.append([cell if cell else " " for cell in cells])

        if not rows:
            return "\n".join(chunks).strip()

        col_count = max(len(row) for row in rows)
        normalized_rows = [row + [" "] * (col_count - len(row)) for row in rows]

        if len(normalized_rows) == 1:
            header = [f"Column {idx + 1}" for idx in range(col_count)]
            normalized_rows.insert(0, header)

        header_cells = normalized_rows[0]
        chunks.append("| " + " | ".join(header_cells) + " |")
        chunks.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in normalized_rows[1:]:
            chunks.append("| " + " | ".join(row) + " |")

        return "\n".join(chunks).strip()

    @staticmethod
    def _find_text(node: ET.Element, xpath: str) -> str:
        found = node.find(xpath)
        if found is None:
            return ""
        return ElsevierXmlReader._clean_text("".join(found.itertext()))

    @staticmethod
    def _first_non_empty(*values: str) -> str:
        for value in values:
            if value:
                return value
        return ""

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()
