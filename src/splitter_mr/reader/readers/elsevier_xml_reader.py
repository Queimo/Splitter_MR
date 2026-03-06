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

    def __init__(self, place_tables_near_mentions: bool = True) -> None:
        super().__init__()
        self.place_tables_near_mentions = place_tables_near_mentions

    def read(self, file_path: str | Path = None, **kwargs: Any) -> ReaderOutput:
        if file_path is None:
            raise ReaderConfigException("file_path must be provided.")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise VanillaReaderException(f"File not found: {file_path}")

        ext = file_path_obj.suffix.lower().lstrip(".")
        if ext != "xml":
            return super().read(file_path=file_path, **kwargs)

        place_near_mentions = kwargs.get(
            "place_tables_near_mentions", self.place_tables_near_mentions
        )

        try:
            xml_text = file_path_obj.read_text(encoding="utf-8")
            markdown_text = self._xml_to_markdown(
                xml_text, place_tables_near_mentions=bool(place_near_mentions)
            )
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

    def _xml_to_markdown(
        self, xml_text: str, *, place_tables_near_mentions: bool = True
    ) -> str:
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

        table_blocks_by_label, table_order = self._collect_tables(root)
        placed_tables: set[str] = set()

        body_nodes = root.findall(".//{*}body//{*}section")
        if not body_nodes:
            body_nodes = root.findall(".//{*}sections/{*}section")
        if body_nodes:
            sections.append("## Body\n")
            for node in body_nodes:
                md_section = self._section_to_markdown(
                    node,
                    table_blocks_by_label=table_blocks_by_label,
                    placed_tables=placed_tables,
                    place_tables_near_mentions=place_tables_near_mentions,
                )
                if md_section:
                    sections.append(md_section)

        unplaced_table_blocks = [
            table_blocks_by_label[label]
            for label in table_order
            if label not in placed_tables and label in table_blocks_by_label
        ]
        if unplaced_table_blocks:
            sections.append("## Tables\n")
            sections.extend(f"{table}\n" for table in unplaced_table_blocks)

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

    def _collect_tables(self, root: ET.Element) -> tuple[dict[str, str], list[str]]:
        table_blocks_by_label: dict[str, str] = {}
        table_order: list[str] = []

        for idx, table_node in enumerate(root.findall(".//{*}table"), start=1):
            table_markdown = self._table_to_markdown(table_node)
            if not table_markdown:
                continue
            label = self._first_non_empty(
                self._find_text(table_node, "./{*}label"),
                self._find_text(table_node, "./{*}alt-text"),
                f"Table {idx}",
            )
            if label not in table_blocks_by_label:
                table_blocks_by_label[label] = table_markdown
                table_order.append(label)

        return table_blocks_by_label, table_order

    def _section_to_markdown(
        self,
        section_node: ET.Element,
        level: int = 3,
        table_blocks_by_label: dict[str, str] | None = None,
        placed_tables: set[str] | None = None,
        place_tables_near_mentions: bool = True,
    ) -> str:
        chunks: list[str] = []
        table_blocks_by_label = table_blocks_by_label or {}
        placed_tables = placed_tables if placed_tables is not None else set()

        title = self._first_non_empty(
            self._find_text(section_node, "./{*}title"),
            self._find_text(section_node, "./{*}section-title"),
            self._find_text(section_node, "./{*}label"),
        )
        if title:
            chunks.append(f"{'#' * min(level, 6)} {title}\n")

        for para in section_node.findall("./{*}para"):
            paragraph = self._clean_text("".join(para.itertext()))
            if not paragraph:
                continue

            if place_tables_near_mentions and table_blocks_by_label:
                for table_label, table_markdown in table_blocks_by_label.items():
                    if table_label in placed_tables:
                        continue
                    if re.search(rf"\b{re.escape(table_label)}\b", paragraph, re.IGNORECASE):
                        chunks.append(f"{table_markdown}\n")
                        placed_tables.add(table_label)

            chunks.append(f"{paragraph}\n")

        for list_item in section_node.findall(".//{*}list-item"):
            item_text = self._clean_text("".join(list_item.itertext()))
            if item_text:
                chunks.append(f"- {item_text}\n")

        for child_section in section_node.findall("./{*}section"):
            nested = self._section_to_markdown(
                child_section,
                level=level + 1,
                table_blocks_by_label=table_blocks_by_label,
                placed_tables=placed_tables,
                place_tables_near_mentions=place_tables_near_mentions,
            )
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
