# Warnings

SplitterMR uses standard Python `UserWarning` subclasses to alert users about suspicious inputs, ambiguous file types, or heuristic fallbacks that do not halt execution but may affect data quality.

Warnings live in:

```python
from splitter_mr.schema.warnings import *
```

## Why a custom warning hierarchy?

* **Granular filtering**: You can ignore specific categories (e.g., `FiletypeAmbiguityWarning`) while keeping others active using Python's `warnings` filter.
* **Process observability**: Distinguishes between input data issues (`SplitterInputWarning`) and processing results (`SplitterOutputWarning`).

## Reader warnings

Readers emit `BaseReaderWarning` (or subclasses) when input files are ambiguous or minor issues occur during ingestion.

Hierarchy

```sh
`BaseReaderWarning` (`UserWarning`)
└── `FiletypeAmbiguityWarning`
```

### General

#### BaseReaderWarning

::: src.splitter_mr.schema.warnings.BaseReaderWarning
    handler: python
    options:
        members_order: source

### I/O and Heuristics

#### FiletypeAmbiguityWarning

::: src.splitter_mr.schema.warnings.FiletypeAmbiguityWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* A file has a `.json` extension but contains HTML content.
* MIME type sniffing disagrees with the provided file extension.

## Splitter warnings

Splitters emit `BaseSplitterWarning` (or subclasses) regarding suspicious chunking inputs, outputs, or fallback behaviors.

Hierarchy

```sh
`BaseSplitterWarning` (`UserWarning`)
├── `SplitterInputWarning`
│   ├── `AutoTagFallbackWarning`
│   └── `BatchHtmlTableWarning`
└── `SplitterOutputWarning`
    ├── `ChunkUnderflowWarning`
    └── `ChunkOverflowWarning`

```

### General

#### BaseSplitterWarning

::: src.splitter_mr.schema.warnings.BaseSplitterWarning
handler: python
options:
members_order: source

### I/O and Validation

#### SplitterInputWarning

::: src.splitter_mr.schema.warnings.SplitterInputWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* Input text is empty.
* Input text is expected to be JSON but parsing failed (fallback to raw text).

#### SplitterOutputWarning

::: src.splitter_mr.schema.warnings.SplitterOutputWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* The resulting chunks contain empty text fields.
* Metadata generation produced suspicious values.

#### ChunkUnderflowWarning

::: src.splitter_mr.schema.warnings.ChunkUnderflowWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* The document structure resulted in significantly fewer chunks than the `chunk_size` configuration suggested.

#### ChunkOverflowWarning

::: src.splitter_mr.schema.warnings.ChunkOverflowWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* Chunking produced unexpected volume or size deviations based on paragraph constraints.

### Splitters-specific warnings

#### HtmlTagSplitter

**`AutoTagFallbackWarning`**

::: src.splitter_mr.schema.warnings.AutoTagFallbackWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* The specific tag requested was not found, triggering an auto-tagging strategy.

**`BatchHtmlTableWarning`**

::: src.splitter_mr.schema.warnings.BatchHtmlTableWarning
    handler: python
    options:
        members_order: source

**Typical cases**:

* A target tag is located inside an HTML table during batch processing (split occurs by table to preserve context).

## Reference table

| **Area** | **Warning** | **Parent** | **Description** |
| --- | --- | --- | --- |
| Reader | `BaseReaderWarning` | `UserWarning` | Base reader warning |
| Reader | `FiletypeAmbiguityWarning` | `BaseReaderWarning` | Extension vs. content mismatch |
| Splitter | `BaseSplitterWarning` | `UserWarning` | Base splitter warning |
| Splitter | `SplitterInputWarning` | `BaseSplitterWarning` | Suspicious input (empty/malformed) |
| Splitter | `SplitterOutputWarning` | `BaseSplitterWarning` | Suspicious output elements |
| Splitter | `ChunkUnderflowWarning` | `SplitterOutputWarning` | Fewer chunks than expected |
| Splitter | `ChunkOverflowWarning` | `SplitterOutputWarning` | Chunks deviation/overflow |
| HTML Splitter | `AutoTagFallbackWarning` | `SplitterInputWarning` | Tag not found; auto-tagging used |
| HTML Splitter | `BatchHtmlTableWarning` | `SplitterInputWarning` | Tag inside table (batch mode) |

!!! note

    More warnings will be introduced soon. Stay aware to updates!