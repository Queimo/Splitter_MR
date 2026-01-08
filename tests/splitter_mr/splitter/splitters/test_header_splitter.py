from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.schema.exceptions import (
    HeaderLevelOutOfRangeError,
    HtmlConversionError,
    InvalidHeaderNameError,
    NormalizationError,
)
from splitter_mr.schema.warnings import FiletypeAmbiguityWarning, SplitterInputWarning
from splitter_mr.splitter import HeaderSplitter

# ---- Helpers, mocks and fixtures ---- #

SAMPLE_MD: str = """# Sample Markdown

This is some basic, sample markdown.

## Second Heading

- Unordered lists, and:
  1. One
  1. Two
  1. Three
- More
> Blockquote

And **bold**, *italics*, and even *italics and later **bold***. Even ~~strikethrough~~. [A link](https://markdowntohtml.com) to somewhere.

And code highlighting:

```js
var foo = 'bar';

function baz(s) {
   return foo + ':' + s;
}
````

Or inline code like `var foo = 'bar';`.

Or an image of bears

![bears](http://placebear.com/200/200)

The end ...
"""

SAMPLE_HTML: str = """

<h1 id="sample-markdown">Sample Markdown</h1>
<p>This is some basic, sample markdown.</p>
<h2 id="second-heading">Second Heading</h2>
<ul>
<li>Unordered lists, and:<ol>
<li>One</li>
<li>Two</li>
<li>Three</li>
</ol>
</li>
<li>More</li>
</ul>
<blockquote>
<p>Blockquote</p>
</blockquote>
<p>And <strong>bold</strong>, <em>italics</em>, and even <em>italics and later <strong>bold</strong></em>. Even <del>strikethrough</del>. <a href="https://markdowntohtml.com">A link</a> to somewhere.</p>
<p>And code highlighting:</p>
<pre><code class="lang-js">var foo = 'bar';

function baz(s) {
return foo + ':' + s;
} </code></pre>

<p>Or inline code like <code>var foo = &#39;bar&#39;;</code>.</p>
<p>Or an image of bears</p>
<p><img src="http://placebear.com/200/200" alt="bears"></p>
<p>The end ...</p>
"""


@pytest.fixture
def markdown_reader_output():
    return ReaderOutput(
        text=SAMPLE_MD,
        document_name="doc.md",
        document_path="/tmp/doc.md",
        document_id="42",
        conversion_method=None,
        ocr_method=None,
    )


@pytest.fixture
def html_reader_output():
    return ReaderOutput(
        text=SAMPLE_HTML,
        document_name="doc.html",
        document_path="/tmp/doc.html",
        document_id="99",
        conversion_method="html",
        ocr_method=None,
    )


# ---- Test cases ---- #


def test_markdown_splitter_on_md_content(markdown_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [
            MagicMock(page_content="Chunk 1"),
            MagicMock(page_content="Chunk 2"),
        ]
        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
        result = splitter.split(markdown_reader_output)
        MockMD.assert_called_once_with(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
            return_each_line=False,
            strip_headers=False,
        )
        mock_md.split_text.assert_called_once_with(SAMPLE_MD)
        assert result.chunks == ["Chunk 1", "Chunk 2"]
        assert result.split_method == "header_splitter"


def test_html_conversion_and_split(monkeypatch, html_reader_output):
    # Patch MarkdownHeaderTextSplitter
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [
            MagicMock(page_content="Converted chunk 1"),
            MagicMock(page_content="Converted chunk 2"),
        ]
        # Patch HtmlToMarkdown WHERE IT IS USED
        with patch(
            "splitter_mr.splitter.splitters.header_splitter.HtmlToMarkdown"
        ) as MockHtmlToMd:
            mock_converter = MockHtmlToMd.return_value
            mock_converter.convert.return_value = SAMPLE_MD
            splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
            result = splitter.split(html_reader_output)
            mock_converter.convert.assert_called_once_with(SAMPLE_HTML)
            mock_md.split_text.assert_called_once_with(SAMPLE_MD)
            assert result.chunks == ["Converted chunk 1", "Converted chunk 2"]
            assert result.split_method == "header_splitter"


def test_value_error_on_bad_semantic_header(markdown_reader_output):
    with pytest.raises(InvalidHeaderNameError):
        HeaderSplitter(headers_to_split_on=["NOPE", "Header 2"])


def test_output_metadata_fields(markdown_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [MagicMock(page_content="chunk")]
        splitter = HeaderSplitter()
        result = splitter.split(markdown_reader_output)

        for field in [
            "chunks",
            "chunk_id",
            "document_name",
            "document_path",
            "document_id",
            "conversion_method",
            "ocr_method",
            "split_method",
            "split_params",
            "metadata",
        ]:
            assert hasattr(result, field)


def test_empty_text_warns_and_returns_single_empty_chunk():
    ro = ReaderOutput(
        text="",
        document_name="empty.md",
        document_path="/tmp/empty.md",
        document_id="1",
    )
    splitter = HeaderSplitter()
    with pytest.warns(SplitterInputWarning, match="empty or whitespace-only"):
        out = splitter.split(ro)
    assert out.chunks == [""]
    assert out.split_method == "header_splitter"
    assert len(out.chunk_id) == 1


def test_guess_filetype_logic():
    # .md extension
    ro = ReaderOutput(
        text="# test", document_name="foo.md", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "md"
    # .html extension
    ro = ReaderOutput(
        text="<h1>x</h1>", document_name="foo.html", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "html"
    # HTML content, but not extension
    ro = ReaderOutput(
        text="<h1>x</h1>", document_name="foo.txt", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "html"
    # Markdown by default
    ro = ReaderOutput(
        text="# X", document_name="foo.txt", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "md"


# ---- Header-level validation errors ---- #


def test_header_level_invalid_name_raises_invalid_header_name_error():
    # Call the static method directly to isolate behavior
    with pytest.raises(InvalidHeaderNameError, match="Expected 'Header N'"):
        HeaderSplitter._header_level("H2")  # wrong pattern
    with pytest.raises(InvalidHeaderNameError):
        HeaderSplitter._header_level("##")  # wrong pattern


def test_header_level_out_of_range_raises_header_level_out_of_range_error():
    with pytest.raises(HeaderLevelOutOfRangeError, match="out of range"):
        HeaderSplitter._header_level("Header 0")
    with pytest.raises(HeaderLevelOutOfRangeError, match="out of range"):
        HeaderSplitter._header_level("Header 8")


# ---- HTMLConversionError ---- #


def test_html_conversion_error_bubbles_as_custom_exception(html_reader_output):
    # Force HtmlToMarkdown.convert to throw
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.HtmlToMarkdown"
    ) as MockHtmlToMd:
        mock_converter = MockHtmlToMd.return_value
        mock_converter.convert.side_effect = RuntimeError("boom")

        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
        with pytest.raises(HtmlConversionError, match="HTML→Markdown failed"):
            splitter.split(html_reader_output)


# ---- FiletypeAmbiguityWarning when ext and DOM disagree ---- #


def test_filetype_ambiguity_emits_warning_and_uses_dom_hint():
    # .html extension but markdown body → expect warning and proceed as md
    ro = ReaderOutput(
        text="# Title\n\nBody",
        document_name="looks-like-html.html",
        document_path="/tmp/looks-like-html.html",
        document_id="ab",
    )
    with pytest.warns(FiletypeAmbiguityWarning, match="heuristics disagree"):
        # We also patch the header splitter to ensure it’s called with MD text unchanged
        with patch(
            "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
        ) as MockMD:
            mock_md = MockMD.return_value
            mock_md.split_text.return_value = [MagicMock(page_content="chunk")]
            splitter = HeaderSplitter(headers_to_split_on=["Header 1"])
            out = splitter.split(ro)
            # Because DOM said "md", we should not have attempted HTML conversion
            mock_md.split_text.assert_called_once_with(ro.text)
            assert out.chunks == ["chunk"]


# ---- Setext Normalization Error handling ---- #


def test_normalization_error_when_setext_remains_inside_code_fence():
    # The normalization protects fenced blocks, restores them, then checks for remaining setext.
    # Setext inside the code fence remains → triggers NormalizationError per your rule.
    md_with_setext_in_code = """\
Before
```
Title
=====
```
After
"""
    ro = ReaderOutput(
        text=md_with_setext_in_code,
        document_name="code_setext.md",
        document_path="/tmp/code_setext.md",
        document_id="77",
    )
    splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
    with pytest.raises(NormalizationError, match="Unnormalized Setext"):
        splitter.split(ro)


def test_no_normalization_error_for_inline_code_without_setext():
    ro = ReaderOutput(
        text="Before\n`Title`\nAfter\n",
        document_name="inline.md",
        document_path="/tmp/inline.md",
        document_id="88",
    )
    splitter = HeaderSplitter()
    # should not raise; no setext pattern present
    splitter.split(ro)


def test_setext_outside_fence_is_normalized_and_does_not_raise():
    md_with_setext = "Heading\n----\n\nBody\n"
    ro = ReaderOutput(
        text=md_with_setext,
        document_name="setext.md",
        document_path="/tmp/setext.md",
        document_id="89",
    )
    splitter = HeaderSplitter()
    # normalization should convert and not raise
    splitter.split(ro)
