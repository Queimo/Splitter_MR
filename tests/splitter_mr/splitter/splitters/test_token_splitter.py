import warnings
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from splitter_mr.schema import ReaderOutput
from splitter_mr.schema.exceptions import InvalidChunkException, SplitterConfigException
from splitter_mr.schema.warnings import ChunkUnderflowWarning, SplitterInputWarning
from splitter_mr.splitter.splitters import TokenSplitter

# ---- Mocks, fixtures and helpers ---- #


@pytest.fixture
def simple_reader_output():
    return ReaderOutput(
        text=(
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs."
        ),
        document_name="pangrams.txt",
        document_path=(
            "https://raw.githubusercontent.com/andreshere00/"
            "Splitter_MR/refs/heads/main/data/pangrams.txt"
        ),
        document_id="doc1",
        conversion_method="text",
        reader_method="plain",
        ocr_method=None,
        metadata={},
    )


# ---- Test cases ---- #


def test_split_tiktoken(monkeypatch, simple_reader_output):
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = [
        "The quick brown fox jumps",
        "over the lazy dog. Pack my",
        "box with five dozen liquor jugs.",
    ]
    with patch(
        "splitter_mr.splitter.splitters.token_splitter.RecursiveCharacterTextSplitter"
    ) as mock_class:
        mock_class.from_tiktoken_encoder.return_value = mock_splitter
        splitter = TokenSplitter(chunk_size=5, model_name="tiktoken/cl100k_base")
        output = splitter.split(simple_reader_output)
        assert output.chunks == mock_splitter.split_text.return_value
        assert output.split_method == "token_splitter"
        assert output.split_params["model_name"] == "tiktoken/cl100k_base"


def test_split_spacy(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = True
        mock_spacy.load.return_value = None
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
        ]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.SpacyTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="spacy/en_core_web_sm")
            output = splitter.split(simple_reader_output)
            assert output.chunks == mock_splitter.split_text.return_value


def test_split_nltk(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.nltk") as mock_nltk:
        mock_nltk.data.find.side_effect = None
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
        ]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.NLTKTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="nltk/punkt")
            output = splitter.split(simple_reader_output)
            assert output.chunks == mock_splitter.split_text.return_value


def test_split_invalid_tokenizer(simple_reader_output):
    splitter = TokenSplitter(chunk_size=10, model_name="unknown/foobar")
    with pytest.raises(SplitterConfigException) as exc_info:
        splitter.split(simple_reader_output)
    msg = str(exc_info.value)
    assert "Unsupported tokenizer 'unknown'" in msg
    assert "Supported tokenizers:" in msg


def test_spacy_model_download_if_not_present(simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = False
        mock_spacy.cli.download.return_value = None
        mock_spacy.load.return_value = None

        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.SpacyTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="spacy/en_core_web_sm")
            output = splitter.split(simple_reader_output)
            assert output.chunks == ["chunk1", "chunk2"]

        mock_spacy.cli.download.assert_called_once_with("en_core_web_sm")


def test_split_tiktoken_raises_on_missing_encoding(monkeypatch, simple_reader_output):
    class DummyTiktoken:
        @staticmethod
        def list_encoding_names():
            return ["cl100k_base"]

    monkeypatch.setattr(
        "splitter_mr.splitter.splitters.token_splitter.tiktoken", DummyTiktoken
    )
    splitter = TokenSplitter(chunk_size=10, model_name="tiktoken/foobar")
    with pytest.raises(SplitterConfigException) as e:
        splitter.split(simple_reader_output)
    assert "tiktoken encoding 'foobar' is not available" in str(e.value)


def test_split_spacy_download_fails(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = False
        mock_spacy.cli.download.side_effect = Exception("fail download")

        with pytest.raises(SplitterConfigException) as exc_info:
            TokenSplitter(chunk_size=10, model_name="spacy/en_core_web_sm").split(
                simple_reader_output
            )

    msg = str(exc_info.value)
    assert "spaCy model 'en_core_web_sm' is not available for download" in msg
    assert "Common models include:" in msg


def test_split_spacy_warns_on_huge_chunk_size(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = True
        mock_spacy.load.return_value = None

        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["ok"]

        with patch(
            "splitter_mr.splitter.splitters.token_splitter.SpacyTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(
                chunk_size=1_500_000, model_name="spacy/en_core_web_sm"
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                splitter.split(simple_reader_output)

                # Check message and warning type
                assert any(
                    issubclass(warn.category, SplitterInputWarning)
                    and "Configured chunk_size is very large" in str(warn.message)
                    for warn in w
                )


def test_split_nltk_downloads_if_missing(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.nltk") as mock_nltk:
        mock_nltk.data.find.side_effect = LookupError()
        mock_nltk.download.return_value = True
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["a", "b"]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.NLTKTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=2, model_name="nltk/punkt")
            output = splitter.split(simple_reader_output)
            assert output.chunks == ["a", "b"]
        mock_nltk.download.assert_called_once_with("punkt_tab")


def test_split_nltk_download_failure(monkeypatch, simple_reader_output):
    with patch("splitter_mr.splitter.splitters.token_splitter.nltk") as mock_nltk:
        mock_nltk.data.find.side_effect = LookupError()
        mock_nltk.download.side_effect = Exception("boom")

        splitter = TokenSplitter(chunk_size=2, model_name="nltk/punkt")
        with pytest.raises(SplitterConfigException) as exc_info:
            splitter.split(simple_reader_output)

    assert "NLTK Punkt data could not be downloaded" in str(exc_info.value)


def test_list_nltk_punkt_languages(monkeypatch, tmp_path):
    punkt_dir = tmp_path / "tokenizers" / "punkt"
    punkt_dir.mkdir(parents=True)
    (punkt_dir / "english.pickle").touch()
    (punkt_dir / "german.pickle").touch()
    monkeypatch.setattr("nltk.data.path", [str(tmp_path)])
    langs = TokenSplitter.list_nltk_punkt_languages()
    assert set(langs) == {"english", "german"}


def test_splitter_metadata_passthrough(monkeypatch, simple_reader_output):
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = ["a"]
    with patch(
        "splitter_mr.splitter.splitters.token_splitter.RecursiveCharacterTextSplitter"
    ) as mock_class:
        mock_class.from_tiktoken_encoder.return_value = mock_splitter
        splitter = TokenSplitter(chunk_size=5, model_name="tiktoken/cl100k_base")
        output = splitter.split(simple_reader_output)
        assert output.document_name == simple_reader_output.document_name
        assert output.document_path == simple_reader_output.document_path
        assert output.document_id == simple_reader_output.document_id
        assert output.conversion_method == simple_reader_output.conversion_method
        assert output.reader_method == simple_reader_output.reader_method
        assert output.ocr_method == simple_reader_output.ocr_method
        assert isinstance(output.metadata, dict)
        assert output.split_method == "token_splitter"
        assert output.split_params["model_name"] == "tiktoken/cl100k_base"
        assert output.split_params["chunk_size"] == 5


def test_split_invalid_model_name_format(simple_reader_output):
    splitter = TokenSplitter(chunk_size=10, model_name="badformat")
    with pytest.raises(SplitterConfigException) as exc_info:
        splitter.split(simple_reader_output)
    assert "model_name must be in the format 'tokenizer/model'" in str(exc_info.value)


def test_token_splitter_invalid_chunk_size(simple_reader_output):
    with pytest.raises(SplitterConfigException) as exc_info:
        TokenSplitter(chunk_size=0, model_name="tiktoken/cl100k_base").split(
            simple_reader_output
        )
    assert "chunk_size must be a positive integer" in str(exc_info.value)


def test_split_raises_invalid_chunk_when_splitter_returns_none(
    monkeypatch, simple_reader_output
):
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = None
    with patch(
        "splitter_mr.splitter.splitters.token_splitter.RecursiveCharacterTextSplitter"
    ) as mock_class:
        mock_class.from_tiktoken_encoder.return_value = mock_splitter
        splitter = TokenSplitter(chunk_size=5, model_name="tiktoken/cl100k_base")
        with pytest.raises(InvalidChunkException):
            splitter.split(simple_reader_output)


def test_split_empty_text_warns(monkeypatch):
    empty_ro = ReaderOutput(
        text="   ",
        document_name="empty.txt",
        document_path="/tmp/empty.txt",
        document_id="empty",
        conversion_method="text",
        reader_method="plain",
        ocr_method=None,
        metadata={},
    )

    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = []

    with patch(
        "splitter_mr.splitter.splitters.token_splitter.RecursiveCharacterTextSplitter"
    ) as mock_class:
        mock_class.from_tiktoken_encoder.return_value = mock_splitter
        splitter = TokenSplitter(chunk_size=10, model_name="tiktoken/cl100k_base")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ValidationError):
                splitter.split(empty_ro)

        categories = {warn.category for warn in w}
        assert any(issubclass(cat, SplitterInputWarning) for cat in categories), (
            "Expected SplitterInputWarning for empty text"
        )
        assert any(issubclass(cat, ChunkUnderflowWarning) for cat in categories), (
            "Expected ChunkUnderflowWarning when no chunks are produced"
        )
