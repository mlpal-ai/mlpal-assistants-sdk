"""Tests for File class."""

import base64
import tempfile
from pathlib import Path

import pytest

from mlpal_assistants import File


class TestFileFromPath:
    """Tests for File initialized from path."""

    def test_image_from_path(self, tmp_path: Path):
        """Test creating file from image path."""
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image data")

        f = File(str(img_path))

        assert f.type == "image"
        assert f.mime_type == "image/jpeg"
        assert not f.is_url
        api_format = f.to_api_format()
        assert api_format["type"] == "image"
        assert api_format["mime_type"] == "image/jpeg"
        assert api_format["base64"] == base64.b64encode(b"fake image data").decode()

    def test_pdf_from_path(self, tmp_path: Path):
        """Test creating file from PDF path."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf data")

        f = File(pdf_path)

        assert f.type == "document"
        assert f.mime_type == "application/pdf"

    def test_audio_from_path(self, tmp_path: Path):
        """Test creating file from audio path."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio data")

        f = File(audio_path)

        assert f.type == "audio"
        assert f.mime_type == "audio/mpeg"

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            File("/nonexistent/path.jpg")


class TestFileFromBytes:
    """Tests for File initialized from bytes."""

    def test_image_from_bytes(self):
        """Test creating file from image bytes."""
        data = b"fake image data"
        f = File(data, mime_type="image/png")

        assert f.type == "image"
        assert f.mime_type == "image/png"
        api_format = f.to_api_format()
        assert api_format["base64"] == base64.b64encode(data).decode()

    def test_bytes_requires_type_info(self):
        """Test that bytes without type info raises ValueError."""
        with pytest.raises(ValueError, match="mime_type or type required"):
            File(b"data")

    def test_bytes_with_type_only(self):
        """Test creating file from bytes with only type."""
        f = File(b"data", type="document")
        assert f.type == "document"


class TestFileFromUrl:
    """Tests for File initialized from URL."""

    def test_image_from_url(self):
        """Test creating file from URL."""
        url = "https://example.com/image.jpg"
        f = File.from_url(url)

        assert f.is_url
        assert f.type == "image"
        assert f.mime_type == "image/jpeg"
        api_format = f.to_api_format()
        assert api_format["url"] == url
        assert "base64" not in api_format

    def test_url_with_explicit_type(self):
        """Test URL with explicit type override."""
        f = File.from_url(
            "https://example.com/file",
            type="document",
            mime_type="application/pdf",
        )
        assert f.type == "document"
        assert f.mime_type == "application/pdf"


class TestFileFactoryMethods:
    """Tests for File factory methods."""

    def test_image_factory(self, tmp_path: Path):
        """Test File.image() factory."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"data")

        f = File.image(str(img_path))
        assert f.type == "image"

    def test_document_factory(self, tmp_path: Path):
        """Test File.document() factory."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_bytes(b"data")

        f = File.document(str(doc_path))
        assert f.type == "document"

    def test_audio_factory(self, tmp_path: Path):
        """Test File.audio() factory."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"data")

        f = File.audio(str(audio_path))
        assert f.type == "audio"


class TestFileRepr:
    """Tests for File string representation."""

    def test_repr_for_url(self):
        """Test repr for URL-based file."""
        f = File.from_url("https://example.com/img.jpg")
        assert "from_url" in repr(f)
        assert "https://example.com/img.jpg" in repr(f)

    def test_repr_for_bytes(self):
        """Test repr for bytes-based file."""
        f = File(b"data", type="document", mime_type="text/plain")
        assert "document" in repr(f)
        assert "text/plain" in repr(f)
