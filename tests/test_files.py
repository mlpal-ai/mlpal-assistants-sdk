"""Tests for File class."""

import base64
from pathlib import Path

import pytest

from mlpal_assistants import File


class TestFileFromPath:
    """Tests for File.from_path factory method."""

    def test_image_from_path(self, tmp_path: Path):
        """Test creating file from image path."""
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image data")

        f = File.from_path(str(img_path))

        assert f.type == "image"
        assert f.mime_type == "image/jpeg"
        api_format = f.to_api_format()
        assert api_format["type"] == "image"
        assert api_format["mime_type"] == "image/jpeg"
        assert api_format["base64"] == base64.b64encode(b"fake image data").decode()

    def test_pdf_from_path(self, tmp_path: Path):
        """Test creating file from PDF path."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf data")

        f = File.from_path(pdf_path)

        assert f.type == "document"
        assert f.mime_type == "application/pdf"

    def test_audio_from_path(self, tmp_path: Path):
        """Test creating file from audio path."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio data")

        f = File.from_path(audio_path)

        assert f.type == "audio"
        assert f.mime_type == "audio/mpeg"

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            File.from_path("/nonexistent/path.jpg")


class TestFileFromBytes:
    """Tests for File.from_bytes factory method."""

    def test_image_from_bytes(self):
        """Test creating file from image bytes."""
        data = b"fake image data"
        f = File.from_bytes(data, mime_type="image/png")

        assert f.type == "image"
        assert f.mime_type == "image/png"
        api_format = f.to_api_format()
        assert api_format["base64"] == base64.b64encode(data).decode()

    def test_bytes_with_explicit_type(self):
        """Test creating file from bytes with explicit type."""
        f = File.from_bytes(b"data", mime_type="text/plain", type="document")
        assert f.type == "document"
        assert f.mime_type == "text/plain"


class TestFileFromUrl:
    """Tests for File.from_url factory method."""

    def test_image_from_url(self):
        """Test creating file from URL."""
        url = "https://example.com/image.jpg"
        f = File.from_url(url)

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
    """Tests for File convenience factory methods."""

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

    def test_image_from_bytes(self):
        """Test File.image() with bytes."""
        f = File.image(b"data", mime_type="image/png")
        assert f.type == "image"


class TestFileRepr:
    """Tests for File string representation."""

    def test_repr_for_url(self):
        """Test repr for URL-based file."""
        f = File.from_url("https://example.com/img.jpg")
        assert "image" in repr(f)
        assert "https://example.com/img.jpg" in repr(f)

    def test_repr_for_bytes(self):
        """Test repr for bytes-based file."""
        f = File.from_bytes(b"data", mime_type="text/plain")
        assert "document" in repr(f)
        assert "text/plain" in repr(f)
