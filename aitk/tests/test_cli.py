# tests/test_rptodo.py

import os
from unittest.mock import mock_open, patch

import pytest
from typer.testing import CliRunner

from aitk import __app_name__, __version__, cli
from aitk.config import settings
from aitk.utils import (
    convert_image as convert_image_util,
)
from aitk.utils import (
    generate_image as generate_image_util,
)
from aitk.utils import (
    transcribe_video as transcribe_video_util,
)

TEST_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "test_assets")
IMAGE_FILEPATH = os.path.join(TEST_ASSETS_DIR, "test_image.jpeg")
VIDEO_FILEPATH = os.path.join(TEST_ASSETS_DIR, "test_video.mp4")

runner = CliRunner()


def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout


def test_convert_image():
    with patch("aitk.cli.convert_image_util") as mock_convert, patch(
        "builtins.open", mock_open()
    ) as mock_file:
        mock_convert.return_value = b"converted_image_data"
        result = runner.invoke(cli.app, ["convert-image", IMAGE_FILEPATH, "-f", "png"])
        assert result.exit_code == 0
        assert "Image converted successfully" in result.stdout
        mock_convert.assert_called_once()
        mock_file.assert_called()


def test_convert_image_error():
    with patch("aitk.cli.convert_image_util", side_effect=Exception("Test error")):
        result = runner.invoke(cli.app, ["convert-image", IMAGE_FILEPATH, "-f", "png"])
        assert result.exit_code == 1


def test_transcribe_video_locally():
    with patch("aitk.cli.transcribe_video_util") as mock_transcribe:
        mock_transcribe.return_value = "Transcribed text locally"
        result = runner.invoke(
            cli.app,
            ["transcribe-video", VIDEO_FILEPATH, "-m", "local"],
        )
        assert result.exit_code == 0
        assert "Transcribed text locally" in result.stdout
        mock_transcribe.assert_called_once_with(VIDEO_FILEPATH, "local")


def test_transcribe_video_remotely():
    with patch("aitk.cli.transcribe_video_util") as mock_transcribe:
        mock_transcribe.return_value = "Transcribed text remotely"
        result = runner.invoke(
            cli.app,
            ["transcribe-video", VIDEO_FILEPATH, "-m", "remote"],
        )
        assert result.exit_code == 0
        assert "Transcribed text remotely" in result.stdout
        mock_transcribe.assert_called_once_with(VIDEO_FILEPATH, "remote")


def test_transcribe_video_error():
    with patch("aitk.cli.transcribe_video_util", side_effect=Exception("Test error")):
        result = runner.invoke(
            cli.app, ["transcribe-video", "video.mp4", "-m", "local"]
        )
        assert result.exit_code == 1
        assert "Error transcribing video: Test error" in result.stdout


def test_generate_image():
    with patch("aitk.cli.OpenAI") as mock_openai, patch(
        "aitk.cli.generate_image_util"
    ) as mock_generate, patch("builtins.open", mock_open()) as mock_file:
        mock_response = mock_openai.return_value
        mock_response.content = b"generated_image_data"
        mock_generate.return_value = mock_response
        result = runner.invoke(
            cli.app, ["generate-image", "test prompt", "-o", "output.png"]
        )
        assert result.exit_code == 0
        assert "Image generated successfully" in result.stdout
        mock_generate.assert_called_once()
        mock_file.assert_called_with("output.png", "wb")


def test_generate_image_error():
    with patch("aitk.cli.OpenAI"), patch(
        "aitk.cli.generate_image_util", side_effect=Exception("Test error")
    ):
        result = runner.invoke(cli.app, ["generate-image", "test prompt"])
        assert result.exit_code == 1
        assert "Error generating image: Test error" in result.stdout
