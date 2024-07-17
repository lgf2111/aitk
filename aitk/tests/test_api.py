import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from aitk.api import app
from aitk.utils import convert_image, transcribe_video, generate_image


TEST_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "test_assets")
TEST_IMAGE_FILEPATH = os.path.join(TEST_ASSETS_DIR, "test_image.jpeg")
TEST_VIDEO_FILEPATH = os.path.join(TEST_ASSETS_DIR, "test_video.mp4")

client = TestClient(app)


def test_redirect_to_docs():
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.path == "/docs"


@pytest.mark.skip()
def test_convert_image():
    input_filepath = os.path.join(TEST_ASSETS_DIR, "test_image.jpeg")
    with open(input_filepath, "rb") as image_file:
        test_input_image = image_file.read()

    output_filepath = os.path.join(TEST_ASSETS_DIR, "test_image.png")
    with open(output_filepath, "rb") as image_file:
        test_output_image = image_file.read()

    with patch("aitk.api.convert_image", return_value=test_output_image):
        response = client.post(
            "/convert-image",
            files={"image": ("test.png", test_input_image)},
            data={"output_format": "jpeg"},
        )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/jpeg"
    assert (
        response.headers["Content-Disposition"]
        == 'attachment; filename="converted_image.jpeg"'
    )
    assert response.content == test_output_image


@pytest.mark.skip()
def test_transcribe_video():
    test_video = b"fake video content"
    test_transcript = "This is a test transcript."

    with patch("aitk.api.transcribe_video", return_value=test_transcript):
        response = client.post(
            "/transcribe-video",
            files={"video": ("test.mp4", test_video)},
            data={"transcription_method": "local"},
        )

    assert response.status_code == 200
    assert response.json() == {"transcript": test_transcript}


def test_generate_image():
    test_prompt = "A beautiful sunset"
    test_image = b"generated image content"

    mock_image_response = MagicMock()
    mock_image_response.content = test_image

    with patch("aitk.api.generate_image", return_value=mock_image_response):
        response = client.post("/generate-image", params={"prompt": test_prompt})

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert (
        response.headers["Content-Disposition"]
        == 'attachment; filename="generated_image.png"'
    )
    assert response.content == test_image
