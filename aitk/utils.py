from io import BufferedReader, BytesIO
from typing import Union

import httpx
import whisper
from fastapi import HTTPException
from openai import OpenAI
from PIL import Image

from aitk.config import Settings


def convert_image(
    image_input: Union[bytes, BufferedReader], output_format: str
) -> bytes:
    if isinstance(image_input, BufferedReader):
        # If image_input is a file object
        input_image = Image.open(image_input)
    else:
        # If image_input is bytes
        input_image = Image.open(BytesIO(image_input))

    output_buffer = BytesIO()
    input_image.convert("RGB").save(output_buffer, format=output_format.upper())
    output_buffer.seek(0)

    return output_buffer.getvalue()


def transcribe_video(video_path: str, transcription_method: str) -> str:
    if transcription_method == "remote":
        # Use OpenAI's Whisper API for remote transcription
        client = OpenAI(api_key=Settings.openai_api_key)

        with open(video_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        return transcription.text

    else:  # local transcription
        # Use OpenAI's Whisper model locally
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result["text"]


def generate_image(openai_client: OpenAI, prompt: str):
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url

        # Download the image
        with httpx.Client() as client:
            image_response = client.get(image_url)
            image_response.raise_for_status()

        return image_response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )
