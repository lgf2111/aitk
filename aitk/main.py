import os
import tempfile
from typing import Annotated

from fastapi import Depends, FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from openai import OpenAI

from aitk.dependencies import get_openai_client
from aitk.utils import convert_image, generate_image, transcribe_video

app = FastAPI()


@app.get("/")
async def _redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post("/convert-image")
async def _convert_image(
    image: UploadFile = File(...),
    output_format: str = Query(..., enum=["jpeg", "png", "gif", "bmp", "tiff"]),
):
    # Prepare the response
    filename = f"converted_image.{output_format.lower()}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    # Create a temporary file to store the converted image
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{output_format.lower()}"
    ) as temp_file:
        converted_image = convert_image(await image.read(), output_format)
        temp_file.write(converted_image)
        temp_file_path = temp_file.name

    return FileResponse(
        temp_file_path,
        headers=headers,
        media_type=f"image/{output_format.lower()}",
        filename=filename,
    )


@app.post("/transcribe-video")
async def _transcribe_video(
    video: UploadFile = File(...),
    transcription_method: str = Query(..., enum=["local", "remote"]),
):
    # Read the uploaded video file
    contents = await video.read()

    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    transcript = transcribe_video(temp_file_path, transcription_method)

    # Clean up the temporary file
    os.unlink(temp_file_path)

    return {"transcript": transcript}


@app.post("/generate-image")
async def _generate_image(
    openai_client: Annotated[OpenAI, Depends(get_openai_client)],
    prompt: str = Query(..., description="The prompt for image generation"),
):
    image_response = generate_image(openai_client, prompt)
    # Save the image to a temporary file

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_response.content)
        temp_file_path = temp_file.name

    # Return the image file using FileResponse
    return FileResponse(
        temp_file_path, media_type="image/png", filename="generated_image.png"
    )
