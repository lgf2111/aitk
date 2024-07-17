"""This module provides the AI Toolkit CLI."""
# aitk/cli.py

from typing import Optional

import typer

from aitk import __app_name__, __version__
from aitk.utils import (
    convert_image as convert_image_util,
    transcribe_video as transcribe_video_util,
    generate_image as generate_image_util,
)
from aitk.config import settings
from openai import OpenAI


app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    return


@app.command(
    no_args_is_help=True,
    help="Convert an image to a specified format. (also 'ci')",
)
@app.command("ci", hidden=True)
def convert_image(
    image: str = typer.Argument(..., help="Path to the input image file"),
    output_format: str = typer.Option(
        ...,
        "--format",
        "-f",
        help="Output format (jpeg, png, gif, bmp, or tiff)",
    ),
):
    """Convert an image to a specified format."""
    try:
        with open(image, "rb") as file:
            result = convert_image_util(file, output_format)
            output_filename = f"{image.rsplit('.', 1)[0]}.{output_format}"
            with open(output_filename, "wb") as output_file:
                output_file.write(result)
            typer.echo(f"Image converted successfully. Output: {output_filename}")

    except Exception as e:
        typer.echo(f"Error converting image: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command(
    no_args_is_help=True,
    help="Transcribe a video using either local or remote processing. (also 'tv')",
)
@app.command("tv", hidden=True)
def transcribe_video(
    video: str = typer.Argument(..., help="Path to the input video file"),
    transcription_method: str = typer.Option(
        "local",
        "--method",
        "-m",
        help="Transcription method (local or remote)",
    ),
):
    """Transcribe a video using either local or remote processing."""
    try:
        transcript = transcribe_video_util(video, transcription_method)
        typer.echo(transcript)

    except Exception as e:
        typer.echo(f"Error transcribing video: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command(
    no_args_is_help=True,
    help="Generate an image based on a text prompt. (also 'gi')",
)
@app.command("gi", hidden=True)
def generate_image(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    output_path: str = typer.Option(
        "generated_image.png",
        "--output",
        "-o",
        help="Path to save the generated image",
    ),
):
    """Generate an image based on a text prompt."""
    try:
        openai_client = OpenAI(api_key=settings.openai_api_key)
        image_response = generate_image_util(openai_client, prompt)

        # Save the image to the specified output path
        with open(output_path, "wb") as f:
            f.write(image_response.content)

        typer.echo(f"Image generated successfully. Saved to: {output_path}")

    except Exception as e:
        typer.echo(f"Error generating image: {str(e)}", err=True)
        raise typer.Exit(code=1)
