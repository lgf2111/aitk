from openai import OpenAI
from aitk.config import settings


async def get_openai_client():
    openai_api_key = settings.openai_api_key
    if openai_api_key == "sk-pleasechangeme":
        raise ValueError("Please change the OpenAI API key in the settings")
    client = OpenAI(api_key=openai_api_key)
    return client
