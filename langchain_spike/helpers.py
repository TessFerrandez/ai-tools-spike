import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI


load_dotenv(find_dotenv())


def get_gpt35_chat_client() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        openai_api_type="azure",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment="gpt-35-turbo",
        temperature=0.0
    )


