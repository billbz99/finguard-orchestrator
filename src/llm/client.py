import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("XAI_API_KEY")
    model = os.getenv("XAI_MODEL", "grok-4.3")

    if not api_key:
        raise RuntimeError("XAI_API_KEY is not configured")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        temperature=0,
    )