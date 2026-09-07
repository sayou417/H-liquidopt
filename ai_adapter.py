import os

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.6-luna"


def get_openai_client(api_key=None):
    """
    Create an OpenAI client.

    API key should be provided through
    Streamlit Secrets / environment variables,
    never hard-coded in the source code.
    """

    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=key)


def test_openai_connection(api_key=None):
    """
    Minimal API connection test.
    No engineering calculation is performed here.
    """

    client = get_openai_client(api_key)

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=(
            "Reply with exactly this text and nothing else: "
            "H-LiquidOpt AI connected"
        ),
    )

    return response.output_text.strip()
