import os
import base64
import hashlib
import json
import logging
from dotenv import load_dotenv
import time
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AccessToken, TokenCredential
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv()

# Retrieve credentials from .env file or environment
endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
# Initialize Azure OpenAI client for GPT model

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def _base64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _try_decode_jwt_payload(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = _base64url_decode(parts[1]).decode("utf-8")
        return json.loads(payload)
    except Exception:
        return None


def _token_fingerprint(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"sha256={digest} token={token[:8]}...{token[-8:]}"


def log_access_token_and_audience(access_token: str) -> None:
    payload = _try_decode_jwt_payload(access_token)
    aud = payload.get("aud") if isinstance(payload, dict) else None
    logger.info("AAD access token acquired", extra={"aud": aud, "token": _token_fingerprint(access_token)})


if not endpoint:
    raise ValueError(
        "gpt_endpoint is required and must be the Projects endpoint (..services.ai.azure.com/api/projects/..) when using AAD auth"
    )


class _LoggingTokenCredential(TokenCredential):
    def __init__(self, inner: TokenCredential):
        self._inner = inner

    def get_token(self, *scopes: str, **kwargs) -> AccessToken:
        access_token = self._inner.get_token(*scopes, **kwargs)
        log_access_token_and_audience(access_token.token)
        # Helpful when debugging auth failures/mismatched audiences
        logger.info("Token scopes requested", extra={"scopes": list(scopes)})
        return access_token


# Prefer the official Projects env var if present; fall back to gpt_endpoint.
project_endpoint = os.getenv("FOUNDRY_ENDPOINT") or endpoint
credential = _LoggingTokenCredential(DefaultAzureCredential())

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=credential,
)

# This returns an OpenAI client configured for the Projects endpoint.
client: AzureOpenAI = project_client.get_openai_client()
 

#log client token
logger.info("Azure OpenAI client initialized", extra={"endpoint": project_endpoint})

def generate_response(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    # Prepare the full prompt in Responses API schema (type: message with input_text content)
    chat_prompt = [
        {
            "type": "message",
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                        Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                        If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                        Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
                    """
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text_input,
                }
            ],
        },
    ]

    # Call Azure OpenAI Responses API (Projects endpoint supports this path)
    response = client.responses.create(
        model=deployment,
        input=chat_prompt,
        max_output_tokens=1000,
        top_p=1,
        stream=False,
    )
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")

    # Extract output_text: could be a string or list of strings
    output_text = getattr(response, "output_text", None)
    if output_text:
        if isinstance(output_text, str):
            return output_text
        elif isinstance(output_text, list) and len(output_text) > 0:
            return output_text[0]

    # Fallback: try to surface the first text chunk in the output
    output = getattr(response, "output", None)
    if output and len(output) > 0 and hasattr(output[0], "content"):
        content = output[0].content
        if isinstance(content, list) and len(content) > 0 and hasattr(content[0], "text"):
            return getattr(content[0].text, "value", "")

    return "(no response text returned)"
