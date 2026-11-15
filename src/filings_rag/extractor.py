"""Extract structured fields from filing text with an LLM, validating and retrying."""

import json
import os
from typing import Protocol

import httpx
from pydantic import ValidationError

from filings_rag.schema import FilingExtraction

BRIDGE_URL = "http://127.0.0.1:8787/v1"  # where scripts/claude_cli_bridge.py listens
MODEL = "opus"  # a model alias the bridge passes to `claude --model`
# The bridge stops a call after 180 seconds, so wait a little longer than that.
REQUEST_TIMEOUT_SECONDS = 200
START_COMMAND = "python scripts/claude_cli_bridge.py"
MAX_ERROR_CHARACTERS = 300
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You extract key fields from SEC 10-K and 10-Q filings.
Answer with one JSON object and nothing else. It must match this JSON schema:
{schema}

Rules:
- Use the current fiscal period only, not prior-year comparison columns.
- For a 10-Q, use the three-month figure for revenue and net income, not the year to date.
- Convert amounts to US dollars. A table headed "in millions" means multiply by 1,000,000.
- "page" is the number in the [page N] marker of the text you read the value from.
- Use null for a value the text does not state. Do not guess."""


class Completer(Protocol):
    """Anything that turns a system prompt and a user prompt into text."""

    def complete(self, system: str, user: str) -> str: ...


class CompletionError(Exception):
    """The bridge could not return an answer, so retrying with feedback would not help."""


class BridgeCompleter:
    """Asks the local Claude Code bridge, which runs the signed in `claude` CLI.

    The bridge speaks the OpenAI chat format. No API key is needed, and the calls count against
    the Claude Code plan that is signed in on this machine.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ):
        # An environment variable that is set but empty counts as not set.
        base_url = base_url or os.environ.get("LLM_BASE_URL") or BRIDGE_URL
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model or os.environ.get("LLM_MODEL") or MODEL
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.ConnectError as error:
            raise CompletionError(
                f"Cannot reach the bridge at {self._url}. "
                f"Start it with `{START_COMMAND}` (or `make bridge`)."
            ) from error
        except httpx.TimeoutException as error:
            raise CompletionError(
                f"The bridge at {self._url} did not answer within {REQUEST_TIMEOUT_SECONDS} seconds"
            ) from error
        except httpx.HTTPError as error:
            raise CompletionError(
                f"The request to {self._url} failed: {type(error).__name__}: {error}"
            ) from error

        if response.is_error:
            raise CompletionError(
                f"The bridge returned {response.status_code}: {_error_message(response)}"
            )
        return _reply_text(response)


def _error_message(response: httpx.Response) -> str:
    """The bridge's own error message when it sent one, cut short, else the start of the body."""
    try:
        text = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        text = response.text
    return text[:MAX_ERROR_CHARACTERS]


def _reply_text(response: httpx.Response) -> str:
    try:
        text = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise CompletionError(
            f"Expected an OpenAI style chat reply but got: {response.text[:MAX_ERROR_CHARACTERS]}"
        ) from error
    if not isinstance(text, str):
        raise CompletionError("The bridge reply has no text")
    return text


def completer_from_env() -> BridgeCompleter:
    return BridgeCompleter()


class ExtractionError(Exception):
    """The model never produced output that passes the schema."""


def extract_filing(
    completer: Completer, context: str, description: str = "", max_attempts: int = MAX_ATTEMPTS
) -> FilingExtraction:
    """Ask for the fields, and on invalid output show the model its error and ask again."""
    schema = json.dumps(FilingExtraction.model_json_schema())
    system = SYSTEM_PROMPT.format(schema=schema)
    header = f"Filing: {description}\n\n" if description else ""
    prompt = f"{header}Filing excerpts:\n\n{context}"
    user = prompt

    last_error = ""
    for _ in range(max_attempts):
        reply = completer.complete(system, user)
        try:
            return FilingExtraction.model_validate_json(_strip_code_fence(reply))
        except ValidationError as error:
            last_error = str(error)
            user = (
                f"{prompt}\n\n"
                f"Your previous answer was invalid:\n{last_error}\nReturn corrected JSON only."
            )
    raise ExtractionError(f"No valid output after {max_attempts} attempts: {last_error}")


def _strip_code_fence(reply: str) -> str:
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```")
    return text.strip()
