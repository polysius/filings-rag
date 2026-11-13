"""Extract structured fields from filing text with an LLM, validating and retrying."""

import json
from typing import Protocol

from pydantic import ValidationError

from filings_rag.schema import FilingExtraction

MODEL = "claude-opus-5-5"
# Thinking tokens count against the limit, so leave room for them and for the JSON answer.
MAX_TOKENS = 16_000
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
    """The model stopped without a complete answer, so retrying with feedback would not help."""


class AnthropicCompleter:
    def __init__(self, model: str = MODEL):
        import anthropic  # imported here so the rest of the package works without the SDK

        # The client finds credentials itself: an API key, an auth token or a logged-in profile.
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            # If a safety classifier declines the request, the API reruns it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise CompletionError(f"Model stopped early: {response.stop_reason}")
        return "".join(block.text for block in response.content if block.type == "text")


def completer_from_env() -> AnthropicCompleter:
    return AnthropicCompleter()


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
