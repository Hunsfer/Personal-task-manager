"""Client for the local Ollama server. Only anonymized text may be passed to it."""

from dataclasses import dataclass
from typing import Any

import httpx

REQUEST_TIMEOUT_S = 300.0
CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 1500
SEED = 42


class OllamaUnavailableError(Exception):
    """The server can't be reached or refused the request. Its message contains no text."""


class CloudModelError(ValueError):
    """Names a model that runs on Ollama's servers rather than on this machine."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    prompt_tokens: int
    output_tokens: int

    @property
    def truncated_input(self) -> bool:
        """Ollama silently drops the start of a prompt that doesn't fit."""
        return self.prompt_tokens >= CONTEXT_TOKENS

    @property
    def truncated_output(self) -> bool:
        return self.finish_reason == "length"


def ensure_local(model: str) -> None:
    """Second guard next to OLLAMA_NO_CLOUD: messages must never leave this machine."""
    if model.endswith("-cloud") or model.endswith(":cloud"):
        raise CloudModelError(f"{model} is a cloud model; only local models may be used")


def think_setting(show: dict[str, Any]) -> bool | str | None:
    """Least thinking the model allows. None leaves the field out of the request."""
    thinking = show.get("thinking")
    values = thinking.get("values") if isinstance(thinking, dict) else None
    if not values:
        return None
    if False in values:
        return False
    return "low" if "low" in values else None


class OllamaClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=REQUEST_TIMEOUT_S, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(type(error).__name__) from None
        if response.status_code != httpx.codes.OK:
            raise OllamaUnavailableError(f"HTTP {response.status_code}")
        return response.json()

    async def ping(self) -> None:
        """Raises OllamaUnavailableError if the server doesn't answer."""
        try:
            response = await self._client.get("/api/version")
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(type(error).__name__) from None
        if response.status_code != httpx.codes.OK:
            raise OllamaUnavailableError(f"HTTP {response.status_code}")

    async def show(self, model: str) -> dict[str, Any]:
        ensure_local(model)
        return await self._post("/api/show", {"model": model})

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        think: bool | str | None = None,
        seed: int = SEED,
    ) -> ChatResult:
        ensure_local(model)
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_ctx": CONTEXT_TOKENS,
                "num_predict": MAX_OUTPUT_TOKENS,
            },
        }
        if think is not None:
            body["think"] = think
        data = await self._post("/api/chat", body)
        return ChatResult(
            content=data.get("message", {}).get("content", ""),
            finish_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
