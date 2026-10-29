"""Thin LLM wrapper. Provider is picked from env; with no key we fall back to None
and callers use an extractive answer so the whole pipeline still runs offline."""
import os

from .config import settings


_REJECTED: set[str] = set()  # providers whose key was refused this run; auto mode skips them


class LLMError(Exception):
    """Safe-to-show LLM failure (never includes the provider's message, which can echo the API key)."""


class LLM:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif provider == "openai":
            import openai

            self._client = openai.OpenAI()
        elif provider == "ollama":
            import openai

            # Ollama exposes an OpenAI-compatible API; it ignores the key but the client requires one.
            self._client = openai.OpenAI(base_url=settings.ollama_url, api_key="ollama", timeout=300)

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        try:
            return self._complete(system, user, max_tokens)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (401, 403) and settings.llm_provider.lower() == "auto":
                _REJECTED.add(self.provider)
                fallback = get_llm()  # e.g. a local Ollama model
                if fallback:
                    return fallback.complete(system, user, max_tokens)
            hint = " - check the API key in .env" if status in (401, 403) else ""
            raise LLMError(f"{self.provider} request failed ({type(e).__name__}{f' {status}' if status else ''}){hint}") from None

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            r = self._client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in r.content if b.type == "text").strip()
        r = self._client.chat.completions.create(
            model=settings.ollama_model if self.provider == "ollama" else settings.openai_model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()


def _ollama_up() -> bool:
    import requests

    try:
        return requests.get(settings.ollama_url.removesuffix("/v1") + "/api/tags", timeout=0.5).ok
    except requests.RequestException:
        return False


def get_llm() -> LLM | None:
    p = settings.llm_provider.lower()
    if p == "auto":
        if os.getenv("ANTHROPIC_API_KEY") and "anthropic" not in _REJECTED:
            p = "anthropic"
        elif os.getenv("OPENAI_API_KEY") and "openai" not in _REJECTED:
            p = "openai"
        elif _ollama_up():
            p = "ollama"
        else:
            return None
    if p == "none":
        return None
    return LLM(p)
