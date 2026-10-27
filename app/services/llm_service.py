"""
app/services/llm_service.py
============================
LLM access with automatic provider + model selection.

Supports Groq and Google Gemini (both over plain HTTPS, no SDK needed).

HOW IT PICKS A MODEL (so a retired model never breaks the app again):
    1. Providers are tried in order: Groq, then Gemini (or only the one set in
       LLM_PROVIDER). A provider is skipped if its API key is rejected.
    2. Per provider: the model from .env (GROQ_MODEL / GEMINI_MODEL) first,
       then well-known models, then whatever the provider's own
       "list models" endpoint says your key can use.
    3. The first (provider, model) that answers is remembered.
    4. Temporary errors (network reset, 429, 5xx) are retried before moving on.
"""

import time

import requests

from app.core.config import get_config
from app.core.models import Chunk
from app.core.exceptions import APIKeyError, RateLimitError, LLMResponseError
from app.utils.logger import get_logger
from app.utils.helpers import timer

logger = get_logger(__name__)
cfg    = get_config()

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Answer questions strictly based on the document context provided.
- ONLY use information from the provided context
- If context doesn't contain the answer, say so clearly
- Be precise, professional, and concise
- Never fabricate facts not present in the context"""

GROQ_URL   = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"

# Tried before anything discovered; order = preference.
GROQ_PREFERRED = [
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct", "gemma2-9b-it",
]
GEMINI_PREFERRED = [
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash",
    "gemini-flash-latest", "gemini-1.5-flash",
]
# Name fragments of models that are not text-chat models
_SKIP = ("whisper", "tts", "guard", "playai", "embed", "orpheus", "distil",
         "image", "imagen", "veo", "live", "audio", "aqa", "robotics",
         "computer-use", "learnlm", "vision", "gemma-3n")


class ProviderError(Exception):
    """A failed call to one provider. status=None means a network failure."""
    def __init__(self, status, message):
        self.status  = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}" if status else message)

    @property
    def is_auth(self) -> bool:
        m = self.message.lower()
        return self.status in (401, 403) or "api key not valid" in m or "api_key_invalid" in m

    @property
    def is_temporary(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500


def _short(text, n: int = 300) -> str:
    return " ".join(str(text).split())[:n]


# ── Providers ────────────────────────────────────────────────────────────────

class _Provider:
    name      = ""
    preferred : list[str] = []

    def __init__(self, api_key: str, configured_model: str = ""):
        self.api_key          = api_key
        self.configured_model = configured_model

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def call(self, model: str, system: str, messages: list[dict]) -> tuple[str, int]:
        raise NotImplementedError

    def candidates(self) -> list[str]:
        first = [self.configured_model] if self.configured_model else []
        return first + [m for m in self.preferred if m != self.configured_model]

    @staticmethod
    def _http(method, url, **kw):
        try:
            return requests.request(method, url, timeout=kw.pop("timeout", 60), **kw)
        except requests.RequestException as e:
            raise ProviderError(None, f"Network error: {_short(e)}") from e

    @staticmethod
    def _check(r):
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message") or r.text
            except Exception:
                msg = r.text
            raise ProviderError(r.status_code, _short(msg))


class GroqProvider(_Provider):
    name      = "groq"
    preferred = GROQ_PREFERRED

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        r = self._http("GET", f"{GROQ_URL}/models", headers=self._headers(), timeout=20)
        self._check(r)
        ids  = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        chat = [m for m in ids if not any(x in m.lower() for x in _SKIP)]
        chat.sort(key=lambda m: (("instant" not in m and "versatile" not in m), len(m)))
        return chat

    def call(self, model, system, messages):
        r = self._http(
            "POST", f"{GROQ_URL}/chat/completions", headers=self._headers(),
            json={
                "model"      : model,
                "messages"   : [{"role": "system", "content": system}, *messages],
                "temperature": cfg.GEMINI_TEMPERATURE,
                "max_tokens" : min(cfg.GEMINI_MAX_TOKENS, 1024),
            },
        )
        self._check(r)
        try:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip(), data.get("usage", {}).get("total_tokens", 0)
        except Exception as e:
            raise ProviderError(200, f"Unexpected response format: {e}") from e


class GeminiProvider(_Provider):
    name      = "gemini"
    preferred = GEMINI_PREFERRED

    def _headers(self):
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        r = self._http("GET", f"{GEMINI_URL}/models?pageSize=200", headers=self._headers(), timeout=20)
        self._check(r)
        out = []
        for m in r.json().get("models", []):
            name = m.get("name", "").replace("models/", "")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if not name.startswith("gemini") or any(x in name.lower() for x in _SKIP):
                continue
            out.append(name)
        # stable, fast models first: flash before pro, previews/experiments last
        out.sort(key=lambda n: (("preview" in n or "exp" in n), "flash" not in n, len(n)))
        return out

    def call(self, model, system, messages):
        # Gemini wants alternating user/model turns that start with "user"
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n\n" + m["content"]
            else:
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        while contents and contents[0]["role"] != "user":
            contents.pop(0)

        r = self._http(
            "POST", f"{GEMINI_URL}/models/{model}:generateContent", headers=self._headers(),
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents"         : contents,
                "generationConfig" : {
                    "temperature"    : cfg.GEMINI_TEMPERATURE,
                    "maxOutputTokens": max(cfg.GEMINI_MAX_TOKENS, 1024),
                },
            },
        )
        self._check(r)
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text  = "".join(p.get("text", "") for p in parts).strip()
            if text:
                return text, data.get("usageMetadata", {}).get("totalTokenCount", 0)
        except (KeyError, IndexError, TypeError):
            pass
        reason = data.get("promptFeedback", {}).get("blockReason") or \
                 (data.get("candidates") or [{}])[0].get("finishReason", "") or "empty response"
        raise ProviderError(200, f"Gemini returned no text ({reason})")


# ── Service ──────────────────────────────────────────────────────────────────

class LLMService:
    """Generates answers; picks provider + model automatically."""

    def __init__(self):
        wanted = (cfg.LLM_PROVIDER or "auto").lower()
        self._providers: list[_Provider] = []
        if cfg.GROQ_API_KEY and wanted in ("auto", "groq"):
            self._providers.append(GroqProvider(cfg.GROQ_API_KEY.strip(), cfg.GROQ_MODEL.strip()))
        if cfg.GEMINI_API_KEY and wanted in ("auto", "gemini"):
            self._providers.append(GeminiProvider(cfg.GEMINI_API_KEY.strip(), cfg.GEMINI_MODEL.strip()))
        if not self._providers:
            raise APIKeyError()

        self._model   = "auto"
        self._working = None          # (provider, model) that last worked
        logger.info(f"✅ LLM ready, providers: {[p.name for p in self._providers]}")

    # ── answering ────────────────────────────────────────────────────────────

    @timer
    def generate_answer(
        self,
        query         : str,
        context_chunks: list[Chunk],
        chat_history  : list[dict] | None = None
    ) -> tuple[str, int]:

        messages = [*(chat_history or []),
                    {"role": "user", "content": self._build_prompt(query, context_chunks)}]

        # 1) the combination that worked last time
        if self._working:
            provider, model = self._working
            try:
                return self._call_with_retries(provider, model, messages)
            except ProviderError as e:
                logger.warning(f"Remembered model {provider.name}/{model} failed ({e}); searching again")
                self._working = None

        # 2) search providers, then models
        errors      : dict[str, str] = {}
        auth_failed : set[str]       = set()
        rate_limited = False

        for provider in self._providers:
            for model in self._model_order(provider):
                try:
                    result = self._call_with_retries(provider, model, messages)
                except ProviderError as e:
                    errors[provider.name] = f"{model}: {e}"
                    logger.warning(f"{provider.name}/{model} failed: {e}")
                    if e.is_auth:
                        auth_failed.add(provider.name)
                        break                      # bad key: skip this provider
                    rate_limited = rate_limited or e.status == 429
                    continue
                self._working = (provider, model)
                self._model   = f"{provider.name}/{model}"
                logger.info(f"Using {provider.name}/{model}")
                return result

        if len(auth_failed) == len(self._providers):
            raise APIKeyError()
        if rate_limited:
            raise RateLimitError()
        raise LLMResponseError(" | ".join(f"[{k}] {v}" for k, v in errors.items()))

    def _call_with_retries(self, provider, model, messages, attempts: int = 3):
        for i in range(attempts):
            try:
                answer, tokens = provider.call(model, SYSTEM_PROMPT, messages)
                return answer or "(The model returned an empty answer.)", tokens
            except ProviderError as e:
                if not e.is_temporary or i == attempts - 1:
                    raise
                time.sleep(1.5 * (i + 1))

    def _model_order(self, provider: _Provider) -> list[str]:
        """Known models first; then anything the provider says the key can use."""
        order = list(provider.candidates())
        try:
            available = provider.list_models()
        except ProviderError as e:
            logger.warning(f"{provider.name}: could not list models ({e}); using built-in list")
            return order[:12]
        if not available:
            return order[:12]
        # Keep known names that really exist for this key, then the rest of what exists
        ordered = [m for m in order if m in available] + [m for m in available if m not in order]
        return ordered[:12]

    # ── diagnostics (used by check_models.py) ────────────────────────────────

    def diagnose(self, max_per_provider: int = 12):
        """Test models on every configured provider. Yields (provider, model, ok, detail)."""
        ping = [{"role": "user", "content": "Reply with the single word: OK"}]
        for provider in self._providers:
            models = []
            try:
                models = provider.list_models()
                yield provider.name, "(list models)", True, f"{len(models)} models visible to this key"
            except ProviderError as e:
                yield provider.name, "(list models)", False, str(e)
                if e.is_auth:
                    continue
            tests = list(dict.fromkeys(models + provider.candidates()))[:max_per_provider]
            for m in tests:
                try:
                    ans, _ = provider.call(m, "You are a test.", ping)
                    yield provider.name, m, True, _short(ans, 40)
                except ProviderError as e:
                    yield provider.name, m, False, str(e)

    # ── prompt building ──────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return f"No relevant context found in documents.\n\nQuestion: {query}\n\nTell the user you couldn't find relevant information."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] File: {chunk.filename} | Page: {chunk.page_number}\n"
                f"{chunk.content}"
            )

        return f"""Document excerpts:

{chr(10).join(context_parts)}

---
Answer this question using ONLY the above context:
{query}"""

    def build_history_for_api(self, messages: list, max_messages: int = 6) -> list[dict]:
        """Last few turns as OpenAI-style messages so follow-up questions work."""
        history = []
        for m in messages[-max_messages:]:
            role = getattr(m.role, "value", m.role)
            if role in ("user", "assistant") and m.content:
                history.append({"role": role, "content": m.content[:2000]})
        return history

