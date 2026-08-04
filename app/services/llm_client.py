"""Provider-agnostic LLM client with automatic fallback."""

from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from app.config.settings import settings

_REQUEST_TIMEOUT = 30.0


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    api_url: str
    model: str
    format: str = "openai"


class LLMClient:
    def __init__(self) -> None:
        self._providers: list[ProviderConfig] = self._build_providers()
        logger.info(
            f"LLM client initialized with providers: "
            f"{[p.name for p in self._providers]}"
        )

    def _build_providers(self) -> list[ProviderConfig]:
        priority = [p.strip() for p in settings.llm_provider_priority.split(",")]
        providers: list[ProviderConfig] = []

        for name in priority:
            if name == "groq" and settings.groq_api_key:
                providers.append(
                    ProviderConfig(
                        name="groq",
                        api_key=settings.groq_api_key,
                        api_url="https://api.groq.com/openai/v1/chat/completions",
                        model=settings.groq_model,
                    )
                )
            elif name == "gemini" and settings.gemini_api_key:
                providers.append(
                    ProviderConfig(
                        name="gemini",
                        api_key=settings.gemini_api_key,
                        api_url=(
                            "https://generativelanguage.googleapis.com/v1beta/"
                            f"models/{settings.gemini_model}:generateContent"
                        ),
                        model=settings.gemini_model,
                        format="gemini",
                    )
                )
            elif name == "openrouter" and settings.openrouter_api_key:
                models = [m.strip() for m in settings.openrouter_models.split(",")]
                for model in models:
                    providers.append(
                        ProviderConfig(
                            name=f"openrouter/{model}",
                            api_key=settings.openrouter_api_key,
                            api_url="https://openrouter.ai/api/v1/chat/completions",
                            model=model,
                        )
                    )

        return providers

    @property
    def available(self) -> bool:
        return len(self._providers) > 0

    def _call_openai(
        self, provider: ProviderConfig, system: str, user: str
    ) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.post(provider.api_url, headers=headers, json=body)
            if resp.status_code == 429:
                logger.warning(f"{provider.name} rate-limited (429)")
                return None
            if resp.status_code >= 500:
                logger.warning(f"{provider.name} server error ({resp.status_code})")
                return None
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(
        self, provider: ProviderConfig, system: str, user: str
    ) -> Optional[str]:
        url = f"{provider.api_url}?key={provider.api_key}"
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system}\n\n{user}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
            },
        }
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.post(url, json=body)
            if resp.status_code == 429:
                logger.warning(f"{provider.name} rate-limited (429)")
                return None
            if resp.status_code >= 500:
                logger.warning(f"{provider.name} server error ({resp.status_code})")
                return None
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self._providers:
            raise RuntimeError("No LLM providers configured")

        last_error = ""
        for provider in self._providers:
            logger.info(f"Trying LLM provider: {provider.name}")
            try:
                if provider.format == "gemini":
                    result = self._call_gemini(provider, system_prompt, user_prompt)
                else:
                    result = self._call_openai(provider, system_prompt, user_prompt)

                if result is not None:
                    logger.info(f"LLM response received from {provider.name}")
                    return result

                last_error = f"{provider.name} returned no response"
            except Exception as exc:
                last_error = f"{provider.name} error: {exc}"
                logger.warning(last_error)
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
