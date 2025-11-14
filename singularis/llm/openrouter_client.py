"""OpenRouter API client (chat completions).

Env vars:
- OPENROUTER_API_KEY (required)
- OPENROUTER_SITE_URL (optional HTTP-Referer)
- OPENROUTER_SITE_NAME (optional X-Title)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import aiohttp


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self.site_url = os.getenv("OPENROUTER_SITE_URL")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def _ensure(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("OpenRouter API key not configured (OPENROUTER_API_KEY)")

        session = await self._ensure()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(extra)

        async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        return {"content": content, "raw": data}

    async def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = await self.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return str(resp.get("content", ""))
