"""Perplexity API client (chat completions with web research).

Env:
- PERPLEXITY_API_KEY (required)

Docs:
- Endpoint: https://api.perplexity.ai/chat/completions
- Models (examples): 'sonar-small-online', 'sonar-medium-online'
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import aiohttp


class PerplexityClient:
    """
    An asynchronous client for the Perplexity API.

    This class provides a wrapper around the Perplexity chat completions API,
    which is specialized for web research.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.perplexity.ai",
        timeout: int = 120,
        default_model: str = "sonar-medium-online",
    ) -> None:
        """
        Initializes the PerplexityClient.

        Args:
            api_key (Optional[str], optional): The Perplexity API key. If not
                                             provided, it is read from the
                                             PERPLEXITY_API_KEY environment
                                             variable. Defaults to None.
            base_url (str, optional): The base URL for the Perplexity API.
                                      Defaults to "https://api.perplexity.ai".
            timeout (int, optional): The request timeout in seconds. Defaults to 120.
            default_model (str, optional): The default model to use.
                                           Defaults to "sonar-medium-online".
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None

    def is_available(self) -> bool:
        """
        Checks if the client is available to make requests.

        Returns:
            bool: True if an API key is configured, False otherwise.
        """
        return bool(self.api_key)

    async def _ensure(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request to the Perplexity API.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the chat.
            model (Optional[str], optional): The model to use. If not provided,
                                             the default model is used. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.3.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 800.
            extra (Optional[Dict[str, Any]], optional): Extra parameters to pass to
                                                      the API. Defaults to None.

        Raises:
            RuntimeError: If the API key is not configured.

        Returns:
            Dict[str, Any]: A dictionary containing the generated content and the
                            raw API response.
        """
        if not self.is_available():
            raise RuntimeError("Perplexity API key not configured (PERPLEXITY_API_KEY)")

        session = await self._ensure()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
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
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        """
        A convenience method that calls the `chat` method and returns only the
        generated text content.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): The system prompt. Defaults to None.
            model (Optional[str], optional): The model to use. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.3.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 800.

        Returns:
            str: The generated text.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = await self.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return str(resp.get("content", ""))
