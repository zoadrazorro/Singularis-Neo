"""OpenAI API client for GPT-4o and other models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import aiohttp


class OpenAIClient:
    """
    An asynchronous client for the OpenAI API.

    This class provides a thin wrapper around the OpenAI chat completions API,
    with support for asynchronous requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
    ) -> None:
        """
        Initializes the OpenAIClient.

        Args:
            api_key (Optional[str], optional): The OpenAI API key. If not provided,
                                             it will be read from the OPENAI_API_KEY
                                             environment variable. Defaults to None.
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://api.openai.com/v1".
            timeout (int, optional): The request timeout in seconds. Defaults to 120.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def is_available(self) -> bool:
        """
        Checks if the client is available to make requests.

        Returns:
            bool: True if an API key is configured, False otherwise.
        """
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request to the OpenAI API.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 4096.
            messages (Optional[List[Dict[str, str]]], optional): A list of messages
                                                                   to use instead of the
                                                                   prompt. Defaults to None.

        Raises:
            RuntimeError: If the API key is not configured.

        Returns:
            Dict[str, Any]: A dictionary containing the generated content, usage
                            statistics, and the raw API response.
        """

        if not self.is_available():
            raise RuntimeError("OpenAI API key not configured (OPENAI_API_KEY)")

        session = await self._ensure_session()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # Extract response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        return {
            "content": content,
            "usage": data.get("usage", {}),
            "raw": data,
        }

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        A convenience method that calls the `generate` method and returns only
        the generated text content.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): The system prompt. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 4096.

        Returns:
            str: The generated text.
        """

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.get("content", "")
