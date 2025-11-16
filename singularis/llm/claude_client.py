"""Async client for the Claude API (Anthropic)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import aiohttp


class ClaudeClient:
    """
    An asynchronous client for the Anthropic Claude API.

    This class provides a thin wrapper around the Claude messages API, with
    support for asynchronous requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: int = 60,
    ) -> None:
        """
        Initializes the ClaudeClient.

        Args:
            api_key (Optional[str], optional): The Anthropic API key. If not provided,
                                             it will be read from the ANTHROPIC_API_KEY
                                             environment variable. Defaults to None.
            model (str, optional): The Claude model to use.
                                   Defaults to "claude-sonnet-4-5-20250929".
            base_url (str, optional): The base URL for the Anthropic API.
                                      Defaults to "https://api.anthropic.com/v1".
            timeout (int, optional): The request timeout in seconds. Defaults to 60.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
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
        max_tokens: int = 512,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a single-turn message request to the Claude API.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): The system prompt. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            thinking (Optional[Dict[str, Any]], optional): Extended thinking configuration.
                                                         This is now handled by the model name.
                                                         Defaults to None.

        Raises:
            RuntimeError: If the API key is not configured.

        Returns:
            Dict[str, Any]: A dictionary containing the generated content, any "thinking"
                            content, and the raw API response.
        """

        if not self.is_available():
            raise RuntimeError("Claude API key not configured (ANTHROPIC_API_KEY)")

        session = await self._ensure_session()
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        message_content = [{"type": "text", "text": prompt}]
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        # Extended thinking is enabled via model name (e.g., claude-sonnet-4-20250514)
        # No separate parameter needed

        async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # Anthropic may return a list of content blocks; combine text parts
        content_blocks = data.get("content", [])
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
        thinking_parts = [block.get("thinking", "") for block in content_blocks if block.get("type") == "thinking"]
        combined = "\n".join(part for part in text_parts if part)
        thinking_combined = "\n".join(part for part in thinking_parts if part)

        return {
            "content": combined or data.get("output", ""),
            "thinking": thinking_combined,
            "raw": data,
        }

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        A convenience method that calls the `generate` method and returns only
        the generated text content.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): The system prompt. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 512.

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
