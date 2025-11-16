"""Hyperbolic API client for advanced models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import aiohttp
from loguru import logger


class HyperbolicClient:
    """
    An asynchronous client for the Hyperbolic API, which provides access to
    advanced language and vision models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",  # 235B parameter reasoning model
        vlm_model: str = "Qwen/Qwen2.5-VL-72B-Instruct",  # Vision-language model
        base_url: str = "https://api.hyperbolic.xyz/v1",
        timeout: int = 300,  # Increased to 300s (5 minutes) for 235B parameter models
    ) -> None:
        """
        Initializes the HyperbolicClient.

        Args:
            api_key (Optional[str], optional): The Hyperbolic API key. If not
                                             provided, it is read from the
                                             HYPERBOLIC_API_KEY environment
                                             variable. Defaults to None.
            model (str, optional): The language model to use for text generation.
                                   Defaults to "Qwen/Qwen3-235B-A22B-Instruct-2507".
            vlm_model (str, optional): The vision-language model to use for image
                                       analysis. Defaults to "Qwen/Qwen2.5-VL-72B-Instruct".
            base_url (str, optional): The base URL for the Hyperbolic API.
                                      Defaults to "https://api.hyperbolic.xyz/v1".
            timeout (int, optional): The request timeout in seconds. Defaults to 300.
        """
        self.api_key = api_key or os.getenv("HYPERBOLIC_API_KEY")
        self.model = model
        self.vlm_model = vlm_model
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
        Sends a chat completion request to the Hyperbolic API.

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
            raise RuntimeError("Hyperbolic API key not configured (HYPERBOLIC_API_KEY)")

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

        # Use aiohttp.ClientTimeout for proper timeout handling
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
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

    async def analyze_image(
        self,
        prompt: str,
        image,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> str:
        """
        Analyzes an image using the vision-language model.

        Args:
            prompt (str): The prompt for the analysis.
            image: The image to analyze (e.g., a PIL Image object).
            temperature (float, optional): The sampling temperature. Defaults to 0.4.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 2048.

        Raises:
            RuntimeError: If the API key is not configured.

        Returns:
            str: The analysis of the image.
        """
        if not self.is_available():
            raise RuntimeError("Hyperbolic API key not configured (HYPERBOLIC_API_KEY)")

        if image is None:
            logger.warning("analyze_image called with None image")
            return ""

        import base64
        import io

        # Convert image to base64
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            logger.debug(f"Hyperbolic image converted to base64: {len(image_bytes)} bytes")
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            raise

        session = await self._ensure_session()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": self.vlm_model,  # Use VLM model for vision
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(f"Hyperbolic vision request: model={self.vlm_model}, prompt={prompt[:80]}...")

        try:
            # Use aiohttp.ClientTimeout for proper timeout handling
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Hyperbolic returned {resp.status}: {error_text[:500]}")
                resp.raise_for_status()
                data = await resp.json()

            # Extract response
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")

            if not content:
                logger.warning("Hyperbolic returned empty content")
            else:
                logger.info(f"Hyperbolic vision success: {len(content)} chars")

            return content
        except Exception as e:
            logger.error(f"Hyperbolic vision request failed: {type(e).__name__}: {str(e)[:200]}")
            raise

    async def generate_with_image(
        self,
        prompt: str,
        image,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> str:
        """Alias for analyze_image to match MoE orchestrator interface."""
        return await self.analyze_image(
            prompt=prompt,
            image=image,
            temperature=temperature,
            max_tokens=max_tokens,
        )
