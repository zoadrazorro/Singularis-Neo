"""Async client for Google Gemini API."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Optional

import aiohttp
from loguru import logger


class GeminiClient:
    """Minimal async wrapper around the Gemini generative API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 90,  # Increased from 60s for vision requests
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_output_tokens: int = 768,
    ) -> str:
        """Text-only generation helper."""

        response = await self._generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return self._extract_text(response)

    async def analyze_image(
        self,
        prompt: str,
        image,
        temperature: float = 0.4,
        max_output_tokens: int = 768,
        max_retries: int = 3,
    ) -> str:
        """Send prompt + image to Gemini for multimodal analysis with retries."""

        if not self.is_available():
            raise RuntimeError("Gemini API key not configured (GEMINI_API_KEY)")

        if image is None:
            logger.warning("analyze_image called with None image")
            return ""

        # Convert image to base64
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            inline_data = {
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
            logger.debug(f"Image converted to base64: {len(image_bytes)} bytes")
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            raise

        # Retry logic for better reliability
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini vision attempt {attempt + 1}/{max_retries}: prompt={prompt[:80]}...")
                
                response = await self._generate_content(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {"inline_data": inline_data},
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                
                logger.debug(f"Gemini vision response received: {response.keys() if isinstance(response, dict) else type(response)}")
                result = self._extract_text(response)
                
                # Validate response is not empty
                if not result or len(result.strip()) == 0:
                    logger.warning(f"Gemini returned empty response on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Gemini failed after {max_retries} attempts - all returned empty")
                        await self.close()
                        return ""  # Return empty on final attempt
                
                logger.info(f"Gemini vision success: {len(result)} chars (attempt {attempt + 1}/{max_retries})")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini vision attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {str(e)[:200]}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Gemini vision failed after {max_retries} attempts: {type(last_error).__name__}")
                    await self.close()
                    raise last_error
        
        # Should not reach here, but just in case
        if last_error:
            await self.close()
            raise last_error
        return ""

    async def _generate_content(
        self,
        contents: Any,
        temperature: float,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Gemini API key not configured (GEMINI_API_KEY)")

        session = await self._ensure_session()
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        async with session.post(url, json=payload, timeout=self.timeout) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _extract_text(self, response: Dict[str, Any]) -> str:
        candidates = response.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(texts).strip()
