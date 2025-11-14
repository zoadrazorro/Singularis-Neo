from __future__ import annotations

import time
from typing import Optional, Dict

from singularis.llm.openrouter_client import OpenRouterClient


class ResearchAdvisor:
    def __init__(self, client: Optional[OpenRouterClient] = None, model: Optional[str] = None) -> None:
        self.client = client or OpenRouterClient()
        self.model = model or ("openai/gpt-4o")
        self.last_refresh_ts: float = 0.0
        self.cache_text: Optional[str] = None
        self.min_refresh_seconds: float = 600.0

    async def refresh_if_due(self, scene: Optional[str] = None) -> None:
        now = time.time()
        if self.cache_text and (now - self.last_refresh_ts) < self.min_refresh_seconds:
            return
        if not self.client.is_available():
            return
        query = (
            "Research best practices, walkthrough strategies, and top tutorials for playing Skyrim. "
            "Provide 6-8 concise, practical bullet points for moment-to-moment decisions. "
            "Cover combat, exploration, stealth, leveling, inventory, and dialogue."
        )
        if scene:
            query += f" Current scene: {scene}. Adapt guidance briefly."
        messages = [
            {"role": "system", "content": "You are a concise game strategy researcher."},
            {"role": "user", "content": query},
        ]
        try:
            resp = await self.client.chat(messages=messages, model=self.model, temperature=0.2, max_tokens=600)
            text = str(resp.get("content", "")).strip()
            if text:
                self.cache_text = text
                self.last_refresh_ts = now
        except Exception:
            pass

    def get_context(self) -> Dict[str, str]:
        if not self.cache_text:
            return {}
        return {"research_skyrim_best_practices": self.cache_text}
