from __future__ import annotations

import time
from typing import Optional, Dict

from singularis.llm.perplexity_client import PerplexityClient


class ResearchAdvisor:
    """Provides high-level, real-time strategic advice by querying an external API.

    This class periodically queries a Perplexity model for general Skyrim best
    practices and caches the results. This cached advice can then be injected
    into the AGI's context to inform its decision-making.
    """
    def __init__(self, client: Optional[PerplexityClient] = None, model: Optional[str] = None) -> None:
        """Initializes the ResearchAdvisor.

        Args:
            client: An optional PerplexityClient instance. If not provided, a new one is created.
            model: The name of the Perplexity model to use for generating advice.
        """
        self.client = client or PerplexityClient()
        self.model = model or ("sonar-medium-online")
        self.last_refresh_ts: float = 0.0
        self.cache_text: Optional[str] = None
        self.min_refresh_seconds: float = 600.0

    async def refresh_if_due(self, scene: Optional[str] = None) -> bool:
        """Refreshes the cached strategic advice if the cache is stale.

        This method checks if the minimum refresh interval has passed or if the
        cache is empty. If so, it queries the Perplexity API for updated
        Skyrim strategies, optionally tailoring the query to the current game scene.

        Args:
            scene: The current in-game scene, used to specialize the advice.

        Returns:
            True if the cache was successfully refreshed, False otherwise.
        """
        now = time.time()
        if self.cache_text and (now - self.last_refresh_ts) < self.min_refresh_seconds:
            return False
        if not self.client.is_available():
            return False
        query = (
            "Research best practices, walkthrough strategies, and top tutorials for playing Skyrim. "
            "Provide 6-8 concise, practical bullet points for moment-to-moment decisions. "
            "Cover combat, exploration, stealth, leveling, inventory, and dialogue."
        )
        if scene:
            query += f" Current scene: {scene}. Adapt guidance briefly."
        try:
            text = await self.client.generate_text(
                prompt=query,
                system_prompt="You are a concise game strategy researcher.",
                model=self.model,
                temperature=0.2,
                max_tokens=700,
            )
            text = (text or "").strip()
            if text:
                self.cache_text = text
                self.last_refresh_ts = now
                return True
        except Exception:
            pass
        return False

    def get_context(self) -> Dict[str, str]:
        """Returns the cached research advice as a dictionary for context injection.

        Returns:
            A dictionary containing the cached advice, or an empty dictionary if the cache is empty.
        """
        if not self.cache_text:
            return {}
        return {"research_skyrim_best_practices": self.cache_text}
