from __future__ import annotations

from typing import Optional

from singularis.llm.openrouter_client import OpenRouterClient


class MetaCognitionAdvisor:
    """Provides high-level metacognitive analysis and long-term planning.

    This class uses external large language models to reflect on the AGI's
    overall state. It can generate short, diagnostic reports on the agent's
    performance and propose multi-step, long-term plans to guide its behavior
    over extended periods.
    """
    def __init__(self, client: Optional[OpenRouterClient] = None) -> None:
        """Initializes the MetaCognitionAdvisor.

        Args:
            client: An optional, pre-configured OpenRouterClient for making
                    LLM API calls. If not provided, a new one is created.
        """
        self.client = client or OpenRouterClient()

    async def metameta_report(self, snapshot: str) -> Optional[str]:
        """Generates a high-level, short diagnostic report of the AGI's current state.

        This "meta-meta" report aggregates information from all AGI layers to
        identify trends, risks, and opportunities.

        Args:
            snapshot: A string containing a snapshot of the AGI's current state.

        Returns:
            A formatted string with the diagnostic report, or None if the
            LLM call fails.
        """
        if not self.client.is_available():
            return None
        system = (
            "You are the Meta-Meta-Cognition layer. Aggregate throughput from all layers (perception, world model, "
            "action, consciousness) into a short diagnostic with 5 bullets: coherence trend, risks, opportunities, "
            "stuck signals, and one concrete next adjustment. Keep under 140 words."
        )
        prompt = f"SNAPSHOT INPUT\n---\n{snapshot}\n---\nProduce the diagnostic now."
        try:
            text = await self.client.generate_text(
                prompt=prompt,
                model="openai/gpt-4o",
                system_prompt=system,
                temperature=0.2,
                max_tokens=400,
            )
            return text.strip() if text else None
        except Exception:
            return None

    async def deepseek_long_term_plan(self, snapshot: str) -> Optional[str]:
        """Generates a concrete, 3-step long-term plan for the next 30-60 minutes.

        This uses an LLM to propose a strategic plan focused on character
        progression and safety.

        Args:
            snapshot: A string containing a summary of the AGI's current state.

        Returns:
            A formatted string with the 3-step plan, or None if the LLM call fails.
        """
        if not self.client.is_available():
            return None
        system = (
            "You are the Long-Term Planning layer. Propose a 3-step plan for the next 30-60 minutes of Skyrim "
            "play focused on progression and safety, in 90-120 words. Be concrete."
        )
        prompt = f"STATE SUMMARY\n---\n{snapshot}\n---\nPropose the plan now."
        try:
            text = await self.client.generate_text(
                prompt=prompt,
                model="deepseek/deepseek-chat-v3.1:free",
                system_prompt=system,
                temperature=0.4,
                max_tokens=500,
            )
            return text.strip() if text else None
        except Exception:
            return None
