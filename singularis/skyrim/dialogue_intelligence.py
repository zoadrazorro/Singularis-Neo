"""
Dialogue intelligence utilities for Skyrim AGI.

Analyzes dialogue options, tracks NPC relationships, and supplies
recommendations that balance quest progress with relationship gains.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RelationshipRecord:
    """Stores relationship data for a single NPC.

    Attributes:
        disposition: A float from -1.0 (hostile) to 1.0 (friendly) representing
                     the NPC's disposition towards the agent.
        interactions: The total number of interactions with this NPC.
        preferred_topics: A list of topics the NPC has responded to positively.
    """
    disposition: float = 0.0
    interactions: int = 0
    preferred_topics: List[str] = None

    def __post_init__(self) -> None:
        """Initializes preferred_topics to an empty list if not provided."""
        if self.preferred_topics is None:
            self.preferred_topics = []


class DialogueIntelligence:
    """A helper class for making intelligent dialogue decisions.

    This class tracks relationships with NPCs, analyzes available dialogue options
    using both heuristics and an optional LLM, and recommends the best choice
    to achieve goals like quest progression and relationship building.
    """

    def __init__(self) -> None:
        """Initializes the DialogueIntelligence system.

        Sets up an empty dictionary to store NPC relationship records.
        """
        self.relationships: Dict[str, RelationshipRecord] = {}
        self._llm_interface: Optional[Any] = None

    def set_llm_interface(self, llm_interface: Any) -> None:
        """Sets the LLM interface for advanced dialogue analysis.

        Args:
            llm_interface: An object capable of handling LLM generate calls.
        """
        self._llm_interface = llm_interface

    async def analyze_dialogue_options(
        self,
        npc_name: str,
        options: List[str],
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Analyzes a list of dialogue options and returns the recommended choice.

        First, a heuristic-based score is calculated for each option. If an
        LLM interface is available, it is queried for a more nuanced
        recommendation, which can override the heuristic choice.

        Args:
            npc_name: The name of the NPC the agent is talking to.
            options: A list of dialogue option strings.
            context: Optional context about the current conversation.

        Returns:
            The recommended dialogue option string, or None if no options are given.
        """
        if not options:
            return None

        # Heuristic preference: choose options that progress quests or improve relations.
        relationship = self.relationships.get(npc_name, RelationshipRecord())
        best_option = max(options, key=lambda opt: self._score_option(opt, relationship))

        if self._llm_interface:
            prompt = (
                f"NPC: {npc_name}\n"
                f"Context: {context or 'None'}\n"
                "Options:\n" + "\n".join(f"- {opt}" for opt in options) + "\n\n"
                "Recommend the option that provides the best outcome for quest progression "
                "or relationship building. Respond with the exact option text."
            )
            try:
                response = await asyncio.wait_for(
                    self._llm_interface.generate(prompt=prompt, max_tokens=64),
                    timeout=6.0,
                )
                suggestion = response.get("content", "").strip()
                for option in options:
                    if option.lower() in suggestion.lower():
                        best_option = option
                        break
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

        return best_option

    def _score_option(self, option: str, relationship: RelationshipRecord) -> float:
        """Calculates a heuristic score for a single dialogue option.

        Args:
            option: The dialogue option text.
            relationship: The RelationshipRecord for the current NPC.

        Returns:
            A float score, where higher is better.
        """
        score = 0.0
        lower = option.lower()
        if "help" in lower or "quest" in lower:
            score += 0.6
        if "thank" in lower or "friend" in lower:
            score += 0.3
        for topic in relationship.preferred_topics:
            if topic in lower:
                score += 0.2
        return score

    def update_relationship(self, npc_name: str, outcome: str) -> None:
        """Updates the relationship record for an NPC based on a dialogue outcome.

        Args:
            npc_name: The name of the NPC.
            outcome: The outcome of the interaction ('positive' or 'negative').
        """
        record = self.relationships.setdefault(npc_name, RelationshipRecord())
        record.interactions += 1
        if outcome == "positive":
            record.disposition = min(1.0, record.disposition + 0.1)
        elif outcome == "negative":
            record.disposition = max(-1.0, record.disposition - 0.1)

    def get_relationship_status(self, npc_name: str) -> RelationshipRecord:
        """Retrieves the current relationship record for a given NPC.

        Args:
            npc_name: The name of the NPC.

        Returns:
            The RelationshipRecord for the NPC. Returns a new, default record
            if the NPC has not been interacted with before.
        """
        return self.relationships.get(npc_name, RelationshipRecord())
