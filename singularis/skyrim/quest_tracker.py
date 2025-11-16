"""
Quest tracking utilities for Skyrim AGI.

Maintains lightweight quest knowledge, parses journal snippets, and provides
prioritized recommendations for advancing objectives.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuestObjective:
    """Represents a single objective within a quest.

    Attributes:
        description: The text of the objective.
        completed: A boolean indicating if the objective is complete.
        location: The location associated with the objective, if known.
        hints: A list of hints or additional details.
    """
    description: str
    completed: bool = False
    location: Optional[str] = None
    hints: List[str] = field(default_factory=list)


@dataclass
class QuestRecord:
    """Represents the complete record of a single quest.

    Attributes:
        name: The name of the quest.
        objectives: A list of QuestObjective objects for this quest.
        priority: A calculated priority score for the quest (0.0 to 1.0).
        stage: The current stage or status of the quest.
        last_update_cycle: The game cycle when this quest was last updated.
    """
    name: str
    objectives: List[QuestObjective] = field(default_factory=list)
    priority: float = 0.5
    stage: str = "unknown"
    last_update_cycle: int = 0


class QuestTracker:
    """Tracks the state of active and completed quests for the AGI.

    This class parses quest information from the game's journal, maintains a
    record of all known quests and their objectives, and uses heuristics and an
    optional LLM to prioritize quests and suggest next steps.
    """

    def __init__(self) -> None:
        """Initializes the QuestTracker."""
        self.active_quests: Dict[str, QuestRecord] = {}
        self.completed_quests: Dict[str, QuestRecord] = {}
        self.quest_patterns = self._load_quest_patterns()
        self._llm_interface: Optional[Any] = None

    def set_llm_interface(self, llm_interface: Any) -> None:
        """Sets the LLM interface to be used for advanced journal analysis.

        Args:
            llm_interface: An object with a compatible `generate` method.
        """
        self._llm_interface = llm_interface

    def _load_quest_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Loads predefined patterns and metadata for known quests.

        Returns:
            A dictionary of known quest patterns.
        """
        return {
            "Bleak Falls Barrow": {
                "keywords": ["Golden Claw", "Dragonstone", "Whiterun"],
                "reward_value": 0.6,
                "difficulty": 0.3,
            },
            "Dragon Rising": {
                "keywords": ["Western Watchtower", "Dragon", "Whiterun"],
                "reward_value": 0.8,
                "difficulty": 0.6,
            },
        }

    async def analyze_quest_journal(
        self,
        journal_text: str,
        default_location: Optional[str] = None,
        cycle: int = 0,
    ) -> Optional[QuestRecord]:
        """Analyzes text from the quest journal to update the quest state.

        This method uses a combination of heuristic parsing and an optional LLM
        to extract the quest name, objective, and other details from raw journal text.

        Args:
            journal_text: The text content from the quest journal.
            default_location: A fallback location to use if one isn't found in the text.
            cycle: The current game cycle number.

        Returns:
            The updated QuestRecord, or None if parsing fails.
        """
        if not journal_text:
            return None

        parsed = self._heuristic_parse(journal_text, default_location)

        if self._llm_interface:
            prompt = (
                "You are assisting an autonomous Skyrim agent. Analyze the following "
                "quest journal excerpt and extract: quest name, current objective, "
                "recommended next action, and whether the objective is near completion.\n\n"
                f"Journal:\n{journal_text}\n\n"
                "Respond as JSON with keys quest_name, objective, next_step, stage, "
                "location, reward_clues."
            )
            try:
                response = await asyncio.wait_for(
                    self._llm_interface.generate(prompt=prompt, max_tokens=256),
                    timeout=10.0,
                )
                parsed = self._merge_llm_result(parsed, response.get("content", ""))
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

        if not parsed:
            return None

        record = self._update_record(parsed, cycle)
        return record

    def _heuristic_parse(self, text: str, fallback_location: Optional[str]) -> Optional[Dict[str, Any]]:
        """Performs a simple, rule-based parsing of the journal text.

        Args:
            text: The journal text.
            fallback_location: A default location if one cannot be parsed.

        Returns:
            A dictionary of parsed information, or None.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        quest_name = lines[0]
        objective = next((line for line in lines[1:] if line.lower().startswith("objective")), None)
        if objective:
            objective = objective.replace("Objective:", "").strip()
        location = fallback_location
        for line in lines:
            if "at" in line.lower() and len(line.split()) <= 12:
                location = line.split(" at ")[-1].strip()
        return {
            "quest_name": quest_name,
            "objective": objective or lines[-1],
            "location": location,
            "stage": self._infer_stage(lines),
        }

    def _merge_llm_result(self, heuristic: Optional[Dict[str, Any]], llm_json: str) -> Optional[Dict[str, Any]]:
        """Merges the structured output from an LLM with the heuristically parsed data.

        The LLM's output is prioritized if it is valid JSON.

        Args:
            heuristic: The dictionary from the heuristic parse.
            llm_json: The JSON string response from the LLM.

        Returns:
            The merged dictionary of parsed information.
        """
        if not heuristic:
            heuristic = {}
        result = dict(heuristic)
        if not llm_json:
            return result
        import json

        try:
            data = json.loads(llm_json)
        except json.JSONDecodeError:
            return result

        for key in ("quest_name", "objective", "next_step", "stage", "location", "reward_clues"):
            value = data.get(key)
            if value:
                result[key] = value
        return result

    def _update_record(self, parsed: Dict[str, Any], cycle: int) -> QuestRecord:
        """Updates or creates a QuestRecord based on parsed information.

        Args:
            parsed: The dictionary of parsed quest data.
            cycle: The current game cycle.

        Returns:
            The newly created or updated QuestRecord.
        """
        quest_name = parsed.get("quest_name", "Unknown Quest")
        record = self.active_quests.get(quest_name) or QuestRecord(name=quest_name)
        objective_text = parsed.get("objective")
        if objective_text:
            if not record.objectives or record.objectives[-1].description != objective_text:
                record.objectives.append(QuestObjective(description=objective_text, location=parsed.get("location")))
        record.stage = parsed.get("stage", record.stage)
        record.priority = self.get_quest_priority(quest_name)
        record.last_update_cycle = cycle
        self.active_quests[quest_name] = record
        return record

    def get_quest_priority(self, quest_name: str) -> float:
        """Calculates a priority score for a quest based on predefined patterns.

        The priority is a simple function of the quest's expected reward and difficulty.

        Args:
            quest_name: The name of the quest.

        Returns:
            A priority score between 0.1 and 1.0.
        """
        pattern = self.quest_patterns.get(quest_name)
        if not pattern:
            return 0.5
        reward = pattern.get("reward_value", 0.5)
        difficulty = pattern.get("difficulty", 0.5)
        return max(0.1, min(1.0, reward - 0.3 * difficulty))

    def mark_completed(self, quest_name: str) -> None:
        """Moves a quest from the active list to the completed list.

        Args:
            quest_name: The name of the quest to mark as completed.
        """
        if quest_name in self.active_quests:
            record = self.active_quests.pop(quest_name)
            self.completed_quests[quest_name] = record

    def get_high_priority_objectives(self) -> List[QuestObjective]:
        """Gets a list of the current, uncompleted objectives from the highest-priority quests.

        Returns:
            A list of QuestObjective objects.
        """
        sorted_quests = sorted(self.active_quests.values(), key=lambda q: q.priority, reverse=True)
        objectives: List[QuestObjective] = []
        for quest in sorted_quests:
            for objective in quest.objectives:
                if not objective.completed:
                    objectives.append(objective)
                    break
        return objectives
