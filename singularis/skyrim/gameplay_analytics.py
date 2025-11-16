"""
Gameplay analytics for Skyrim AGI.

Aggregates metrics about exploration, combat, quests, and resource
management. Useful for progress dashboards and debugging learning signals.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional


class GameplayAnalytics:
    """Collects, aggregates, and reports high-level gameplay metrics.

    This class tracks various statistics throughout a gameplay session, such as
    actions performed, health status, and rewards received. It is useful for
    generating progress reports, monitoring agent performance, and debugging.
    """

    def __init__(self) -> None:
        """Initializes the GameplayAnalytics collector.

        Sets up data structures for storing metrics and counters.
        """
        self.session_start_time: Optional[float] = None
        self.metrics: Dict[str, Any] = defaultdict(float)
        self.counters: Dict[str, int] = defaultdict(int)
        self.last_action: Optional[str] = None

    def update_state(self, state: Dict[str, Any]) -> None:
        """Updates analytics based on the latest game state.

        Args:
            state: A dictionary containing the current game state, which may
                   include keys like 'location', 'health', and 'quest_count'.
        """
        if state.get("location"):
            self.metrics["last_location"] = state["location"]
        if state.get("health") is not None:
            self.metrics["avg_health"] = self._running_average(
                self.metrics.get("avg_health", state["health"]), state["health"], self.counters["health_samples"]
            )
            self.counters["health_samples"] += 1
        if state.get("quest_count"):
            self.metrics["quest_updates"] += 1

    def record_action(self, action: str) -> None:
        """Records that an action has been performed.

        Increments the counter for the given action and updates the last action taken.

        Args:
            action: The name of the action that was performed.
        """
        self.counters[f"action_{action}"] += 1
        self.last_action = action

    def record_reward(self, reward: float) -> None:
        """Records a reward value and updates the running average reward.

        Args:
            reward: The reward value received.
        """
        self.metrics["avg_reward"] = self._running_average(
            self.metrics.get("avg_reward", reward), reward, self.counters["reward_samples"]
        )
        self.counters["reward_samples"] += 1

    def session_report(self) -> Dict[str, Any]:
        """Generates a summary report of the current session's analytics.

        Returns:
            A dictionary containing key metrics like total actions taken,
            average health, average reward, and quest updates.
        """
        return {
            "actions_taken": sum(v for k, v in self.counters.items() if k.startswith("action_")),
            "avg_health": self.metrics.get("avg_health"),
            "avg_reward": self.metrics.get("avg_reward"),
            "quest_updates": self.metrics.get("quest_updates", 0),
            "last_action": self.last_action,
        }

    def _running_average(self, current: float, new_value: float, n: int) -> float:
        """Calculates a running average.

        Args:
            current: The current average value.
            new_value: The new value to incorporate into the average.
            n: The number of samples already included in the current average.

        Returns:
            The updated average.
        """
        return (current * n + new_value) / (n + 1) if n >= 0 else new_value
