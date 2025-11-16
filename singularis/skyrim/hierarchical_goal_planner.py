"""
Hierarchical goal planner for Skyrim AGI.

Maintains strategic, tactical, and immediate objectives so the agent can
balance long-term quest progress with short-term survival and exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoalTier:
    """Represents a single tier in the goal hierarchy.

    Each tier maintains a list of pending goals and a single active goal.

    Attributes:
        goals: A list of goal strings waiting to be activated.
        active_goal: The current goal for this tier.
    """
    goals: List[str] = field(default_factory=list)
    active_goal: Optional[str] = None

    def promote(self) -> Optional[str]:
        """Promotes the next goal from the list to be the active goal.

        Returns:
            The new active goal string, or None if no goals are available.
        """
        if not self.goals:
            return None
        self.active_goal = self.goals.pop(0)
        return self.active_goal


class HierarchicalGoalPlanner:
    """A three-tiered planner for managing strategic, tactical, and immediate goals.

    This planner helps the AGI balance long-term objectives (e.g., completing the
    main quest), medium-term objectives (e.g., clearing a dungeon), and short-term
    needs (e.g., winning a single combat encounter).
    """

    def __init__(self) -> None:
        """Initializes the three tiers of the goal hierarchy."""
        self.strategic = GoalTier()
        self.tactical = GoalTier()
        self.immediate = GoalTier()
        self.last_scene: Optional[str] = None

    def update_state(self, state: Dict[str, Any], scene: str) -> None:
        """Updates the planner with the latest game state and refreshes goals.

        Args:
            state: A dictionary representing the current game state.
            scene: A string describing the current scene or context.
        """
        self.last_scene = scene
        self._refresh_goals(state)

    def _refresh_goals(self, state: Dict[str, Any]) -> None:
        """Refreshes the goal lists in each tier based on the current state.

        If a tier has no pending goals, it attempts to infer new ones. If a tier
        has no active goal, it promotes the next one from its list.

        Args:
            state: The current game state.
        """
        if not self.strategic.goals:
            self.strategic.goals.extend(self._infer_strategic_goals(state))
        if not self.strategic.active_goal:
            self.strategic.promote()
        if not self.tactical.goals:
            self.tactical.goals.extend(self._infer_tactical_goals(state))
        if not self.tactical.active_goal:
            self.tactical.promote()
        if not self.immediate.goals:
            self.immediate.goals.extend(self._infer_immediate_goals(state))
        if not self.immediate.active_goal:
            self.immediate.promote()

    def _infer_strategic_goals(self, state: Dict[str, Any]) -> List[str]:
        """Infers long-term, strategic goals based on the game state.

        Args:
            state: The current game state.

        Returns:
            A list of strategic goal strings.
        """
        goals: List[str] = []
        if state.get("story_progress", 0) < 0.3:
            goals.append("Advance main quest")
        if state.get("skills", {}).get("Smithing", 0) < 40:
            goals.append("Improve crafting skills")
        return goals or ["Strengthen character build"]

    def _infer_tactical_goals(self, state: Dict[str, Any]) -> List[str]:
        """Infers medium-term, tactical goals based on the current situation.

        Args:
            state: The current game state.

        Returns:
            A list of tactical goal strings.
        """
        goals: List[str] = []
        if state.get("health", 100) < 50:
            goals.append("Recover health")
        if state.get("enemies_nearby", 0) > 0:
            goals.append("Survive current encounter")
        if state.get("quest_count", 0) > 0:
            goals.append("Progress active quest")
        return goals or ["Explore nearby area"]

    def _infer_immediate_goals(self, state: Dict[str, Any]) -> List[str]:
        """Infers short-term, immediate goals based on the immediate context.

        Args:
            state: The current game state.

        Returns:
            A list of immediate goal strings.
        """
        goals: List[str] = []
        if state.get("in_combat"):
            goals.append("Win combat")
        elif state.get("in_menu"):
            goals.append("Manage inventory")
        else:
            goals.append("Scout surroundings")
        return goals

    def consume_immediate_goal(self) -> Optional[str]:
        """Consumes the current immediate goal and promotes the next one.

        This is typically called after an action is taken to fulfill the goal.
        If no goals are left, it will attempt to infer new ones.

        Returns:
            The new active immediate goal.
        """
        goal = self.immediate.promote()
        if goal is None:
            self.immediate.goals = self._infer_immediate_goals({})
            goal = self.immediate.promote()
        return goal

    def snapshot(self) -> Dict[str, Any]:
        """Takes a snapshot of the current active goals in all tiers.

        Returns:
            A dictionary mapping each tier (strategic, tactical, immediate)
            to its currently active goal.
        """
        return {
            "strategic": self.strategic.active_goal,
            "tactical": self.tactical.active_goal,
            "immediate": self.immediate.active_goal,
        }
