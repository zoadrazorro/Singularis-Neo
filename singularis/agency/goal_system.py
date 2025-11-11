"""
Goal System - Autonomous Goal Formation

Forms and pursues goals autonomously based on intrinsic motivation.

Key capabilities:
- Generate goals from drives (curiosity → "explore X")
- Prioritize goals
- Track goal progress
- Hierarchical goals (sub-goals)

Philosophical grounding:
- ETHICA Part III: Desire (appetitus) arises from nature
- Goals emerge from conatus (striving for coherence)
"""

import time
import uuid
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class GoalPriority(Enum):
    """Priority levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class Goal:
    """
    A goal to achieve.

    Attributes:
        id: Unique identifier
        description: What to achieve
        motivation_source: Which drive generated this goal
        priority: How important
        status: Current status
        progress: Progress toward completion [0, 1]
        sub_goals: List of sub-goal IDs
        parent_goal: Parent goal ID (if this is a sub-goal)
        created_at: Creation timestamp
        deadline: Optional deadline
        success_criteria: How to know if completed
        context: Additional context
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    motivation_source: str = "unknown"  # curiosity, competence, coherence
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    sub_goals: List[str] = field(default_factory=list)
    parent_goal: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if goal is completed."""
        return self.status == GoalStatus.COMPLETED

    def is_active(self) -> bool:
        """Check if goal is currently being pursued."""
        return self.status == GoalStatus.ACTIVE

    def is_overdue(self) -> bool:
        """Check if past deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline


class GoalSystem:
    """
    Manages autonomous goal formation and pursuit.

    Generates goals from intrinsic motivation drives.
    Prioritizes and schedules goals.
    Tracks progress and completion.
    """

    def __init__(self, max_active_goals: int = 3):
        """
        Initialize goal system.

        Args:
            max_active_goals: Maximum goals to pursue simultaneously
        """
        self.max_active_goals = max_active_goals

        # All goals
        self.goals: Dict[str, Goal] = {}

        # Active goal tracking
        self.active_goals: Set[str] = set()

        # Goal generation rules
        self.goal_generators = {
            'curiosity': self._generate_exploration_goal,
            'competence': self._generate_mastery_goal,
            'coherence': self._generate_coherence_goal,
        }

    def generate_goal(
        self,
        motivation_source: str,
        context: Dict[str, Any]
    ) -> Goal:
        """
        Generate new goal from motivation drive.

        Args:
            motivation_source: Which drive ("curiosity", "competence", "coherence")
            context: Context for goal generation

        Returns:
            Generated goal
        """
        if motivation_source in self.goal_generators:
            goal = self.goal_generators[motivation_source](context)
        else:
            # Default goal
            goal = Goal(
                description=f"Goal from {motivation_source}",
                motivation_source=motivation_source,
                context=context
            )

        # Add to system
        self.goals[goal.id] = goal
        return goal

    def _generate_exploration_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate goal from curiosity drive."""
        uncertain_area = context.get('uncertain_area', 'unknown_area')

        return Goal(
            description=f"Explore {uncertain_area}",
            motivation_source="curiosity",
            priority=GoalPriority.MEDIUM,
            success_criteria={'uncertainty_reduction': 0.3},
            context=context
        )

    def _generate_mastery_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate goal from competence drive."""
        skill = context.get('skill', 'general_skill')

        return Goal(
            description=f"Master {skill}",
            motivation_source="competence",
            priority=GoalPriority.HIGH,
            success_criteria={'competence_level': 0.8},
            context=context
        )

    def _generate_coherence_goal(self, context: Dict[str, Any]) -> Goal:
        """Generate goal from coherence drive."""
        area = context.get('area', 'general')

        return Goal(
            description=f"Increase coherence in {area}",
            motivation_source="coherence",
            priority=GoalPriority.CRITICAL,  # Core drive
            success_criteria={'delta_coherence': 0.05},
            context=context
        )

    def prioritize_goals(self) -> List[Goal]:
        """
        Prioritize all pending goals.

        Returns:
            Sorted list of goals (highest priority first)
        """
        pending = [g for g in self.goals.values() if g.status == GoalStatus.PENDING]

        # Sort by priority, then by creation time
        pending.sort(
            key=lambda g: (g.priority.value, -g.created_at),
            reverse=True
        )

        return pending

    def activate_next_goals(self):
        """Activate next goals up to max_active_goals."""
        while len(self.active_goals) < self.max_active_goals:
            prioritized = self.prioritize_goals()
            if not prioritized:
                break

            # Activate highest priority goal
            goal = prioritized[0]
            self.activate_goal(goal.id)

    def activate_goal(self, goal_id: str):
        """Mark goal as active."""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.ACTIVE
            self.active_goals.add(goal_id)

    def update_progress(self, goal_id: str, progress: float):
        """Update goal progress."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.progress = min(1.0, progress)

            # Auto-complete if progress = 1.0
            if goal.progress >= 1.0:
                self.complete_goal(goal_id)

    def complete_goal(self, goal_id: str):
        """Mark goal as completed."""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.COMPLETED
            self.goals[goal_id].progress = 1.0
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)

    def fail_goal(self, goal_id: str, reason: str = ""):
        """Mark goal as failed."""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.FAILED
            self.goals[goal_id].context['failure_reason'] = reason
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return [self.goals[gid] for gid in self.active_goals if gid in self.goals]

    def get_stats(self) -> Dict[str, Any]:
        """Get goal system statistics."""
        by_status = {}
        by_source = {}

        for goal in self.goals.values():
            by_status[goal.status.value] = by_status.get(goal.status.value, 0) + 1
            by_source[goal.motivation_source] = by_source.get(goal.motivation_source, 0) + 1

        return {
            'total_goals': len(self.goals),
            'active_goals': len(self.active_goals),
            'by_status': by_status,
            'by_source': by_source,
        }