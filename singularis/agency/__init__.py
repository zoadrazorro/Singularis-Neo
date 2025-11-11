"""
Autonomous Agency - Phase 6C

True agency: The system forms its own goals and acts autonomously.

Implements:
- Intrinsic motivation (curiosity, competence, coherence)
- Goal formation and hierarchical planning
- Autonomous exploration
- Self-directed learning

Philosophical grounding:
- ETHICA Part III, Prop VI: Conatus = striving to persevere in being
- All beings have intrinsic drive (not just reactive)
- Freedom = acting from one's own nature (not external causes)
"""

from .intrinsic_motivation import (
    IntrinsicMotivation,
    MotivationType,
    CuriosityDrive,
    CompetenceDrive,
    CoherenceDrive
)
from .goal_system import GoalSystem, Goal, GoalStatus, GoalPriority
from .autonomous_orchestrator import AutonomousOrchestrator
from .hierarchical_planner import HierarchicalPlanner, Plan, Action

__all__ = [
    'IntrinsicMotivation',
    'MotivationType',
    'CuriosityDrive',
    'CompetenceDrive',
    'CoherenceDrive',
    'GoalSystem',
    'Goal',
    'GoalStatus',
    'GoalPriority',
    'AutonomousOrchestrator',
    'HierarchicalPlanner',
    'Plan',
    'Action',
]
