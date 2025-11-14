"""
Motor Control Layer - Hands & Feet for the AGI

This layer bridges abstract decisions (from BeingState/CoherenceEngine) to
concrete game actions. Provides modular, testable, non-embarrassing behavior.

Components:
- ActionSpace: Clean action vocabulary
- AffordanceExtractor: What can I do now?
- MotorController: Press the buttons
- ReflexController: Emergency overrides
- Navigator: Smart exploration
- CombatController: Heuristic combat
- MenuHandler: Anti-stuck menus
"""

from .action_space import HighLevelAction
from .affordances import Affordance, AffordanceExtractor
from .motor_controller import MotorController
from .reflex_controller import ReflexController
from .navigator import Navigator
from .combat_controller import CombatController
from .menu_handler import MenuHandler

__all__ = [
    'HighLevelAction',
    'Affordance',
    'AffordanceExtractor',
    'MotorController',
    'ReflexController',
    'Navigator',
    'CombatController',
    'MenuHandler',
]
