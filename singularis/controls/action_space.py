"""Clean, unified action vocabulary for motor control."""

from enum import Enum, auto


class HighLevelAction(Enum):
    """
    Canonical set of high-level actions the AGI can perform.
    
    Everything upstream (RL, planner, LLM) talks in these terms.
    Everything downstream (key presses, controller) implements these.
    """
    
    # Idle
    IDLE = auto()
    
    # Looking / Camera
    LOOK_AROUND = auto()
    TURN_LEFT_SMALL = auto()
    TURN_RIGHT_SMALL = auto()
    TURN_LEFT_LARGE = auto()
    TURN_RIGHT_LARGE = auto()
    
    # Basic Movement
    STEP_FORWARD = auto()
    STEP_BACKWARD = auto()
    STRAFE_LEFT = auto()
    STRAFE_RIGHT = auto()
    SPRINT_FORWARD = auto()
    JUMP = auto()
    
    # Tactical Movement
    APPROACH_TARGET = auto()
    RETREAT_FROM_TARGET = auto()
    CIRCLE_LEFT = auto()
    CIRCLE_RIGHT = auto()
    
    # Combat
    QUICK_ATTACK = auto()
    POWER_ATTACK = auto()
    BLOCK = auto()
    DODGE = auto()
    BASH = auto()
    USE_SHOUT = auto()
    
    # Interaction
    INTERACT = auto()
    ACTIVATE = auto()
    
    # Menus
    OPEN_INVENTORY = auto()
    OPEN_MAP = auto()
    CLOSE_MENU = auto()
    
    # Items
    USE_POTION_HEALTH = auto()
    USE_POTION_STAMINA = auto()
    USE_POTION_MAGICKA = auto()
    EQUIP_BEST_WEAPON = auto()
    EQUIP_BEST_ARMOR = auto()
    
    # Stealth
    SNEAK = auto()
    UNSNEAK = auto()


def action_from_string(action_str: str) -> HighLevelAction:
    """Convert string action name to HighLevelAction enum."""
    action_map = {
        'idle': HighLevelAction.IDLE,
        'look_around': HighLevelAction.LOOK_AROUND,
        'move_forward': HighLevelAction.STEP_FORWARD,
        'step_forward': HighLevelAction.STEP_FORWARD,
        'move_backward': HighLevelAction.STEP_BACKWARD,
        'step_backward': HighLevelAction.STEP_BACKWARD,
        'turn_left': HighLevelAction.TURN_LEFT_SMALL,
        'turn_right': HighLevelAction.TURN_RIGHT_SMALL,
        'jump': HighLevelAction.JUMP,
        'attack': HighLevelAction.QUICK_ATTACK,
        'quick_attack': HighLevelAction.QUICK_ATTACK,
        'power_attack': HighLevelAction.POWER_ATTACK,
        'block': HighLevelAction.BLOCK,
        'dodge': HighLevelAction.DODGE,
        'retreat': HighLevelAction.RETREAT_FROM_TARGET,
        'activate': HighLevelAction.ACTIVATE,
        'interact': HighLevelAction.INTERACT,
        'inventory': HighLevelAction.OPEN_INVENTORY,
        'map': HighLevelAction.OPEN_MAP,
        'close_menu': HighLevelAction.CLOSE_MENU,
        'heal': HighLevelAction.USE_POTION_HEALTH,
        'sneak': HighLevelAction.SNEAK,
    }
    
    return action_map.get(action_str.lower(), HighLevelAction.STEP_FORWARD)
