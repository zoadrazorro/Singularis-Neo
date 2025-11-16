"""
Enhanced Action System - 60% More Granularity

Expands from ~20 basic actions to ~50+ granular actions with:
- Fine-grained movement (walk, jog, sprint, strafe)
- Combat variations (light/heavy attack, power attack by direction)
- Stealth actions (crouch-walk, pickpocket, backstab)
- Magic variations (spell by hand, dual-cast)
- Interaction granularity (activate, take, equip, drop)
- Contextual actions (mount, dismount, sit, sleep)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


class ActionCategory(Enum):
    """High-level action categories."""
    MOVEMENT = "movement"
    COMBAT = "combat"
    STEALTH = "stealth"
    MAGIC = "magic"
    INTERACTION = "interaction"
    INVENTORY = "inventory"
    SOCIAL = "social"
    UTILITY = "utility"


class EnhancedActionType(Enum):
    """
    Enhanced action types with 60% more granularity.
    
    Expanded from ~20 to ~50+ actions.
    """
    
    # ========================================
    # MOVEMENT (12 actions, was 4)
    # ========================================
    
    # Basic movement
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    
    # Speed variations
    WALK_FORWARD = "walk_forward"  # Slow, quiet
    JOG_FORWARD = "jog_forward"    # Normal speed
    SPRINT_FORWARD = "sprint_forward"  # Fast, drains stamina
    
    # Strafing
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    
    # Vertical
    JUMP = "jump"
    CROUCH = "crouch"
    STAND = "stand"
    
    # ========================================
    # COMBAT (16 actions, was 6)
    # ========================================
    
    # Melee attacks
    LIGHT_ATTACK = "light_attack"  # Fast, low damage
    HEAVY_ATTACK = "heavy_attack"  # Slow, high damage
    
    # Power attacks (directional)
    POWER_ATTACK_FORWARD = "power_attack_forward"
    POWER_ATTACK_BACKWARD = "power_attack_backward"
    POWER_ATTACK_LEFT = "power_attack_left"
    POWER_ATTACK_RIGHT = "power_attack_right"
    POWER_ATTACK_STANDING = "power_attack_standing"
    
    # Ranged
    DRAW_BOW = "draw_bow"
    RELEASE_ARROW = "release_arrow"
    AIM_BOW = "aim_bow"
    
    # Defense
    BLOCK = "block"
    BASH = "bash"  # Shield bash
    DODGE_ROLL = "dodge_roll"
    PARRY = "parry"
    
    # Dual wield
    DUAL_ATTACK = "dual_attack"  # Attack with both weapons
    
    # Special
    SHEATHE_WEAPON = "sheathe_weapon"
    DRAW_WEAPON = "draw_weapon"
    
    # ========================================
    # STEALTH (8 actions, was 2)
    # ========================================
    
    SNEAK = "sneak"  # Enter sneak mode
    SNEAK_FORWARD = "sneak_forward"  # Move while sneaking
    SNEAK_BACKWARD = "sneak_backward"
    
    BACKSTAB = "backstab"  # Sneak attack with dagger
    PICKPOCKET = "pickpocket"
    LOCKPICK = "lockpick"
    
    HIDE_IN_SHADOWS = "hide_in_shadows"  # Stay still in dark area
    DISTRACT = "distract"  # Throw object to distract
    
    # ========================================
    # MAGIC (10 actions, was 3)
    # ========================================
    
    # Casting
    CAST_LEFT_HAND = "cast_left_hand"
    CAST_RIGHT_HAND = "cast_right_hand"
    DUAL_CAST = "dual_cast"  # Cast with both hands
    
    # Spell types (contextual based on equipped spell)
    CAST_DESTRUCTION = "cast_destruction"
    CAST_RESTORATION = "cast_restoration"
    CAST_ALTERATION = "cast_alteration"
    CAST_CONJURATION = "cast_conjuration"
    CAST_ILLUSION = "cast_illusion"
    
    # Shouts
    USE_SHOUT = "use_shout"
    
    # Equipment
    EQUIP_SPELL_LEFT = "equip_spell_left"
    EQUIP_SPELL_RIGHT = "equip_spell_right"
    
    # ========================================
    # INTERACTION (12 actions, was 2)
    # ========================================
    
    # Basic
    ACTIVATE = "activate"  # Generic activate
    
    # Specific interactions
    OPEN_DOOR = "open_door"
    CLOSE_DOOR = "close_door"
    OPEN_CONTAINER = "open_container"
    TAKE_ITEM = "take_item"
    TAKE_ALL = "take_all"
    
    TALK_TO_NPC = "talk_to_npc"
    TRADE_WITH_NPC = "trade_with_npc"
    
    PULL_LEVER = "pull_lever"
    PUSH_BUTTON = "push_button"
    
    READ_BOOK = "read_book"
    SIT_DOWN = "sit_down"
    
    # ========================================
    # INVENTORY (8 actions, was 1)
    # ========================================
    
    OPEN_INVENTORY = "open_inventory"
    CLOSE_INVENTORY = "close_inventory"
    
    EQUIP_ITEM = "equip_item"
    UNEQUIP_ITEM = "unequip_item"
    DROP_ITEM = "drop_item"
    
    USE_POTION = "use_potion"
    USE_FOOD = "use_food"
    
    FAVORITE_ITEM = "favorite_item"
    
    # ========================================
    # SOCIAL (6 actions, was 0)
    # ========================================
    
    GREET_NPC = "greet_npc"
    PERSUADE = "persuade"
    INTIMIDATE = "intimidate"
    BRIBE = "bribe"
    
    FOLLOW_NPC = "follow_npc"
    WAIT_FOR_NPC = "wait_for_npc"
    
    # ========================================
    # UTILITY (8 actions, was 2)
    # ========================================
    
    WAIT = "wait"
    WAIT_1_HOUR = "wait_1_hour"
    WAIT_UNTIL_MORNING = "wait_until_morning"
    
    SLEEP = "sleep"
    
    FAST_TRAVEL = "fast_travel"
    
    MOUNT_HORSE = "mount_horse"
    DISMOUNT_HORSE = "dismount_horse"
    
    QUICK_SAVE = "quick_save"


@dataclass
class ActionAffordance:
    """
    Describes when an action is available and what it requires.
    
    Affordances define the "action space" at any given moment.
    """
    action_type: EnhancedActionType
    category: ActionCategory
    
    # Prerequisites
    requires_combat: bool = False
    requires_stealth: bool = False
    requires_target: bool = False
    requires_item: Optional[str] = None  # e.g., "weapon", "spell", "lockpick"
    requires_stamina: float = 0.0  # 0-1, amount needed
    requires_magicka: float = 0.0
    
    # Context
    available_in_menu: bool = False
    available_in_dialogue: bool = False
    available_while_moving: bool = True
    available_while_in_air: bool = False
    
    # Effects
    drains_stamina: float = 0.0
    drains_magicka: float = 0.0
    makes_noise: float = 0.0  # 0-1, how loud
    breaks_stealth: bool = False
    
    # Timing
    duration: float = 0.5  # Seconds
    cooldown: float = 0.0  # Seconds before can use again
    
    # Priority
    priority: int = 0  # Higher = more important
    
    # Description
    description: str = ""


@dataclass
class EnhancedAction:
    """
    Enhanced action with more granular parameters.
    """
    action_type: EnhancedActionType
    
    # Timing
    duration: float = 0.5
    delay: float = 0.0  # Delay before starting
    
    # Parameters (action-specific)
    target_id: Optional[str] = None
    target_position: Optional[tuple] = None
    direction: Optional[str] = None  # "forward", "left", "right", "backward"
    intensity: float = 1.0  # 0-1, how hard/fast
    
    # Item/spell
    item_id: Optional[str] = None
    spell_id: Optional[str] = None
    
    # Metadata
    priority: int = 0
    reason: str = ""
    confidence: float = 1.0
    
    # Execution tracking
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action_type': self.action_type.value,
            'duration': self.duration,
            'delay': self.delay,
            'target_id': self.target_id,
            'target_position': self.target_position,
            'direction': self.direction,
            'intensity': self.intensity,
            'item_id': self.item_id,
            'spell_id': self.spell_id,
            'priority': self.priority,
            'reason': self.reason,
            'confidence': self.confidence
        }
    
    def start(self):
        """Mark action as started."""
        self.started_at = time.time()
    
    def complete(self, success: bool = True):
        """Mark action as completed."""
        self.completed_at = time.time()
        self.success = success
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self.started_at == 0.0:
            return 0.0
        return time.time() - self.started_at
    
    def is_complete(self) -> bool:
        """Check if action duration has elapsed."""
        if self.started_at == 0.0:
            return False
        return self.get_elapsed_time() >= self.duration


# ========================================
# Action Affordance Database
# ========================================

ACTION_AFFORDANCES: Dict[EnhancedActionType, ActionAffordance] = {
    # ========================================
    # MOVEMENT
    # ========================================
    
    EnhancedActionType.WALK_FORWARD: ActionAffordance(
        action_type=EnhancedActionType.WALK_FORWARD,
        category=ActionCategory.MOVEMENT,
        available_while_moving=True,
        makes_noise=0.2,
        duration=1.0,
        description="Walk forward slowly and quietly"
    ),
    
    EnhancedActionType.JOG_FORWARD: ActionAffordance(
        action_type=EnhancedActionType.JOG_FORWARD,
        category=ActionCategory.MOVEMENT,
        available_while_moving=True,
        makes_noise=0.5,
        duration=1.0,
        description="Jog forward at normal speed"
    ),
    
    EnhancedActionType.SPRINT_FORWARD: ActionAffordance(
        action_type=EnhancedActionType.SPRINT_FORWARD,
        category=ActionCategory.MOVEMENT,
        requires_stamina=0.1,
        drains_stamina=0.2,
        available_while_moving=True,
        makes_noise=0.9,
        breaks_stealth=True,
        duration=1.0,
        description="Sprint forward quickly (drains stamina)"
    ),
    
    EnhancedActionType.STRAFE_LEFT: ActionAffordance(
        action_type=EnhancedActionType.STRAFE_LEFT,
        category=ActionCategory.MOVEMENT,
        requires_combat=True,
        available_while_moving=True,
        makes_noise=0.4,
        duration=0.5,
        description="Strafe left while facing forward"
    ),
    
    EnhancedActionType.JUMP: ActionAffordance(
        action_type=EnhancedActionType.JUMP,
        category=ActionCategory.MOVEMENT,
        requires_stamina=0.05,
        drains_stamina=0.1,
        available_while_moving=True,
        makes_noise=0.6,
        breaks_stealth=True,
        duration=0.8,
        cooldown=0.5,
        description="Jump over obstacles"
    ),
    
    # ========================================
    # COMBAT
    # ========================================
    
    EnhancedActionType.LIGHT_ATTACK: ActionAffordance(
        action_type=EnhancedActionType.LIGHT_ATTACK,
        category=ActionCategory.COMBAT,
        requires_item="weapon",
        requires_stamina=0.05,
        drains_stamina=0.1,
        makes_noise=0.7,
        breaks_stealth=True,
        duration=0.4,
        cooldown=0.3,
        priority=5,
        description="Fast attack with low damage"
    ),
    
    EnhancedActionType.HEAVY_ATTACK: ActionAffordance(
        action_type=EnhancedActionType.HEAVY_ATTACK,
        category=ActionCategory.COMBAT,
        requires_item="weapon",
        requires_stamina=0.15,
        drains_stamina=0.25,
        makes_noise=0.9,
        breaks_stealth=True,
        duration=0.8,
        cooldown=0.5,
        priority=7,
        description="Slow attack with high damage"
    ),
    
    EnhancedActionType.POWER_ATTACK_FORWARD: ActionAffordance(
        action_type=EnhancedActionType.POWER_ATTACK_FORWARD,
        category=ActionCategory.COMBAT,
        requires_item="weapon",
        requires_stamina=0.25,
        drains_stamina=0.4,
        makes_noise=1.0,
        breaks_stealth=True,
        duration=1.0,
        cooldown=1.0,
        priority=9,
        description="Powerful forward strike"
    ),
    
    EnhancedActionType.BLOCK: ActionAffordance(
        action_type=EnhancedActionType.BLOCK,
        category=ActionCategory.COMBAT,
        requires_item="shield",
        drains_stamina=0.05,
        duration=1.0,
        priority=8,
        description="Block incoming attacks"
    ),
    
    EnhancedActionType.BASH: ActionAffordance(
        action_type=EnhancedActionType.BASH,
        category=ActionCategory.COMBAT,
        requires_item="shield",
        requires_stamina=0.15,
        drains_stamina=0.2,
        makes_noise=0.8,
        duration=0.5,
        cooldown=1.0,
        priority=6,
        description="Shield bash to stagger enemy"
    ),
    
    EnhancedActionType.DODGE_ROLL: ActionAffordance(
        action_type=EnhancedActionType.DODGE_ROLL,
        category=ActionCategory.COMBAT,
        requires_stamina=0.2,
        drains_stamina=0.3,
        makes_noise=0.5,
        duration=0.6,
        cooldown=1.5,
        priority=7,
        description="Roll to evade attacks"
    ),
    
    EnhancedActionType.DRAW_BOW: ActionAffordance(
        action_type=EnhancedActionType.DRAW_BOW,
        category=ActionCategory.COMBAT,
        requires_item="bow",
        drains_stamina=0.1,
        duration=0.5,
        description="Draw bow and aim"
    ),
    
    EnhancedActionType.RELEASE_ARROW: ActionAffordance(
        action_type=EnhancedActionType.RELEASE_ARROW,
        category=ActionCategory.COMBAT,
        requires_item="bow",
        requires_target=True,
        makes_noise=0.6,
        duration=0.3,
        cooldown=0.5,
        priority=6,
        description="Release arrow at target"
    ),
    
    # ========================================
    # STEALTH
    # ========================================
    
    EnhancedActionType.SNEAK: ActionAffordance(
        action_type=EnhancedActionType.SNEAK,
        category=ActionCategory.STEALTH,
        available_while_moving=False,
        duration=0.5,
        description="Enter sneak mode"
    ),
    
    EnhancedActionType.SNEAK_FORWARD: ActionAffordance(
        action_type=EnhancedActionType.SNEAK_FORWARD,
        category=ActionCategory.STEALTH,
        requires_stealth=True,
        available_while_moving=True,
        makes_noise=0.1,
        duration=1.0,
        description="Move forward while sneaking"
    ),
    
    EnhancedActionType.BACKSTAB: ActionAffordance(
        action_type=EnhancedActionType.BACKSTAB,
        category=ActionCategory.STEALTH,
        requires_stealth=True,
        requires_target=True,
        requires_item="dagger",
        makes_noise=0.3,
        breaks_stealth=True,
        duration=0.8,
        cooldown=2.0,
        priority=10,
        description="Sneak attack with dagger (high damage)"
    ),
    
    EnhancedActionType.PICKPOCKET: ActionAffordance(
        action_type=EnhancedActionType.PICKPOCKET,
        category=ActionCategory.STEALTH,
        requires_stealth=True,
        requires_target=True,
        duration=2.0,
        cooldown=5.0,
        priority=3,
        description="Steal from NPC's pockets"
    ),
    
    EnhancedActionType.LOCKPICK: ActionAffordance(
        action_type=EnhancedActionType.LOCKPICK,
        category=ActionCategory.STEALTH,
        requires_item="lockpick",
        requires_target=True,
        duration=3.0,
        priority=4,
        description="Pick lock on container or door"
    ),
    
    EnhancedActionType.HIDE_IN_SHADOWS: ActionAffordance(
        action_type=EnhancedActionType.HIDE_IN_SHADOWS,
        category=ActionCategory.STEALTH,
        requires_stealth=True,
        available_while_moving=False,
        makes_noise=0.0,
        duration=2.0,
        description="Stay still in dark area to avoid detection"
    ),
    
    # ========================================
    # MAGIC
    # ========================================
    
    EnhancedActionType.CAST_LEFT_HAND: ActionAffordance(
        action_type=EnhancedActionType.CAST_LEFT_HAND,
        category=ActionCategory.MAGIC,
        requires_item="spell",
        requires_magicka=0.1,
        drains_magicka=0.15,
        makes_noise=0.5,
        duration=0.5,
        cooldown=0.3,
        priority=5,
        description="Cast spell from left hand"
    ),
    
    EnhancedActionType.DUAL_CAST: ActionAffordance(
        action_type=EnhancedActionType.DUAL_CAST,
        category=ActionCategory.MAGIC,
        requires_item="spell",
        requires_magicka=0.25,
        drains_magicka=0.35,
        makes_noise=0.8,
        duration=0.8,
        cooldown=1.0,
        priority=8,
        description="Cast spell with both hands (more powerful)"
    ),
    
    EnhancedActionType.USE_SHOUT: ActionAffordance(
        action_type=EnhancedActionType.USE_SHOUT,
        category=ActionCategory.MAGIC,
        makes_noise=1.0,
        breaks_stealth=True,
        duration=1.0,
        cooldown=30.0,
        priority=9,
        description="Use dragon shout"
    ),
    
    # ========================================
    # INTERACTION
    # ========================================
    
    EnhancedActionType.ACTIVATE: ActionAffordance(
        action_type=EnhancedActionType.ACTIVATE,
        category=ActionCategory.INTERACTION,
        requires_target=True,
        duration=0.5,
        priority=3,
        description="Activate object or NPC"
    ),
    
    EnhancedActionType.OPEN_DOOR: ActionAffordance(
        action_type=EnhancedActionType.OPEN_DOOR,
        category=ActionCategory.INTERACTION,
        requires_target=True,
        makes_noise=0.4,
        duration=0.8,
        description="Open door"
    ),
    
    EnhancedActionType.TAKE_ITEM: ActionAffordance(
        action_type=EnhancedActionType.TAKE_ITEM,
        category=ActionCategory.INTERACTION,
        requires_target=True,
        duration=0.5,
        description="Take single item"
    ),
    
    EnhancedActionType.TALK_TO_NPC: ActionAffordance(
        action_type=EnhancedActionType.TALK_TO_NPC,
        category=ActionCategory.INTERACTION,
        requires_target=True,
        available_in_dialogue=True,
        duration=1.0,
        priority=4,
        description="Initiate conversation with NPC"
    ),
    
    EnhancedActionType.READ_BOOK: ActionAffordance(
        action_type=EnhancedActionType.READ_BOOK,
        category=ActionCategory.INTERACTION,
        requires_target=True,
        available_in_menu=True,
        duration=2.0,
        description="Read book or note"
    ),
    
    # ========================================
    # INVENTORY
    # ========================================
    
    EnhancedActionType.USE_POTION: ActionAffordance(
        action_type=EnhancedActionType.USE_POTION,
        category=ActionCategory.INVENTORY,
        requires_item="potion",
        duration=1.0,
        cooldown=0.5,
        priority=8,
        description="Drink potion for immediate effect"
    ),
    
    EnhancedActionType.EQUIP_ITEM: ActionAffordance(
        action_type=EnhancedActionType.EQUIP_ITEM,
        category=ActionCategory.INVENTORY,
        requires_item="equipment",
        duration=1.0,
        description="Equip weapon, armor, or spell"
    ),
    
    # ========================================
    # SOCIAL
    # ========================================
    
    EnhancedActionType.PERSUADE: ActionAffordance(
        action_type=EnhancedActionType.PERSUADE,
        category=ActionCategory.SOCIAL,
        requires_target=True,
        available_in_dialogue=True,
        duration=2.0,
        priority=5,
        description="Attempt to persuade NPC"
    ),
    
    EnhancedActionType.INTIMIDATE: ActionAffordance(
        action_type=EnhancedActionType.INTIMIDATE,
        category=ActionCategory.SOCIAL,
        requires_target=True,
        available_in_dialogue=True,
        duration=2.0,
        priority=5,
        description="Attempt to intimidate NPC"
    ),
    
    # ========================================
    # UTILITY
    # ========================================
    
    EnhancedActionType.WAIT: ActionAffordance(
        action_type=EnhancedActionType.WAIT,
        category=ActionCategory.UTILITY,
        available_while_moving=False,
        duration=1.0,
        description="Wait and observe"
    ),
    
    EnhancedActionType.SLEEP: ActionAffordance(
        action_type=EnhancedActionType.SLEEP,
        category=ActionCategory.UTILITY,
        requires_target=True,  # Bed
        duration=5.0,
        description="Sleep to restore health and pass time"
    ),
    
    EnhancedActionType.MOUNT_HORSE: ActionAffordance(
        action_type=EnhancedActionType.MOUNT_HORSE,
        category=ActionCategory.UTILITY,
        requires_target=True,
        duration=1.5,
        description="Mount horse for faster travel"
    ),
}


def get_affordance(action_type: EnhancedActionType) -> Optional[ActionAffordance]:
    """Get affordance for action type."""
    return ACTION_AFFORDANCES.get(action_type)


def get_available_actions(
    in_combat: bool = False,
    is_sneaking: bool = False,
    has_target: bool = False,
    stamina: float = 1.0,
    magicka: float = 1.0,
    equipped_items: List[str] = None
) -> List[EnhancedActionType]:
    """
    Get list of available actions based on current context.
    
    Args:
        in_combat: Whether player is in combat
        is_sneaking: Whether player is sneaking
        has_target: Whether there's a valid target
        stamina: Current stamina (0-1)
        magicka: Current magicka (0-1)
        equipped_items: List of equipped item types
    
    Returns:
        List of available action types
    """
    if equipped_items is None:
        equipped_items = []
    
    available = []
    
    for action_type, affordance in ACTION_AFFORDANCES.items():
        # Check combat requirement
        if affordance.requires_combat and not in_combat:
            continue
        
        # Check stealth requirement
        if affordance.requires_stealth and not is_sneaking:
            continue
        
        # Check target requirement
        if affordance.requires_target and not has_target:
            continue
        
        # Check stamina
        if affordance.requires_stamina > stamina:
            continue
        
        # Check magicka
        if affordance.requires_magicka > magicka:
            continue
        
        # Check item requirement
        if affordance.requires_item and affordance.requires_item not in equipped_items:
            continue
        
        available.append(action_type)
    
    return available


def get_actions_by_category(category: ActionCategory) -> List[EnhancedActionType]:
    """Get all actions in a category."""
    return [
        action_type for action_type, affordance in ACTION_AFFORDANCES.items()
        if affordance.category == category
    ]
