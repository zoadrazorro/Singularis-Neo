"""
Action Affordance System - Dynamic Action Availability

Tracks what actions are available at any moment based on:
- Game state (combat, stealth, menu, dialogue)
- Player state (health, stamina, magicka, equipment)
- Environment (targets, obstacles, cover)
- Context (layer, location, time of day)

Provides intelligent action filtering and prioritization.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from loguru import logger

from .enhanced_actions import (
    EnhancedActionType,
    EnhancedAction,
    ActionAffordance,
    ActionCategory,
    get_affordance,
    get_available_actions,
    get_actions_by_category
)


class GameLayer(Enum):
    """Game layers that affect action availability."""
    EXPLORATION = "exploration"
    COMBAT = "combat"
    STEALTH = "stealth"
    MENU = "menu"
    DIALOGUE = "dialogue"
    INVENTORY = "inventory"
    MAP = "map"


@dataclass
class GameContext:
    """
    Current game context that determines action affordances.
    """
    # Layer
    current_layer: GameLayer = GameLayer.EXPLORATION
    
    # Player state
    health: float = 1.0
    stamina: float = 1.0
    magicka: float = 1.0
    is_sneaking: bool = False
    is_in_combat: bool = False
    is_moving: bool = False
    is_in_air: bool = False
    
    # Equipment
    equipped_weapon: Optional[str] = None  # "sword", "bow", "dagger", etc.
    equipped_shield: bool = False
    equipped_spell_left: Optional[str] = None
    equipped_spell_right: Optional[str] = None
    has_arrows: bool = False
    has_lockpicks: bool = False
    
    # Environment
    has_target: bool = False
    target_type: Optional[str] = None  # "enemy", "npc", "door", "container", etc.
    target_distance: float = 999.0
    target_is_aware: bool = False
    
    num_enemies_nearby: int = 0
    nearest_enemy_distance: float = 999.0
    
    has_cover_nearby: bool = False
    cover_distance: float = 999.0
    
    is_in_light: bool = True
    is_in_water: bool = False
    is_indoors: bool = False
    
    # Cooldowns (action_type -> time when available again)
    cooldowns: Dict[EnhancedActionType, float] = field(default_factory=dict)
    
    # Recent actions (for pattern detection)
    recent_actions: List[EnhancedActionType] = field(default_factory=list)
    
    def update_from_game_state(self, game_state: Dict):
        """Update context from game state dict."""
        # Player state
        player = game_state.get('player', {})
        self.health = player.get('health', 1.0)
        self.stamina = player.get('stamina', 1.0)
        self.magicka = player.get('magicka', 1.0)
        self.is_sneaking = player.get('sneaking', False)
        self.is_in_combat = player.get('in_combat', False)
        
        # Equipment
        equipment = player.get('equipment', {})
        self.equipped_weapon = equipment.get('weapon_type')
        self.equipped_shield = equipment.get('has_shield', False)
        self.equipped_spell_left = equipment.get('spell_left')
        self.equipped_spell_right = equipment.get('spell_right')
        self.has_arrows = equipment.get('arrow_count', 0) > 0
        self.has_lockpicks = equipment.get('lockpick_count', 0) > 0
        
        # Environment
        npcs = game_state.get('npcs', [])
        enemies = [npc for npc in npcs if npc.get('is_enemy', False)]
        self.num_enemies_nearby = len(enemies)
        
        if enemies:
            self.nearest_enemy_distance = min(e.get('distance_to_player', 999.0) for e in enemies)
        
        # Layer detection
        if player.get('in_menu', False):
            self.current_layer = GameLayer.MENU
        elif player.get('in_dialogue', False):
            self.current_layer = GameLayer.DIALOGUE
        elif self.is_in_combat:
            self.current_layer = GameLayer.COMBAT
        elif self.is_sneaking:
            self.current_layer = GameLayer.STEALTH
        else:
            self.current_layer = GameLayer.EXPLORATION
    
    def is_action_on_cooldown(self, action_type: EnhancedActionType) -> bool:
        """Check if action is on cooldown."""
        if action_type not in self.cooldowns:
            return False
        return time.time() < self.cooldowns[action_type]
    
    def start_cooldown(self, action_type: EnhancedActionType, duration: float):
        """Start cooldown for action."""
        self.cooldowns[action_type] = time.time() + duration
    
    def add_recent_action(self, action_type: EnhancedActionType):
        """Add to recent actions history."""
        self.recent_actions.append(action_type)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)


class ActionAffordanceSystem:
    """
    Manages action affordances and availability.
    
    Provides:
    - Dynamic action filtering based on context
    - Action prioritization
    - Cooldown management
    - Pattern detection (e.g., stuck in loop)
    """
    
    def __init__(self):
        self.context = GameContext()
        self.action_history: List[Tuple[float, EnhancedActionType]] = []
        
        logger.info("[ActionAffordance] System initialized")
    
    def update_context(self, game_state: Dict):
        """Update game context from game state."""
        self.context.update_from_game_state(game_state)
    
    def get_available_actions(
        self,
        include_low_priority: bool = True,
        filter_by_layer: bool = True
    ) -> List[EnhancedActionType]:
        """
        Get list of currently available actions.
        
        Args:
            include_low_priority: Include low-priority actions
            filter_by_layer: Filter actions by current game layer
        
        Returns:
            List of available action types
        """
        available = []
        
        # Get equipped items list
        equipped_items = []
        if self.context.equipped_weapon:
            equipped_items.append(self.context.equipped_weapon)
        if self.context.equipped_shield:
            equipped_items.append("shield")
        if self.context.equipped_spell_left or self.context.equipped_spell_right:
            equipped_items.append("spell")
        if self.context.has_arrows:
            equipped_items.append("arrows")
        if self.context.has_lockpicks:
            equipped_items.append("lockpick")
        
        # Get base available actions
        base_available = get_available_actions(
            in_combat=self.context.is_in_combat,
            is_sneaking=self.context.is_sneaking,
            has_target=self.context.has_target,
            stamina=self.context.stamina,
            magicka=self.context.magicka,
            equipped_items=equipped_items
        )
        
        for action_type in base_available:
            affordance = get_affordance(action_type)
            if not affordance:
                continue
            
            # Check cooldown
            if self.context.is_action_on_cooldown(action_type):
                continue
            
            # Check layer restrictions
            if filter_by_layer:
                if self.context.current_layer == GameLayer.MENU:
                    if not affordance.available_in_menu:
                        continue
                elif self.context.current_layer == GameLayer.DIALOGUE:
                    if not affordance.available_in_dialogue:
                        continue
            
            # Check movement restrictions
            if self.context.is_moving and not affordance.available_while_moving:
                continue
            
            if self.context.is_in_air and not affordance.available_while_in_air:
                continue
            
            # Check priority
            if not include_low_priority and affordance.priority < 3:
                continue
            
            available.append(action_type)
        
        return available
    
    def get_prioritized_actions(
        self,
        available_actions: Optional[List[EnhancedActionType]] = None
    ) -> List[Tuple[EnhancedActionType, int]]:
        """
        Get available actions sorted by priority.
        
        Args:
            available_actions: Optional pre-filtered list
        
        Returns:
            List of (action_type, priority) tuples, sorted by priority
        """
        if available_actions is None:
            available_actions = self.get_available_actions()
        
        prioritized = []
        for action_type in available_actions:
            affordance = get_affordance(action_type)
            if affordance:
                prioritized.append((action_type, affordance.priority))
        
        # Sort by priority (descending)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    def get_actions_for_goal(
        self,
        goal: str,
        available_actions: Optional[List[EnhancedActionType]] = None
    ) -> List[EnhancedActionType]:
        """
        Get actions that help achieve a specific goal.
        
        Args:
            goal: Goal description (e.g., "escape", "attack", "stealth")
            available_actions: Optional pre-filtered list
        
        Returns:
            List of relevant action types
        """
        if available_actions is None:
            available_actions = self.get_available_actions()
        
        goal_lower = goal.lower()
        relevant = []
        
        for action_type in available_actions:
            action_name = action_type.value.lower()
            affordance = get_affordance(action_type)
            
            # Match by keywords
            if goal_lower in action_name:
                relevant.append(action_type)
            elif goal_lower == "escape" and ("flee" in action_name or "backward" in action_name or "sprint" in action_name):
                relevant.append(action_type)
            elif goal_lower == "attack" and affordance and affordance.category == ActionCategory.COMBAT:
                relevant.append(action_type)
            elif goal_lower == "stealth" and affordance and affordance.category == ActionCategory.STEALTH:
                relevant.append(action_type)
            elif goal_lower == "defend" and ("block" in action_name or "dodge" in action_name):
                relevant.append(action_type)
        
        return relevant
    
    def filter_by_situation(
        self,
        available_actions: Optional[List[EnhancedActionType]] = None
    ) -> Dict[str, List[EnhancedActionType]]:
        """
        Categorize actions by situation.
        
        Returns:
            Dict mapping situation names to action lists
        """
        if available_actions is None:
            available_actions = self.get_available_actions()
        
        situations = {
            'offensive': [],
            'defensive': [],
            'mobility': [],
            'stealth': [],
            'utility': [],
            'emergency': []
        }
        
        for action_type in available_actions:
            affordance = get_affordance(action_type)
            if not affordance:
                continue
            
            action_name = action_type.value.lower()
            
            # Offensive
            if 'attack' in action_name or 'bash' in action_name or 'cast_destruction' in action_name:
                situations['offensive'].append(action_type)
            
            # Defensive
            if 'block' in action_name or 'dodge' in action_name or 'parry' in action_name:
                situations['defensive'].append(action_type)
            
            # Mobility
            if affordance.category == ActionCategory.MOVEMENT:
                situations['mobility'].append(action_type)
            
            # Stealth
            if affordance.category == ActionCategory.STEALTH:
                situations['stealth'].append(action_type)
            
            # Utility
            if affordance.category == ActionCategory.UTILITY:
                situations['utility'].append(action_type)
            
            # Emergency (low health)
            if self.context.health < 0.3:
                if 'heal' in action_name or 'potion' in action_name or 'flee' in action_name:
                    situations['emergency'].append(action_type)
        
        return situations
    
    def detect_action_loop(self, window: int = 5) -> bool:
        """
        Detect if agent is stuck in action loop.
        
        Args:
            window: Number of recent actions to check
        
        Returns:
            True if loop detected
        """
        if len(self.context.recent_actions) < window:
            return False
        
        recent = self.context.recent_actions[-window:]
        
        # Check if all actions are the same
        if len(set(recent)) == 1:
            logger.warning(f"[ActionAffordance] Loop detected: repeating {recent[0].value}")
            return True
        
        # Check if alternating between 2 actions
        if len(set(recent)) == 2:
            if all(recent[i] == recent[i % 2] for i in range(len(recent))):
                logger.warning(f"[ActionAffordance] Loop detected: alternating {recent[0].value} <-> {recent[1].value}")
                return True
        
        return False
    
    def suggest_alternative_actions(
        self,
        current_action: EnhancedActionType,
        count: int = 3
    ) -> List[EnhancedActionType]:
        """
        Suggest alternative actions to current one.
        
        Useful for breaking out of loops.
        
        Args:
            current_action: Current action being considered
            count: Number of alternatives to suggest
        
        Returns:
            List of alternative action types
        """
        available = self.get_available_actions()
        
        # Remove current action
        alternatives = [a for a in available if a != current_action]
        
        # Remove recently used actions
        recent_set = set(self.context.recent_actions[-5:])
        alternatives = [a for a in alternatives if a not in recent_set]
        
        # Prioritize
        prioritized = self.get_prioritized_actions(alternatives)
        
        # Return top N
        return [action for action, _ in prioritized[:count]]
    
    def execute_action(self, action: EnhancedAction):
        """
        Mark action as executed (for tracking).
        
        Args:
            action: Action being executed
        """
        # Add to history
        self.action_history.append((time.time(), action.action_type))
        self.context.add_recent_action(action.action_type)
        
        # Start cooldown
        affordance = get_affordance(action.action_type)
        if affordance and affordance.cooldown > 0:
            self.context.start_cooldown(action.action_type, affordance.cooldown)
        
        logger.debug(f"[ActionAffordance] Executed: {action.action_type.value}")
    
    def get_stats(self) -> Dict:
        """Get affordance system statistics."""
        available = self.get_available_actions()
        
        return {
            'current_layer': self.context.current_layer.value,
            'available_actions': len(available),
            'in_combat': self.context.is_in_combat,
            'is_sneaking': self.context.is_sneaking,
            'health': self.context.health,
            'stamina': self.context.stamina,
            'magicka': self.context.magicka,
            'num_enemies': self.context.num_enemies_nearby,
            'actions_on_cooldown': len([a for a in self.context.cooldowns if self.context.is_action_on_cooldown(a)]),
            'recent_actions': [a.value for a in self.context.recent_actions[-5:]],
            'loop_detected': self.detect_action_loop()
        }


# ========================================
# Helper Functions
# ========================================

def create_action_from_type(
    action_type: EnhancedActionType,
    target_id: Optional[str] = None,
    intensity: float = 1.0,
    reason: str = ""
) -> EnhancedAction:
    """
    Create EnhancedAction from type with affordance defaults.
    
    Args:
        action_type: Type of action
        target_id: Optional target ID
        intensity: Action intensity (0-1)
        reason: Reason for choosing this action
    
    Returns:
        EnhancedAction instance
    """
    affordance = get_affordance(action_type)
    
    if affordance:
        return EnhancedAction(
            action_type=action_type,
            duration=affordance.duration,
            target_id=target_id,
            intensity=intensity,
            priority=affordance.priority,
            reason=reason
        )
    else:
        return EnhancedAction(
            action_type=action_type,
            target_id=target_id,
            intensity=intensity,
            reason=reason
        )
