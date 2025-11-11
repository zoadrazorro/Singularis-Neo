"""
Action Affordances System

Tracks what actions are available in each action layer and enables
meta-strategic reasoning about layer transitions.

Key insight: The AGI should be aware of "Combat layer → power attack available"
as part of its world model, enabling strategic decisions like:
- "Enemy has low health → switch to Combat layer for power attack"
- "Multiple enemies → stay in Combat layer for defensive options"
- "Need to heal → switch to Menu layer for inventory access"

Design principles:
- Understanding affordances enables strategic action selection
- Action layers map to different gameplay contexts in Skyrim
- Context-appropriate actions improve gameplay effectiveness
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActionCategory(Enum):
    """Categories of actions for strategic reasoning."""
    MOVEMENT = "movement"
    COMBAT_OFFENSIVE = "combat_offensive"
    COMBAT_DEFENSIVE = "combat_defensive"
    INTERACTION = "interaction"
    NAVIGATION = "navigation"
    INVENTORY = "inventory"
    SOCIAL = "social"
    STEALTH = "stealth"
    MAGIC = "magic"


@dataclass
class ActionAffordance:
    """
    Represents an available action and its properties.
    """
    name: str
    category: ActionCategory
    layer: str
    description: str
    prerequisites: List[str] = None  # What conditions enable this action
    effects: List[str] = None       # What this action can achieve
    cooldown: float = 0.0           # Action cooldown in seconds
    resource_cost: Dict[str, float] = None  # Stamina, magicka, etc.
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.effects is None:
            self.effects = []
        if self.resource_cost is None:
            self.resource_cost = {}


class ActionAffordanceSystem:
    """
    Manages action affordances across different layers.
    
    Enables the AGI to reason about:
    1. What actions are currently available
    2. What actions would become available in other layers
    3. Strategic layer transitions based on desired actions
    4. Learning new affordances through experience
    """
    
    def __init__(self):
        """Initialize the affordance system."""
        # Layer -> List of available actions
        self.layer_affordances: Dict[str, List[ActionAffordance]] = {}
        
        # Action name -> ActionAffordance for quick lookup
        self.action_registry: Dict[str, ActionAffordance] = {}
        
        # Learned affordances from experience
        self.learned_affordances: List[ActionAffordance] = []
        
        # Action usage statistics
        self.action_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default Skyrim affordances
        self._initialize_skyrim_affordances()
    
    def _initialize_skyrim_affordances(self):
        """Initialize known Skyrim action affordances."""
        
        # === EXPLORATION LAYER ===
        exploration_actions = [
            ActionAffordance(
                name="move_forward",
                category=ActionCategory.MOVEMENT,
                layer="Exploration",
                description="Move forward in the world",
                effects=["position_change", "exploration_progress"]
            ),
            ActionAffordance(
                name="jump",
                category=ActionCategory.MOVEMENT,
                layer="Exploration",
                description="Jump over obstacles or reach higher ground",
                effects=["vertical_movement", "obstacle_bypass"],
                resource_cost={"stamina": 5.0}
            ),
            ActionAffordance(
                name="activate",
                category=ActionCategory.INTERACTION,
                layer="Exploration",
                description="Interact with objects, NPCs, or doors",
                effects=["object_interaction", "dialogue_start", "door_open"]
            ),
            ActionAffordance(
                name="sneak",
                category=ActionCategory.STEALTH,
                layer="Exploration",
                description="Toggle sneaking mode",
                effects=["stealth_mode", "reduced_detection"]
            ),
            ActionAffordance(
                name="attack",
                category=ActionCategory.COMBAT_OFFENSIVE,
                layer="Exploration",
                description="Basic attack (limited in exploration mode)",
                effects=["damage_dealt"],
                prerequisites=["weapon_equipped"]
            ),
        ]
        
        # === COMBAT LAYER ===
        combat_actions = [
            ActionAffordance(
                name="quick_attack",
                category=ActionCategory.COMBAT_OFFENSIVE,
                layer="Combat",
                description="Fast attack with current weapon",
                effects=["damage_dealt", "combo_potential"],
                resource_cost={"stamina": 10.0}
            ),
            ActionAffordance(
                name="power_attack",
                category=ActionCategory.COMBAT_OFFENSIVE,
                layer="Combat",
                description="Powerful attack that can stagger enemies",
                effects=["high_damage", "enemy_stagger", "armor_penetration"],
                resource_cost={"stamina": 25.0},
                cooldown=1.0
            ),
            ActionAffordance(
                name="block",
                category=ActionCategory.COMBAT_DEFENSIVE,
                layer="Combat",
                description="Block incoming attacks",
                effects=["damage_reduction", "stagger_prevention"],
                prerequisites=["shield_or_weapon_equipped"]
            ),
            ActionAffordance(
                name="bash",
                category=ActionCategory.COMBAT_OFFENSIVE,
                layer="Combat",
                description="Shield bash or weapon pommel strike",
                effects=["enemy_stagger", "interrupt_casting"],
                resource_cost={"stamina": 15.0},
                prerequisites=["shield_or_weapon_equipped"]
            ),
            ActionAffordance(
                name="shout",
                category=ActionCategory.MAGIC,
                layer="Combat",
                description="Use dragon shout ability",
                effects=["area_effect", "enemy_control", "environmental_change"],
                cooldown=30.0,
                prerequisites=["shout_learned"]
            ),
            ActionAffordance(
                name="dodge",
                category=ActionCategory.COMBAT_DEFENSIVE,
                layer="Combat",
                description="Dodge roll to avoid attacks",
                effects=["damage_avoidance", "position_change"],
                resource_cost={"stamina": 20.0}
            ),
            ActionAffordance(
                name="retreat",
                category=ActionCategory.COMBAT_DEFENSIVE,
                layer="Combat",
                description="Strategic retreat from combat",
                effects=["distance_increase", "combat_disengagement"]
            ),
            ActionAffordance(
                name="heal",
                category=ActionCategory.MAGIC,
                layer="Combat",
                description="Quick heal using favorited item",
                effects=["health_restoration"],
                prerequisites=["healing_item_favorited"]
            ),
        ]
        
        # === MENU LAYER ===
        menu_actions = [
            ActionAffordance(
                name="navigate_inventory",
                category=ActionCategory.NAVIGATION,
                layer="Menu",
                description="Navigate through inventory items",
                effects=["item_selection"]
            ),
            ActionAffordance(
                name="equip_item",
                category=ActionCategory.INVENTORY,
                layer="Menu",
                description="Equip weapons, armor, or accessories",
                effects=["equipment_change", "stat_modification"]
            ),
            ActionAffordance(
                name="consume_item",
                category=ActionCategory.INVENTORY,
                layer="Menu",
                description="Use consumable items like potions",
                effects=["stat_restoration", "temporary_buffs"]
            ),
            ActionAffordance(
                name="drop_item",
                category=ActionCategory.INVENTORY,
                layer="Menu",
                description="Drop items from inventory",
                effects=["inventory_space", "weight_reduction"]
            ),
            ActionAffordance(
                name="favorite_item",
                category=ActionCategory.INVENTORY,
                layer="Menu",
                description="Add item to favorites for quick access",
                effects=["quick_access_enabled"]
            ),
        ]
        
        # === DIALOGUE LAYER ===
        dialogue_actions = [
            ActionAffordance(
                name="select_dialogue_option",
                category=ActionCategory.SOCIAL,
                layer="Dialogue",
                description="Choose conversation response",
                effects=["relationship_change", "quest_progress", "information_gain"]
            ),
            ActionAffordance(
                name="exit_dialogue",
                category=ActionCategory.SOCIAL,
                layer="Dialogue",
                description="End conversation",
                effects=["dialogue_end", "layer_transition"]
            ),
        ]
        
        # === STEALTH LAYER ===
        stealth_actions = [
            ActionAffordance(
                name="sneak_move",
                category=ActionCategory.STEALTH,
                layer="Stealth",
                description="Move silently while crouched",
                effects=["silent_movement", "reduced_detection"]
            ),
            ActionAffordance(
                name="backstab",
                category=ActionCategory.COMBAT_OFFENSIVE,
                layer="Stealth",
                description="Sneak attack for massive damage",
                effects=["critical_damage", "stealth_kill_potential"],
                prerequisites=["undetected", "behind_enemy"]
            ),
            ActionAffordance(
                name="pickpocket",
                category=ActionCategory.STEALTH,
                layer="Stealth",
                description="Steal items from NPCs",
                effects=["item_acquisition", "crime_risk"],
                prerequisites=["undetected", "near_npc"]
            ),
        ]
        
        # Register all affordances
        all_affordances = (
            exploration_actions + combat_actions + menu_actions + 
            dialogue_actions + stealth_actions
        )
        
        for affordance in all_affordances:
            self.register_affordance(affordance)
        
        print(f"[OK] Initialized {len(all_affordances)} action affordances across {len(self.layer_affordances)} layers")
    
    def register_affordance(self, affordance: ActionAffordance):
        """Register a new action affordance."""
        # Add to layer mapping
        if affordance.layer not in self.layer_affordances:
            self.layer_affordances[affordance.layer] = []
        self.layer_affordances[affordance.layer].append(affordance)
        
        # Add to action registry
        self.action_registry[affordance.name] = affordance
        
        # Initialize stats
        self.action_stats[affordance.name] = {
            'usage_count': 0,
            'success_rate': 0.0,
            'avg_effectiveness': 0.0,
            'last_used': 0.0
        }
    
    def get_available_actions(
        self, 
        layer: str, 
        game_state: Dict[str, Any] = None
    ) -> List[ActionAffordance]:
        """
        Get actions available in a specific layer.
        
        Args:
            layer: Action layer name
            game_state: Current game state for prerequisite checking
            
        Returns:
            List of available actions
        """
        if layer not in self.layer_affordances:
            return []
        
        available = []
        for affordance in self.layer_affordances[layer]:
            if self._check_prerequisites(affordance, game_state):
                available.append(affordance)
        
        return available
    
    def _check_prerequisites(
        self, 
        affordance: ActionAffordance, 
        game_state: Dict[str, Any] = None
    ) -> bool:
        """Check if action prerequisites are met."""
        if not affordance.prerequisites or not game_state:
            return True
        
        # Simple prerequisite checking (can be enhanced)
        for prereq in affordance.prerequisites:
            if prereq == "weapon_equipped":
                # Would check if player has weapon equipped
                continue
            elif prereq == "shield_or_weapon_equipped":
                # Would check equipment
                continue
            elif prereq == "undetected":
                # Would check stealth status
                continue
            elif prereq == "shout_learned":
                # Would check known shouts
                continue
            # Add more prerequisite checks as needed
        
        return True
    
    def get_actions_by_category(
        self, 
        category: ActionCategory, 
        layer: str = None
    ) -> List[ActionAffordance]:
        """Get all actions of a specific category, optionally filtered by layer."""
        actions = []
        for affordance in self.action_registry.values():
            if affordance.category == category:
                if layer is None or affordance.layer == layer:
                    actions.append(affordance)
        return actions
    
    def find_layers_with_action_category(
        self, 
        category: ActionCategory
    ) -> List[str]:
        """Find which layers contain actions of a specific category."""
        layers = set()
        for affordance in self.action_registry.values():
            if affordance.category == category:
                layers.add(affordance.layer)
        return list(layers)
    
    def suggest_layer_for_goal(self, goal_effects: List[str]) -> Optional[str]:
        """
        Suggest best layer to achieve desired effects.
        
        Args:
            goal_effects: List of desired effects
            
        Returns:
            Recommended layer name
        """
        layer_scores = {}
        
        for layer_name, affordances in self.layer_affordances.items():
            score = 0
            for affordance in affordances:
                for effect in affordance.effects:
                    if effect in goal_effects:
                        score += 1
            
            if score > 0:
                layer_scores[layer_name] = score
        
        if layer_scores:
            return max(layer_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def learn_affordance_from_experience(
        self,
        action_name: str,
        layer: str,
        observed_effects: List[str],
        success: bool
    ):
        """
        Learn or update affordances based on experience.
        
        Args:
            action_name: Name of action performed
            layer: Layer where action was performed
            observed_effects: Effects that actually occurred
            success: Whether action was successful
        """
        # Update action statistics
        if action_name in self.action_stats:
            stats = self.action_stats[action_name]
            stats['usage_count'] += 1
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            stats['success_rate'] = (
                alpha * (1.0 if success else 0.0) + 
                (1 - alpha) * stats['success_rate']
            )
        
        # Learn new effects if not already known
        if action_name in self.action_registry:
            affordance = self.action_registry[action_name]
            for effect in observed_effects:
                if effect not in affordance.effects:
                    affordance.effects.append(effect)
                    print(f"[LEARN] Discovered new effect '{effect}' for action '{action_name}'")
    
    def get_strategic_analysis(
        self, 
        current_layer: str, 
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide strategic analysis of current situation.
        
        Returns information about:
        - Currently available actions
        - Actions available in other layers
        - Recommended layer transitions
        """
        current_actions = self.get_available_actions(current_layer, game_state)
        
        analysis = {
            'current_layer': current_layer,
            'current_actions': [a.name for a in current_actions],
            'current_categories': list(set(a.category.value for a in current_actions)),
            'layer_options': {},
            'recommendations': []
        }
        
        # Analyze other layers
        for layer_name in self.layer_affordances:
            if layer_name != current_layer:
                layer_actions = self.get_available_actions(layer_name, game_state)
                analysis['layer_options'][layer_name] = {
                    'actions': [a.name for a in layer_actions],
                    'categories': list(set(a.category.value for a in layer_actions)),
                    'unique_actions': [
                        a.name for a in layer_actions 
                        if a.name not in [ca.name for ca in current_actions]
                    ]
                }
        
        # Generate recommendations based on game state
        if game_state.get('in_combat', False):
            if current_layer != "Combat":
                analysis['recommendations'].append({
                    'action': 'switch_to_combat',
                    'reason': 'In combat - Combat layer provides defensive and offensive options',
                    'target_layer': 'Combat'
                })
        
        if game_state.get('health', 100) < 30:
            if current_layer != "Menu":
                analysis['recommendations'].append({
                    'action': 'switch_to_menu',
                    'reason': 'Low health - Menu layer allows healing item access',
                    'target_layer': 'Menu'
                })
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get affordance system statistics."""
        return {
            'total_affordances': len(self.action_registry),
            'layers': list(self.layer_affordances.keys()),
            'actions_per_layer': {
                layer: len(actions) 
                for layer, actions in self.layer_affordances.items()
            },
            'categories': list(set(
                a.category.value for a in self.action_registry.values()
            )),
            'most_used_actions': sorted(
                self.action_stats.items(),
                key=lambda x: x[1]['usage_count'],
                reverse=True
            )[:5]
        }
