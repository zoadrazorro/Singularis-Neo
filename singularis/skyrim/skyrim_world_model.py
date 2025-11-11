"""
Skyrim-Specific World Model

Extends the base world model with Skyrim-specific knowledge:
1. Causal rules ("stealing → guards hostile")
2. NPC relationships and factions
3. Quest mechanics and dependencies
4. Combat and magic systems
5. Geography and location knowledge

Design principles:
- Causal learning enables prediction and planning
- Understanding game mechanics improves decision-making
- World model grounds actions in game reality
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..world_model import CausalGraph, CausalEdge


@dataclass
class NPCRelationship:
    """Relationship between player and NPC."""
    npc_name: str
    faction: str
    relationship_value: float  # -1 (hostile) to +1 (friendly)
    interactions: int = 0


class SkyrimWorldModel:
    """
    Skyrim-specific world model.

    Learns causal relationships through experience:
    - "If I steal, guards become hostile"
    - "If I help NPC, relationship improves"
    - "If I kill chicken, town becomes hostile" (classic Skyrim!)
    - "If I use fire spell on oil, it ignites"
    """

    def __init__(self, base_world_model=None):
        """
        Initialize Skyrim world model.

        Args:
            base_world_model: WorldModelOrchestrator instance
        """
        self.base_world_model = base_world_model

        # Skyrim-specific causal graph
        self.causal_graph = CausalGraph()

        # NPC relationships
        self.npc_relationships: Dict[str, NPCRelationship] = {}

        # Known locations (terrain-focused, not narrative)
        self.locations: Dict[str, Dict[str, Any]] = {}
        
        # Terrain knowledge (environmental understanding)
        self.terrain_knowledge: Dict[str, Dict[str, Any]] = {
            'indoor_spaces': {},  # Confined areas, exits, interactive objects
            'outdoor_spaces': {},  # Open terrain, landmarks, paths
            'vertical_features': {},  # Cliffs, stairs, elevated positions
            'obstacles': {},  # Walls, water, impassable terrain
            'safe_zones': {},  # Areas without threats
            'danger_zones': {},  # Areas with frequent combat
        }

        # Learned rules (environment-focused, not story-focused)
        self.learned_rules: List[Dict[str, Any]] = []
        
        # Layer affordance mappings (learned through experience)
        self.layer_affordance_mappings: Dict[str, Dict[str, Any]] = {}
        
        # Action effectiveness by layer (learned)
        self.action_effectiveness: Dict[str, Dict[str, float]] = {}

        # Initialize common Skyrim causal relationships
        self._initialize_skyrim_causality()
        self._initialize_layer_knowledge()

    def _initialize_skyrim_causality(self):
        """Initialize known Skyrim causal relationships."""

        # Crime and justice
        self.causal_graph.add_edge(CausalEdge(
            cause='steal_item',
            effect='bounty_increased',
            strength=1.0,
            confidence=0.95
        ))
        self.causal_graph.add_edge(CausalEdge(
            cause='bounty_increased',
            effect='guards_hostile',
            strength=0.9,
            confidence=0.9
        ))

        # Classic chicken incident
        self.causal_graph.add_edge(CausalEdge(
            cause='kill_chicken',
            effect='town_hostility',
            strength=1.0,
            confidence=1.0  # This ALWAYS happens!
        ))

        # Combat
        self.causal_graph.add_edge(CausalEdge(
            cause='attack_npc',
            effect='npc_becomes_hostile',
            strength=1.0,
            confidence=0.95
        ))

        # Magic and environment
        self.causal_graph.add_edge(CausalEdge(
            cause='fire_spell_on_oil',
            effect='fire_spreads',
            strength=0.8,
            confidence=0.9
        ))

        # Social
        self.causal_graph.add_edge(CausalEdge(
            cause='help_npc_quest',
            effect='relationship_improves',
            strength=0.7,
            confidence=0.85
        ))

        print("[OK] Initialized Skyrim causal relationships")

    def _initialize_layer_knowledge(self):
        """Initialize known layer→affordance mappings."""
        
        # Combat layer knowledge
        self.layer_affordance_mappings["Combat"] = {
            'primary_purpose': 'offensive_and_defensive_actions',
            'key_affordances': ['power_attack', 'block', 'dodge', 'shout'],
            'effectiveness_context': 'high_threat_situations',
            'transition_triggers': ['enemy_detected', 'health_low', 'multiple_enemies']
        }
        
        # Exploration layer knowledge  
        self.layer_affordance_mappings["Exploration"] = {
            'primary_purpose': 'world_navigation_and_interaction',
            'key_affordances': ['move_forward', 'jump', 'activate', 'sneak'],
            'effectiveness_context': 'peaceful_exploration',
            'transition_triggers': ['no_immediate_threats', 'quest_objectives']
        }
        
        # Menu layer knowledge
        self.layer_affordance_mappings["Menu"] = {
            'primary_purpose': 'inventory_and_character_management',
            'key_affordances': ['equip_item', 'consume_item', 'favorite_item'],
            'effectiveness_context': 'safe_environments',
            'transition_triggers': ['need_healing', 'equipment_change', 'inventory_full']
        }
        
        # Stealth layer knowledge
        self.layer_affordance_mappings["Stealth"] = {
            'primary_purpose': 'covert_operations',
            'key_affordances': ['sneak_move', 'backstab', 'pickpocket'],
            'effectiveness_context': 'stealth_required_situations',
            'transition_triggers': ['avoid_detection', 'assassination_opportunity']
        }
        
        # Initialize action effectiveness tracking
        for layer in self.layer_affordance_mappings:
            self.action_effectiveness[layer] = {}
        
        print("[OK] Initialized layer knowledge base")

    def learn_from_experience(
        self,
        action: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        surprise_threshold: float = 0.3
    ):
        """
        Learn causal relationships from experience.

        Args:
            action: Action taken
            before_state: State before action
            after_state: State after action
            surprise_threshold: Threshold for surprising outcomes
        """
        # Compute surprise (difference between states)
        surprise = self._compute_surprise(before_state, after_state)

        if surprise > surprise_threshold:
            print(f"Surprising outcome! Learning from experience...")

            # Identify what changed
            changes = self._identify_changes(before_state, after_state)

            # Learn causal edges
            for change_var, change_val in changes.items():
                # Add or strengthen causal edge
                edge = CausalEdge(
                    cause=action,
                    effect=change_var,
                    strength=abs(change_val),
                    confidence=0.5  # Start uncertain
                )
                self.causal_graph.add_edge(edge)

                # Record learned rule
                self.learned_rules.append({
                    'action': action,
                    'effect': change_var,
                    'change': change_val,
                    'surprise': surprise
                })
        
        # Learn layer effectiveness if layer info is available
        if 'current_action_layer' in before_state and 'current_action_layer' in after_state:
            self._learn_layer_effectiveness(
                action, 
                before_state['current_action_layer'],
                before_state,
                after_state,
                surprise
            )

    def _learn_layer_effectiveness(
        self,
        action: str,
        layer: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        surprise: float
    ):
        """
        Learn how effective actions are in different layers.
        
        Args:
            action: Action performed
            layer: Layer where action was performed
            before_state: State before action
            after_state: State after action
            surprise: How surprising the outcome was
        """
        if layer not in self.action_effectiveness:
            self.action_effectiveness[layer] = {}
        
        if action not in self.action_effectiveness[layer]:
            self.action_effectiveness[layer][action] = {
                'success_count': 0,
                'total_count': 0,
                'avg_effectiveness': 0.0,
                'contexts': []
            }
        
        stats = self.action_effectiveness[layer][action]
        stats['total_count'] += 1
        
        # Determine if action was successful (low surprise = expected outcome)
        success = surprise < 0.3
        if success:
            stats['success_count'] += 1
        
        # Update effectiveness (success rate)
        stats['avg_effectiveness'] = stats['success_count'] / stats['total_count']
        
        # Record context for pattern learning
        context = {
            'health': before_state.get('health', 100),
            'in_combat': before_state.get('in_combat', False),
            'enemies_nearby': before_state.get('enemies_nearby', 0),
            'success': success
        }
        stats['contexts'].append(context)
        
        # Keep only recent contexts (last 20)
        if len(stats['contexts']) > 20:
            stats['contexts'] = stats['contexts'][-20:]

    def suggest_optimal_layer(
        self,
        desired_action: str,
        current_state: Dict[str, Any]
    ) -> Optional[str]:
        """
        Suggest the optimal layer for performing a desired action.
        
        Args:
            desired_action: Action the AGI wants to perform
            current_state: Current game state
            
        Returns:
            Recommended layer name, or None if no good option
        """
        layer_scores = {}
        
        for layer, actions in self.action_effectiveness.items():
            if desired_action in actions:
                stats = actions[desired_action]
                base_score = stats['avg_effectiveness']
                
                # Adjust score based on current context
                context_bonus = self._compute_context_bonus(
                    stats['contexts'], 
                    current_state
                )
                
                layer_scores[layer] = base_score + context_bonus
        
        if layer_scores:
            best_layer = max(layer_scores.items(), key=lambda x: x[1])
            return best_layer[0] if best_layer[1] > 0.3 else None
        
        return None

    def _compute_context_bonus(
        self,
        historical_contexts: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> float:
        """
        Compute context similarity bonus for layer selection.
        
        Args:
            historical_contexts: Past contexts where action was used
            current_state: Current game state
            
        Returns:
            Bonus score based on context similarity
        """
        if not historical_contexts:
            return 0.0
        
        # Find similar contexts
        similar_contexts = []
        for context in historical_contexts:
            similarity = 0.0
            
            # Health similarity
            if 'health' in current_state and 'health' in context:
                health_diff = abs(current_state['health'] - context['health']) / 100.0
                similarity += max(0, 1.0 - health_diff)
            
            # Combat state similarity
            if (current_state.get('in_combat', False) == 
                context.get('in_combat', False)):
                similarity += 1.0
            
            # Enemy count similarity
            if 'enemies_nearby' in current_state and 'enemies_nearby' in context:
                enemy_diff = abs(
                    current_state['enemies_nearby'] - context['enemies_nearby']
                )
                similarity += max(0, 1.0 - enemy_diff / 5.0)  # Normalize by max 5 enemies
            
            if similarity > 1.5:  # Threshold for "similar context"
                similar_contexts.append(context)
        
        if not similar_contexts:
            return 0.0
        
        # Compute success rate in similar contexts
        success_rate = sum(1 for ctx in similar_contexts if ctx.get('success', False))
        success_rate /= len(similar_contexts)
        
        return (success_rate - 0.5) * 0.3  # Bonus/penalty up to ±0.3

    def get_strategic_layer_analysis(
        self,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide strategic analysis for layer selection.
        
        Args:
            current_state: Current game state
            
        Returns:
            Analysis with layer recommendations
        """
        analysis = {
            'current_layer': current_state.get('current_action_layer', 'Unknown'),
            'recommendations': [],
            'layer_effectiveness': {},
            'context_analysis': {}
        }
        
        # Analyze each layer's effectiveness in current context
        for layer, layer_info in self.layer_affordance_mappings.items():
            effectiveness_score = 0.0
            action_count = 0
            
            if layer in self.action_effectiveness:
                for action, stats in self.action_effectiveness[layer].items():
                    context_bonus = self._compute_context_bonus(
                        stats['contexts'],
                        current_state
                    )
                    effectiveness_score += stats['avg_effectiveness'] + context_bonus
                    action_count += 1
            
            if action_count > 0:
                analysis['layer_effectiveness'][layer] = effectiveness_score / action_count
            else:
                analysis['layer_effectiveness'][layer] = 0.5  # Neutral
        
        # Generate recommendations based on context
        if current_state.get('in_combat', False):
            if analysis['layer_effectiveness'].get('Combat', 0) > 0.6:
                analysis['recommendations'].append({
                    'layer': 'Combat',
                    'reason': 'High combat effectiveness in similar situations',
                    'confidence': analysis['layer_effectiveness']['Combat']
                })
        
        if current_state.get('health', 100) < 30:
            if analysis['layer_effectiveness'].get('Menu', 0) > 0.5:
                analysis['recommendations'].append({
                    'layer': 'Menu',
                    'reason': 'Low health - menu access for healing',
                    'confidence': analysis['layer_effectiveness']['Menu']
                })
        
        return analysis

    def _compute_surprise(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> float:
        """Compute how surprising the outcome was."""
        # Simple heuristic: count changed variables
        changes = self._identify_changes(before, after)
        return len(changes) / (len(before) + 1)

    def _identify_changes(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify what changed between states."""
        changes = {}
        for key in before:
            if key in after and before[key] != after[key]:
                if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
                    changes[key] = after[key] - before[key]
                else:
                    changes[key] = 1.0  # Binary change
        return changes

    def predict_outcome(
        self,
        action: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict outcome of action.

        Args:
            action: Proposed action
            current_state: Current state

        Returns:
            Predicted next state
        """
        predicted_state = current_state.copy()

        # Find causal edges from this action
        edges = self.causal_graph.get_outgoing_edges(action)

        for edge in edges:
            # Apply causal effect
            if edge.effect in predicted_state:
                if isinstance(predicted_state[edge.effect], (int, float)):
                    # Modify numerical value
                    predicted_state[edge.effect] += edge.strength
                else:
                    # Binary change
                    predicted_state[edge.effect] = True

        return predicted_state

    def update_npc_relationship(
        self,
        npc_name: str,
        faction: str,
        delta: float
    ):
        """
        Update relationship with NPC.

        Args:
            npc_name: NPC name
            faction: NPC's faction
            delta: Change in relationship (-1 to +1)
        """
        if npc_name not in self.npc_relationships:
            self.npc_relationships[npc_name] = NPCRelationship(
                npc_name=npc_name,
                faction=faction,
                relationship_value=0.0
            )

        rel = self.npc_relationships[npc_name]
        rel.relationship_value = np.clip(
            rel.relationship_value + delta,
            -1.0,
            1.0
        )
        rel.interactions += 1

    def get_npc_relationship(self, npc_name: str) -> Optional[NPCRelationship]:
        """Get relationship with NPC."""
        return self.npc_relationships.get(npc_name)

    def add_location(
        self,
        location_name: str,
        location_type: str,
        features: Dict[str, Any]
    ):
        """
        Add discovered location.

        Args:
            location_name: Name of location
            location_type: Type (dungeon, city, etc.)
            features: Location features
        """
        self.locations[location_name] = {
            'type': location_type,
            'features': features,
            'visited': True,
            'explored': False,
        }

    def mark_location_explored(self, location_name: str):
        """Mark location as fully explored."""
        if location_name in self.locations:
            self.locations[location_name]['explored'] = True

    def get_unexplored_locations(self) -> List[str]:
        """Get list of discovered but unexplored locations."""
        return [
            name for name, loc in self.locations.items()
            if loc['visited'] and not loc['explored']
        ]

    def add_quest(
        self,
        quest_name: str,
        quest_type: str,
        objectives: List[str]
    ):
        """Add quest (deprecated - using terrain-based system)."""
        # Quests removed in favor of terrain-aware system
        pass

    def complete_quest_objective(self, quest_name: str, objective: str):
        """Complete quest objective (deprecated - using terrain-based system)."""
        # Quests removed in favor of terrain-aware system
        pass

    def get_active_quests(self) -> List[str]:
        """Get active quest names (deprecated - using terrain-based system)."""
        # Quests removed in favor of terrain-aware system
        return []

    def evaluate_moral_choice(
        self,
        choice: str,
        consequences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate moral choice using game-specific impact assessment.

        Args:
            choice: Description of choice
            consequences: Expected consequences

        Returns:
            Evaluation with impact score
        """
        # Estimate game impact
        # Negative actions (stealing, murder) have negative consequences
        # Helpful actions have positive outcomes

        negative_keywords = ['steal', 'kill', 'murder', 'betray', 'lie']
        positive_keywords = ['help', 'save', 'heal', 'defend', 'protect']

        choice_lower = choice.lower()

        # Heuristic impact estimate (replaces coherence delta)
        impact_score = 0.0

        for keyword in negative_keywords:
            if keyword in choice_lower:
                impact_score -= 0.1  # Negative consequences (bounty, hostile NPCs)

        for keyword in positive_keywords:
            if keyword in choice_lower:
                impact_score += 0.1  # Positive outcomes (rewards, friendship)

        # Evaluate outcome
        if impact_score > 0.02:
            outcome_status = "BENEFICIAL"  # Good for the player
        elif abs(impact_score) < 0.02:
            outcome_status = "NEUTRAL"  # Minimal impact
        else:
            outcome_status = "DETRIMENTAL"  # Bad consequences

        return {
            'choice': choice,
            'impact_score': impact_score,
            'outcome_status': outcome_status,
            'recommendation': outcome_status == "BENEFICIAL"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            'causal_edges': len(self.causal_graph.graph.edges()),
            'npc_relationships': len(self.npc_relationships),
            'locations_discovered': len(self.locations),
            'locations_explored': sum(1 for loc in self.locations.values() if loc['explored']),
            'active_quests': len(self.get_active_quests()),
            'learned_rules': len(self.learned_rules),
        }


    def learn_terrain_feature(
        self,
        location: str,
        terrain_type: str,
        feature_data: Dict[str, Any]
    ):
        """
        Learn about terrain features from exploration.
        
        Args:
            location: Location name
            terrain_type: Type of terrain (indoor_spaces, outdoor_spaces, etc.)
            feature_data: Data about the terrain feature
        """
        if terrain_type not in self.terrain_knowledge:
            return
        
        if location not in self.terrain_knowledge[terrain_type]:
            self.terrain_knowledge[terrain_type][location] = {
                'visits': 0,
                'features': []
            }
        
        self.terrain_knowledge[terrain_type][location]['visits'] += 1
        self.terrain_knowledge[terrain_type][location]['features'].append(feature_data)
        
        print(f"[TERRAIN] Learned {terrain_type} feature at {location}")

    def classify_terrain_from_scene(self, scene_type: str, in_combat: bool) -> str:
        """
        Classify terrain type from scene classification.
        
        Args:
            scene_type: Visual scene classification
            in_combat: Whether currently in combat
            
        Returns:
            Terrain type string
        """
        if in_combat:
            return 'danger_zones'
        elif scene_type in ['inventory', 'menu', 'dialogue']:
            return 'indoor_spaces'
        elif scene_type in ['exploration', 'outdoor']:
            return 'outdoor_spaces'
        elif scene_type == 'combat':
            return 'danger_zones'
        else:
            return 'outdoor_spaces'  # Default

    def get_terrain_recommendations(
        self,
        current_location: str,
        scene_type: str,
        in_combat: bool
    ) -> List[str]:
        """
        Get terrain-aware action recommendations.
        
        Args:
            current_location: Current location
            scene_type: Visual scene type
            in_combat: Whether in combat
            
        Returns:
            List of recommended actions based on terrain
        """
        terrain_type = self.classify_terrain_from_scene(scene_type, in_combat)
        recommendations = []
        
        if terrain_type == 'indoor_spaces':
            recommendations.extend([
                "Look for exits and doorways",
                "Interact with objects (activate)",
                "Use vertical space (look up/down for paths)",
                "Navigate carefully in confined space"
            ])
        elif terrain_type == 'outdoor_spaces':
            recommendations.extend([
                "Prioritize forward movement",
                "Scan horizon with camera",
                "Look for elevated positions",
                "Cover distance efficiently"
            ])
        elif terrain_type == 'danger_zones':
            recommendations.extend([
                "Use terrain for cover",
                "Identify retreat paths",
                "Consider elevation advantage",
                "Assess threat positions"
            ])
        elif terrain_type == 'vertical_features':
            recommendations.extend([
                "Look up for climbing paths",
                "Consider jumping mechanics",
                "Check for fall hazards",
                "Use elevation strategically"
            ])
        
        return recommendations

    def update_terrain_safety(
        self,
        location: str,
        had_combat: bool
    ):
        """
        Update terrain safety knowledge based on combat encounters.
        
        Args:
            location: Location name
            had_combat: Whether combat occurred
        """
        if had_combat:
            if location not in self.terrain_knowledge['danger_zones']:
                self.terrain_knowledge['danger_zones'][location] = {
                    'combat_encounters': 0,
                    'last_encounter': None
                }
            self.terrain_knowledge['danger_zones'][location]['combat_encounters'] += 1
            print(f"[TERRAIN] Marked {location} as danger zone")
        else:
            if location not in self.terrain_knowledge['safe_zones']:
                self.terrain_knowledge['safe_zones'][location] = {
                    'safe_visits': 0
                }
            self.terrain_knowledge['safe_zones'][location]['safe_visits'] += 1


# Example usage
if __name__ == "__main__":
    print("Testing Skyrim World Model...")

    wm = SkyrimWorldModel()

    # 1. Test causal prediction
    print("\n1. Testing causal prediction...")
    state = {'bounty': 0, 'guards_hostile': False}
    predicted = wm.predict_outcome('steal_item', state)
    print(f"   Before: {state}")
    print(f"   Action: steal_item")
    print(f"   Predicted: {predicted}")

    # 2. Test learning from experience
    print("\n2. Testing learning from experience...")
    before = {'health': 100, 'in_combat': False}
    after = {'health': 80, 'in_combat': True}
    wm.learn_from_experience('attack_dragon', before, after, surprise_threshold=0.2)
    print(f"   Learned: attack_dragon has consequences!")

    # 3. Test NPC relationships
    print("\n3. Testing NPC relationships...")
    wm.update_npc_relationship('Lydia', 'Whiterun', delta=0.2)
    wm.update_npc_relationship('Ulfric', 'Stormcloaks', delta=-0.5)
    print(f"   Lydia: {wm.get_npc_relationship('Lydia').relationship_value:.2f}")
    print(f"   Ulfric: {wm.get_npc_relationship('Ulfric').relationship_value:.2f}")

    # 4. Test moral evaluation
    print("\n4. Testing moral evaluation...")
    eval_help = wm.evaluate_moral_choice(
        "Help the wounded traveler",
        {'relationship': +0.1}
    )
    print(f"   Help: {eval_help['outcome_status']} (impact={eval_help['impact_score']:.2f})")

    eval_steal = wm.evaluate_moral_choice(
        "Steal the golden claw",
        {'bounty': +50}
    )
    print(f"   Steal: {eval_steal['outcome_status']} (impact={eval_steal['impact_score']:.2f})")

    # Stats
    print(f"\n5. Stats: {wm.get_stats()}")

    print("\n✓ World model tests complete")
