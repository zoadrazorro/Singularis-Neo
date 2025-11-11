"""
Skyrim-Specific World Model

Extends the base world model with Skyrim-specific knowledge:
1. Causal rules ("stealing → guards hostile")
2. NPC relationships and factions
3. Quest mechanics and dependencies
4. Combat and magic systems
5. Geography and location knowledge

Philosophical grounding:
- ETHICA: Understanding causality = increasing adequacy = increasing freedom
- Causal learning enables prediction and planning
- World model grounds agency in reality
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

        # Known locations
        self.locations: Dict[str, Dict[str, Any]] = {}

        # Quest state
        self.quests: Dict[str, Dict[str, Any]] = {}

        # Learned rules
        self.learned_rules: List[Dict[str, Any]] = []

        # Initialize common Skyrim causal relationships
        self._initialize_skyrim_causality()

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

        print("✓ Initialized Skyrim causal relationships")

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
        """Add quest."""
        self.quests[quest_name] = {
            'type': quest_type,
            'objectives': objectives,
            'completed_objectives': [],
            'status': 'active',
        }

    def complete_quest_objective(self, quest_name: str, objective: str):
        """Complete quest objective."""
        if quest_name in self.quests:
            self.quests[quest_name]['completed_objectives'].append(objective)

            # Check if quest complete
            if len(self.quests[quest_name]['completed_objectives']) >= len(self.quests[quest_name]['objectives']):
                self.quests[quest_name]['status'] = 'completed'

    def get_active_quests(self) -> List[str]:
        """Get active quest names."""
        return [
            name for name, quest in self.quests.items()
            if quest['status'] == 'active'
        ]

    def evaluate_moral_choice(
        self,
        choice: str,
        consequences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate moral choice using coherence (Δ𝒞).

        Args:
            choice: Description of choice
            consequences: Expected consequences

        Returns:
            Evaluation with coherence delta
        """
        # Estimate coherence change
        # Negative actions (stealing, murder) decrease coherence
        # Helpful actions increase coherence

        negative_keywords = ['steal', 'kill', 'murder', 'betray', 'lie']
        positive_keywords = ['help', 'save', 'heal', 'defend', 'protect']

        choice_lower = choice.lower()

        # Heuristic coherence estimate
        delta_coherence = 0.0

        for keyword in negative_keywords:
            if keyword in choice_lower:
                delta_coherence -= 0.1

        for keyword in positive_keywords:
            if keyword in choice_lower:
                delta_coherence += 0.1

        # Evaluate
        if delta_coherence > 0.02:
            ethical_status = "ETHICAL"
        elif abs(delta_coherence) < 0.02:
            ethical_status = "NEUTRAL"
        else:
            ethical_status = "UNETHICAL"

        return {
            'choice': choice,
            'delta_coherence': delta_coherence,
            'ethical_status': ethical_status,
            'recommendation': ethical_status == "ETHICAL"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            'causal_edges': len(self.causal_graph.edges),
            'npc_relationships': len(self.npc_relationships),
            'locations_discovered': len(self.locations),
            'locations_explored': sum(1 for loc in self.locations.values() if loc['explored']),
            'active_quests': len(self.get_active_quests()),
            'learned_rules': len(self.learned_rules),
        }


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
    print(f"   Help: {eval_help['ethical_status']} (Δ𝒞={eval_help['delta_coherence']:.2f})")

    eval_steal = wm.evaluate_moral_choice(
        "Steal the golden claw",
        {'bounty': +50}
    )
    print(f"   Steal: {eval_steal['ethical_status']} (Δ𝒞={eval_steal['delta_coherence']:.2f})")

    # Stats
    print(f"\n5. Stats: {wm.get_stats()}")

    print("\n✓ World model tests complete")
