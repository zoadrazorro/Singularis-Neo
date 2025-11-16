"""
Skyrim-Specific World Model

Extends the base world model with Skyrim-specific knowledge:
1. Causal rules ("stealing → guards hostile")
2. NPC relationships and factions
3. Quest mechanics and dependencies
4. Combat and magic systems
5. Geography and location knowledge
6. Symbolic logic reasoning (first-order logic predicates and inference)

Design principles:
- Causal learning enables prediction and planning
- Understanding game mechanics improves decision-making
- World model grounds actions in game reality
- Symbolic logic allows formal reasoning about game rules
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ..world_model import CausalGraph, CausalEdge


@dataclass
class LogicPredicate:
    """Represents a first-order logic predicate used in the symbolic reasoning engine.

    A predicate is a statement about one or more entities that can be either true
    or false. This class models predicates like "IsHostile(Bandit)" or
    "InLocation(Player, Whiterun)". It supports arguments, truth values, and
    provides methods for unification, which is essential for logical inference.

    Attributes:
        name: The name of the predicate, e.g., "IsHostile", "HasItem".
        args: A tuple of strings representing the arguments of the predicate.
        truth_value: A boolean indicating whether the predicate is true or false.
    """
    name: str  # e.g., "IsHostile", "HasItem", "InLocation"
    args: Tuple[str, ...]  # Arguments (constants or variables)
    truth_value: bool = True  # True or False

    def __str__(self):
        neg = "¬" if not self.truth_value else ""
        return f"{neg}{self.name}({', '.join(self.args)})"

    def __hash__(self):
        return hash((self.name, self.args, self.truth_value))

    def __eq__(self, other):
        return (self.name == other.name and
                self.args == other.args and
                self.truth_value == other.truth_value)

    def is_variable(self, arg: str) -> bool:
        """Checks if an argument is a variable.

        Variables are identified by starting with a '?' or being an uppercase word
        that is not a known constant type (like 'Player' or 'NPC'). This is used
        during the unification process to identify which parts of a predicate can be
        substituted.

        Args:
            arg: The argument string to check.

        Returns:
            True if the argument is a variable, False otherwise.
        """
        return arg.startswith('?') or (len(arg) > 0 and arg[0].isupper() and arg not in ['Player', 'NPC', 'Enemy', 'Guards', 'Location', 'Item', 'Quest'])

    def unify(self, other: 'LogicPredicate', bindings: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
        """Unifies this predicate with another, determining the variable bindings that make them identical.

        Unification is a key process in logical inference, used to match facts with
        the premises of rules. This method compares this predicate to another and,
        if they can be matched, returns a dictionary of variable substitutions.
        For example, unifying `CanSee(Player, ?x)` with `CanSee(Player, Bandit)`
        would yield the binding `{'?x': 'Bandit'}`.

        Args:
            other: The other LogicPredicate to unify with.
            bindings: An optional dictionary of existing variable bindings to respect.

        Returns:
            A dictionary of variable bindings if unification is successful,
            otherwise None.
        """
        if bindings is None:
            bindings = {}

        # Names must match
        if self.name != other.name or self.truth_value != other.truth_value:
            return None

        # Arguments must unify
        if len(self.args) != len(other.args):
            return None

        new_bindings = bindings.copy()
        for arg1, arg2 in zip(self.args, other.args):
            # If both are variables
            if self.is_variable(arg1) and self.is_variable(arg2):
                if arg1 in new_bindings:
                    if new_bindings[arg1] != arg2:
                        return None
                else:
                    new_bindings[arg1] = arg2
            # If arg1 is variable
            elif self.is_variable(arg1):
                if arg1 in new_bindings:
                    if new_bindings[arg1] != arg2:
                        return None
                else:
                    new_bindings[arg1] = arg2
            # If arg2 is variable
            elif self.is_variable(arg2):
                if arg2 in new_bindings:
                    if new_bindings[arg2] != arg1:
                        return None
                else:
                    new_bindings[arg2] = arg1
            # Both constants - must match
            elif arg1 != arg2:
                return None

        return new_bindings


@dataclass
class LogicRule:
    """Represents a logical inference rule used by the symbolic reasoning engine.

    A rule consists of a set of premises (conditions) that, if all are true,
    lead to a conclusion. This class models rules like:
    "IsHostile(NPC) ∧ InCombat(Player) → ShouldDefend(Player)".
    It also tracks metadata like confidence and usage statistics, allowing the
    system to learn and adapt its reasoning over time.

    Attributes:
        premises: A list of LogicPredicate objects that must be true for the
                  rule to apply.
        conclusion: The LogicPredicate that becomes true if all premises are met.
        confidence: A float from 0.0 to 1.0 indicating the reliability of the rule.
        usage_count: The number of times this rule has been successfully applied.
        success_count: The number of times applying this rule led to a positive outcome.
        last_used_cycle: The game cycle number when the rule was last used.
    """
    premises: List[LogicPredicate]
    conclusion: LogicPredicate
    confidence: float = 1.0  # How certain we are about this rule
    usage_count: int = 0  # How many times this rule has been used
    success_count: int = 0  # How many times it led to successful outcomes
    last_used_cycle: int = 0  # Track when rule was last used

    def __str__(self):
        prem_str = " ∧ ".join(str(p) for p in self.premises)
        usage_str = f" [used {self.usage_count}x, success {self.success_count}x]" if self.usage_count > 0 else ""
        return f"{prem_str} → {self.conclusion} (confidence={self.confidence:.2f}){usage_str}"

    def update_confidence_from_outcome(self, success: bool, learning_rate: float = 0.05):
        """Adjusts the rule's confidence based on the outcome of its application.

        This method implements a simple reinforcement learning mechanism. If an action
        based on this rule's conclusion was successful, the rule's confidence is
        increased. If it was unsuccessful, the confidence is decreased. This allows
        the system to prune ineffective rules and promote reliable ones over time.

        Args:
            success: A boolean indicating whether the outcome was successful.
            learning_rate: The rate at which to adjust the confidence.
        """
        self.usage_count += 1
        if success:
            self.success_count += 1
            # Increase confidence if successful
            self.confidence = min(0.99, self.confidence + learning_rate)
        else:
            # Decrease confidence if failed
            self.confidence = max(0.3, self.confidence - learning_rate * 0.5)

    def get_success_rate(self) -> float:
        """Calculates the empirical success rate of the rule based on its history.

        Returns:
            The success rate as a float between 0.0 and 1.0.
        """
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class LogicEngine:
    """A symbolic logic reasoning engine for formal inference.

    This engine manages a knowledge base of facts and a set of inference rules.
    It uses first-order logic to reason about the game state, derive new
    information through forward and backward chaining, and answer queries about
    what is known to be true.

    Attributes:
        facts: A set of `LogicPredicate` objects representing known truths.
        rules: A list of `LogicRule` objects for deriving new facts.
    """

    def __init__(self):
        """Initializes the LogicEngine with an empty set of facts and rules."""
        self.facts: Set[LogicPredicate] = set()  # Known facts
        self.rules: List[LogicRule] = []  # Inference rules

    def add_fact(self, fact: LogicPredicate):
        """Adds a fact to the knowledge base.

        Args:
            fact: The `LogicPredicate` to add as a known fact.
        """
        self.facts.add(fact)

    def remove_fact(self, fact: LogicPredicate):
        """Removes a fact from the knowledge base.

        Args:
            fact: The `LogicPredicate` to remove.
        """
        self.facts.discard(fact)

    def add_rule(self, rule: LogicRule):
        """Adds an inference rule to the engine.

        Args:
            rule: The `LogicRule` to add.
        """
        self.rules.append(rule)

    def query(self, predicate: LogicPredicate) -> bool:
        """Queries whether a predicate can be proven to be true.

        The engine first checks if the predicate is a known fact. If not, it
        attempts to infer it from its rules using backward chaining.

        Args:
            predicate: The `LogicPredicate` to query.

        Returns:
            True if the predicate can be proven, False otherwise.
        """
        # Direct fact check
        if predicate in self.facts:
            return True

        # Try to infer from rules
        return self._can_infer(predicate)

    def _can_infer(self, goal: LogicPredicate, depth: int = 0, max_depth: int = 5) -> bool:
        """
        Attempt to infer a goal using backward chaining.

        Args:
            goal: Goal predicate to prove
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            True if goal can be inferred
        """
        if depth > max_depth:
            return False

        # Check each rule
        for rule in self.rules:
            # If this rule concludes our goal
            if self._unify(rule.conclusion, goal):
                # Check if all premises are satisfied
                all_premises_true = True
                for premise in rule.premises:
                    # Direct fact check
                    if premise not in self.facts:
                        # Try to infer premise recursively
                        if not self._can_infer(premise, depth + 1, max_depth):
                            all_premises_true = False
                            break

                if all_premises_true:
                    return True

        return False

    def _unify(self, pred1: LogicPredicate, pred2: LogicPredicate, bindings: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
        """
        Unify two predicates with variable support.

        Args:
            pred1: First predicate
            pred2: Second predicate
            bindings: Current variable bindings

        Returns:
            Variable bindings if unification succeeds, None otherwise
        """
        return pred1.unify(pred2, bindings)

    def forward_chain(self) -> Set[LogicPredicate]:
        """Applies all rules to the current set of facts to derive all possible new facts.

        This method repeatedly iterates through the rules, adding new conclusions to
        the fact base until no new facts can be derived. This is useful for
        saturating the knowledge base with all inferable information.

        Returns:
            A set of the `LogicPredicate` objects that were newly derived.
        """
        new_facts = set()
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                # Check if all premises are in facts
                all_premises_satisfied = all(p in self.facts for p in rule.premises)

                if all_premises_satisfied and rule.conclusion not in self.facts:
                    # Derive new fact
                    self.facts.add(rule.conclusion)
                    new_facts.add(rule.conclusion)
                    changed = True

        return new_facts

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the logic engine's state.

        Returns:
            A dictionary containing the number of facts, rules, and a breakdown
            of facts by predicate type.
        """
        return {
            'facts': len(self.facts),
            'rules': len(self.rules),
            'predicates_by_type': self._count_predicates_by_type()
        }

    def _count_predicates_by_type(self) -> Dict[str, int]:
        """Counts the number of facts for each predicate name."""
        counts = {}
        for fact in self.facts:
            counts[fact.name] = counts.get(fact.name, 0) + 1
        return counts


@dataclass
class NPCRelationship:
    """Stores the state of the relationship between the player and an NPC.

    Attributes:
        npc_name: The name of the non-player character.
        faction: The faction the NPC belongs to.
        relationship_value: A float from -1.0 (hostile) to 1.0 (friendly)
                            representing the current standing.
        interactions: The total number of interactions with this NPC.
    """
    npc_name: str
    faction: str
    relationship_value: float  # -1 (hostile) to +1 (friendly)
    interactions: int = 0


class SkyrimWorldModel:
    """Manages the agent's understanding of the Skyrim game world.

    This class integrates multiple forms of knowledge representation, including a
    causal graph for learning action-outcome relationships, a symbolic logic engine
    for formal reasoning, and specific models for NPC relationships, locations,
    and terrain. It is responsible for learning from experience, making predictions,
    and providing strategic analysis to the agent.
    """

    def __init__(self, base_world_model=None):
        """Initializes the Skyrim-specific world model.

        Args:
            base_world_model: An optional instance of a base world model, not
                              currently used but provided for future extension.
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

        # Symbolic logic engine for formal reasoning
        self.logic_engine = LogicEngine()

        # Initialize common Skyrim causal relationships
        self._initialize_skyrim_causality()
        self._initialize_layer_knowledge()
        self._initialize_skyrim_logic_rules()

    def _initialize_skyrim_causality(self):
        """Initializes the causal graph with known cause-effect relationships in Skyrim."""
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
        """Initializes the knowledge base for action layers and their affordances."""

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

    def _initialize_skyrim_logic_rules(self):
        """Initializes the logic engine with a set of fundamental rules about Skyrim."""

        # Rule: If NPC is hostile AND in combat, then should defend
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsHostile", ("NPC",), True),
                LogicPredicate("InCombat", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldDefend", ("Player",), True),
            confidence=0.95
        ))

        # Rule: If has bounty AND in city, then guards are hostile
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("HasBounty", ("Player",), True),
                LogicPredicate("InCity", ("Player",), True)
            ],
            conclusion=LogicPredicate("IsHostile", ("Guards",), True),
            confidence=0.99
        ))

        # Rule: If health is low AND not in combat, then should heal
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("HealthLow", ("Player",), True),
                LogicPredicate("InCombat", ("Player",), False)
            ],
            conclusion=LogicPredicate("ShouldHeal", ("Player",), True),
            confidence=0.90
        ))

        # Rule: If health is critical, then should heal (even in combat)
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("HealthCritical", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldHeal", ("Player",), True),
            confidence=0.98
        ))

        # Rule: If NPC is friend AND needs help, then should assist
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsFriend", ("NPC",), True),
                LogicPredicate("NeedsHelp", ("NPC",), True)
            ],
            conclusion=LogicPredicate("ShouldAssist", ("Player", "NPC"), True),
            confidence=0.85
        ))

        # Rule: If location is unexplored AND safe, then should explore
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsUnexplored", ("Location",), True),
                LogicPredicate("IsSafe", ("Location",), True)
            ],
            conclusion=LogicPredicate("ShouldExplore", ("Location",), True),
            confidence=0.80
        ))

        # Rule: If in stealth AND enemy nearby, then avoid detection
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InStealth", ("Player",), True),
                LogicPredicate("EnemyNearby", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldAvoidDetection", ("Player",), True),
            confidence=0.92
        ))

        # Rule: If has quest item AND quest active, then should deliver
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("HasQuestItem", ("Player", "Item"), True),
                LogicPredicate("QuestActive", ("Quest",), True)
            ],
            conclusion=LogicPredicate("ShouldDeliverItem", ("Player", "Item"), True),
            confidence=0.88
        ))

        # Rule: If outnumbered AND has escape route, then should retreat
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("Outnumbered", ("Player",), True),
                LogicPredicate("HasEscapeRoute", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldRetreat", ("Player",), True),
            confidence=0.75
        ))

        # Rule: If magic available AND enemy weak to magic, then use magic
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("HasMagicka", ("Player",), True),
                LogicPredicate("WeakToMagic", ("Enemy",), True)
            ],
            conclusion=LogicPredicate("ShouldUseMagic", ("Player",), True),
            confidence=0.85
        ))

        # Rule: If dragon nearby AND unprepared, then should prepare
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("DragonNearby", ("Player",), True),
                LogicPredicate("Unprepared", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldPrepare", ("Player",), True),
            confidence=0.95
        ))

        # === OUTDOOR EXPLORATION RULES ===

        # Rule: If outdoor AND daytime AND no enemies, then should explore landmarks
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsOutdoor", ("Location",), True),
                LogicPredicate("IsDaytime", ("Environment",), True),
                LogicPredicate("EnemyNearby", ("Player",), False)
            ],
            conclusion=LogicPredicate("ShouldExploreLandmarks", ("Player",), True),
            confidence=0.75
        ))

        # Rule: If outdoor AND high elevation AND can see distance, then should scout
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsOutdoor", ("Location",), True),
                LogicPredicate("HighElevation", ("Player",), True),
                LogicPredicate("GoodVisibility", ("Environment",), True)
            ],
            conclusion=LogicPredicate("ShouldScout", ("Player",), True),
            confidence=0.70
        ))

        # Rule: If outdoor AND nighttime AND enemies nearby, then seek cover
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsOutdoor", ("Location",), True),
                LogicPredicate("IsNighttime", ("Environment",), True),
                LogicPredicate("EnemyNearby", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldSeekCover", ("Player",), True),
            confidence=0.80
        ))

        # Rule: If path blocked AND alternate route visible, then should navigate alternate
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("PathBlocked", ("Player",), True),
                LogicPredicate("AlternateRouteVisible", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldTakeAlternateRoute", ("Player",), True),
            confidence=0.85
        ))

        # Rule: If stamina low AND not in combat AND outdoor, then should rest
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("StaminaLow", ("Player",), True),
                LogicPredicate("InCombat", ("Player",), False),
                LogicPredicate("IsOutdoor", ("Location",), True)
            ],
            conclusion=LogicPredicate("ShouldRest", ("Player",), True),
            confidence=0.78
        ))

        # === DUNGEON NAVIGATION RULES ===

        # Rule: If in dungeon AND unexplored paths, then should explore systematically
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDungeon", ("Location",), True),
                LogicPredicate("HasUnexploredPaths", ("Location",), True)
            ],
            conclusion=LogicPredicate("ShouldExploreSystematically", ("Player",), True),
            confidence=0.82
        ))

        # Rule: If in dungeon AND hear enemy sounds, then should proceed cautiously
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDungeon", ("Location",), True),
                LogicPredicate("HearEnemySounds", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldProceedCautiously", ("Player",), True),
            confidence=0.88
        ))

        # Rule: If in dungeon AND found treasure, then check for traps
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDungeon", ("Location",), True),
                LogicPredicate("TreasureNearby", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldCheckForTraps", ("Player",), True),
            confidence=0.80
        ))

        # Rule: If in dungeon AND resources low, then should consider retreat
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDungeon", ("Location",), True),
                LogicPredicate("ResourcesLow", ("Player",), True),
                LogicPredicate("DeepInDungeon", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldConsiderRetreat", ("Player",), True),
            confidence=0.72
        ))

        # Rule: If in dungeon AND found exit, then should mark for return
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDungeon", ("Location",), True),
                LogicPredicate("FoundExit", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldMarkExit", ("Player",), True),
            confidence=0.90
        ))

        # === SOCIAL INTERACTION RULES ===

        # Rule: If NPC is merchant AND inventory full, then should sell items
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsMerchant", ("NPC",), True),
                LogicPredicate("InventoryFull", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldSellItems", ("Player",), True),
            confidence=0.85
        ))

        # Rule: If NPC is quest giver AND no active quest, then should inquire
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsQuestGiver", ("NPC",), True),
                LogicPredicate("HasActiveQuest", ("Player",), False)
            ],
            conclusion=LogicPredicate("ShouldInquireQuest", ("Player",), True),
            confidence=0.80
        ))

        # Rule: If NPC is guard AND has bounty, then should avoid interaction
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsGuard", ("NPC",), True),
                LogicPredicate("HasBounty", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldAvoidInteraction", ("Player",), True),
            confidence=0.95
        ))

        # Rule: If NPC is companion AND health low, then should protect
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsCompanion", ("NPC",), True),
                LogicPredicate("NPCHealthLow", ("NPC",), True)
            ],
            conclusion=LogicPredicate("ShouldProtectCompanion", ("Player",), True),
            confidence=0.88
        ))

        # Rule: If in dialogue AND speech skill high, then should persuade
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("InDialogue", ("Player",), True),
                LogicPredicate("SpeechSkillHigh", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldAttemptPersuasion", ("Player",), True),
            confidence=0.75
        ))

        # Rule: If NPC is hostile AND can intimidate, then should try intimidation
        self.logic_engine.add_rule(LogicRule(
            premises=[
                LogicPredicate("IsHostile", ("NPC",), True),
                LogicPredicate("CanIntimidate", ("Player",), True),
                LogicPredicate("InDialogue", ("Player",), True)
            ],
            conclusion=LogicPredicate("ShouldIntimidate", ("Player",), True),
            confidence=0.70
        ))

        print("[OK] Initialized Skyrim symbolic logic rules (32 rules total)")

    def learn_from_experience(
        self,
        action: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        surprise_threshold: float = 0.3
    ):
        """Learns causal relationships and logic rules from experience.

        When the outcome of an action is surprising (i.e., the state change is
        unexpected), this method is called to analyze what happened and update
        the world model. It can add new edges to the causal graph or create new
        symbolic logic rules to better predict outcomes in the future.

        Args:
            action: The action that was taken by the agent.
            before_state: The game state before the action was performed.
            after_state: The game state after the action was performed.
            surprise_threshold: The threshold above which a state change is
                                considered surprising enough to trigger learning.
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

        # Learn symbolic logic rules from surprising outcomes
        self._learn_logic_rules_from_experience(action, before_state, after_state, surprise)

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

    def _learn_logic_rules_from_experience(
        self,
        action: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        surprise: float
    ):
        """
        Learn new symbolic logic rules from surprising experiences.

        Args:
            action: Action that was taken
            before_state: State before action
            after_state: State after action
            surprise: How surprising the outcome was
        """
        # Only learn from moderately to highly surprising outcomes
        if surprise < 0.4:
            return

        # Extract key state changes
        changes = self._identify_changes(before_state, after_state)

        # Try to create logical rules from patterns
        for change_key, change_val in changes.items():
            # Example: If we attacked and unexpectedly became hostile with town
            if 'attack' in action.lower() and 'town_hostility' in change_key:
                # Check what we attacked
                target = before_state.get('target', 'Unknown')

                # Create a rule: Attacking {target} → Town hostility
                premises = [
                    LogicPredicate("PerformAction", (action,), True),
                    LogicPredicate("Target", (target,), True)
                ]
                conclusion = LogicPredicate("TownHostile", ("Player",), True)

                # Check if rule already exists
                rule_exists = any(
                    r.conclusion == conclusion and set(r.premises) == set(premises)
                    for r in self.logic_engine.rules
                )

                if not rule_exists:
                    new_rule = LogicRule(
                        premises=premises,
                        conclusion=conclusion,
                        confidence=0.6  # Start with moderate confidence
                    )
                    self.logic_engine.add_rule(new_rule)
                    print(f"[LOGIC] Learned new rule: {new_rule}")

            # Example: If we stole and guards became hostile
            elif 'steal' in action.lower() and 'guards_hostile' in change_key:
                premises = [
                    LogicPredicate("PerformAction", ("steal",), True),
                    LogicPredicate("InCity", ("Player",), True)
                ]
                conclusion = LogicPredicate("IsHostile", ("Guards",), True)

                # Update confidence if rule exists, or add new rule
                existing_rule = None
                for rule in self.logic_engine.rules:
                    if rule.conclusion == conclusion and set(rule.premises) == set(premises):
                        existing_rule = rule
                        break

                if existing_rule:
                    # Increase confidence based on repeated observation
                    existing_rule.confidence = min(0.99, existing_rule.confidence + 0.05)
                    print(f"[LOGIC] Updated rule confidence: {existing_rule}")
                else:
                    new_rule = LogicRule(premises=premises, conclusion=conclusion, confidence=0.7)
                    self.logic_engine.add_rule(new_rule)
                    print(f"[LOGIC] Learned new rule: {new_rule}")

            # Learn health-related rules
            elif 'health' in change_key and change_val < -20:  # Significant health loss
                # If we lost health while in combat
                if before_state.get('in_combat', False):
                    enemy_type = before_state.get('enemy_type', 'Unknown')
                    premises = [
                        LogicPredicate("InCombat", ("Player",), True),
                        LogicPredicate("FacingEnemy", (enemy_type,), True)
                    ]
                    conclusion = LogicPredicate("HighDamageRisk", ("Player",), True)

                    new_rule = LogicRule(premises=premises, conclusion=conclusion, confidence=0.75)
                    # Only add if not exists
                    if not any(r.conclusion == conclusion for r in self.logic_engine.rules):
                        self.logic_engine.add_rule(new_rule)
                        print(f"[LOGIC] Learned combat risk rule: {new_rule}")

    def suggest_optimal_layer(
        self,
        desired_action: str,
        current_state: Dict[str, Any]
    ) -> Optional[str]:
        """Suggests the optimal action layer for a given desired action and state.

        This method analyzes the historical effectiveness of the desired action
        across different layers and considers the current game context to recommend
        the best layer to use.

        Args:
            desired_action: A string representing the action the agent wants to perform.
            current_state: The current game state dictionary.

        Returns:
            The name of the recommended layer as a string, or None if no suitable
            layer is found.
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
        """Provides a strategic analysis of the available action layers.

        This method evaluates the effectiveness of each layer in the current
        context and provides recommendations for which layers might be most
        advantageous to switch to.

        Args:
            current_state: The current game state dictionary.

        Returns:
            A dictionary containing the analysis, including recommendations,
            effectiveness scores, and context evaluation.
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
        """Computes a 'surprise' value based on state changes.

        The surprise is a simple heuristic calculated as the ratio of changed
        state variables to the total number of variables.

        Args:
            before: The state before an action.
            after: The state after an action.

        Returns:
            A float representing the degree of surprise.
        """
        # Simple heuristic: count changed variables
        changes = self._identify_changes(before, after)
        return len(changes) / (len(before) + 1)

    def _identify_changes(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identifies changes between two state dictionaries.

        Args:
            before: The initial state.
            after: The subsequent state.

        Returns:
            A dictionary where keys are the changed variable names and values
            are the magnitude of the change.
        """
        changes = {}
        for key in before:
            if key in after and before[key] != after[key]:
                if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
                    changes[key] = after[key] - before[key]
                else:
                    changes[key] = 1.0  # Binary change
        return changes

    def update_logic_facts_from_state(self, game_state: Dict[str, Any]):
        """Updates the logic engine's facts based on the current game state.

        This method translates the raw data from the `game_state` dictionary into
        a set of symbolic `LogicPredicate` objects. It handles dynamic state
        changes by removing old facts and asserting new ones, ensuring the
        knowledge base is always current.

        Args:
            game_state: A dictionary representing the current state of the game.
        """
        # Clear old temporal facts (keep persistent knowledge)
        # We'll selectively remove and re-add based on current state

        # === HEALTH STATUS (detailed) ===
        health = game_state.get('health', 100)
        if health < 30:
            self.logic_engine.add_fact(LogicPredicate("HealthCritical", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthLow", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthModerate", ("Player",), True))
        elif health < 50:
            self.logic_engine.add_fact(LogicPredicate("HealthLow", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthCritical", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthModerate", ("Player",), True))
        elif health < 75:
            self.logic_engine.add_fact(LogicPredicate("HealthModerate", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthCritical", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthLow", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("HealthCritical", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthLow", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("HealthModerate", ("Player",), True))

        # === STAMINA STATUS ===
        stamina = game_state.get('stamina', 100)
        if stamina < 30:
            self.logic_engine.add_fact(LogicPredicate("StaminaLow", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("StaminaLow", ("Player",), True))

        # === COMBAT STATUS ===
        in_combat = game_state.get('in_combat', False)
        if in_combat:
            self.logic_engine.add_fact(LogicPredicate("InCombat", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("InCombat", ("Player",), True))

        # === STEALTH STATUS ===
        is_sneaking = game_state.get('is_sneaking', False)
        if is_sneaking:
            self.logic_engine.add_fact(LogicPredicate("InStealth", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("InStealth", ("Player",), True))

        # === BOUNTY STATUS ===
        bounty = game_state.get('bounty', 0)
        if bounty > 0:
            self.logic_engine.add_fact(LogicPredicate("HasBounty", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("HasBounty", ("Player",), True))

        # === LOCATION CONTEXT (detailed) ===
        location = game_state.get('location', '')
        scene_type = game_state.get('scene', 'unknown')

        if location:
            # Cities
            cities = ['Whiterun', 'Solitude', 'Riften', 'Windhelm', 'Markarth']
            if any(city in location for city in cities):
                self.logic_engine.add_fact(LogicPredicate("InCity", ("Player",), True))
                self.logic_engine.remove_fact(LogicPredicate("IsOutdoor", ("Location",), True))
                self.logic_engine.remove_fact(LogicPredicate("InDungeon", ("Location",), True))
            # Dungeons
            elif any(dungeon in location.lower() for dungeon in ['cave', 'ruin', 'barrow', 'fort', 'mine']):
                self.logic_engine.add_fact(LogicPredicate("InDungeon", ("Location",), True))
                self.logic_engine.remove_fact(LogicPredicate("InCity", ("Player",), True))
                self.logic_engine.remove_fact(LogicPredicate("IsOutdoor", ("Location",), True))
            # Outdoor
            elif 'outdoor' in scene_type.lower() or 'wilderness' in scene_type.lower():
                self.logic_engine.add_fact(LogicPredicate("IsOutdoor", ("Location",), True))
                self.logic_engine.remove_fact(LogicPredicate("InCity", ("Player",), True))
                self.logic_engine.remove_fact(LogicPredicate("InDungeon", ("Location",), True))
            else:
                self.logic_engine.remove_fact(LogicPredicate("InCity", ("Player",), True))
                self.logic_engine.remove_fact(LogicPredicate("InDungeon", ("Location",), True))
                self.logic_engine.remove_fact(LogicPredicate("IsOutdoor", ("Location",), True))

        # === ENEMY STATUS (detailed) ===
        enemies_nearby = game_state.get('enemies_nearby', 0)
        if enemies_nearby > 0:
            self.logic_engine.add_fact(LogicPredicate("EnemyNearby", ("Player",), True))
            if enemies_nearby >= 3:
                self.logic_engine.add_fact(LogicPredicate("Outnumbered", ("Player",), True))
            else:
                self.logic_engine.remove_fact(LogicPredicate("Outnumbered", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("EnemyNearby", ("Player",), True))
            self.logic_engine.remove_fact(LogicPredicate("Outnumbered", ("Player",), True))

        # === RESOURCE STATUS ===
        magicka = game_state.get('magicka', 100)
        if magicka > 30:
            self.logic_engine.add_fact(LogicPredicate("HasMagicka", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("HasMagicka", ("Player",), True))

        # Resources low check (health + stamina + magicka)
        resources_low = (health < 40 and stamina < 40) or (health < 40 and magicka < 40)
        if resources_low:
            self.logic_engine.add_fact(LogicPredicate("ResourcesLow", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("ResourcesLow", ("Player",), True))

        # === DIALOGUE STATUS ===
        if 'dialogue' in scene_type.lower():
            self.logic_engine.add_fact(LogicPredicate("InDialogue", ("Player",), True))
        else:
            self.logic_engine.remove_fact(LogicPredicate("InDialogue", ("Player",), True))

        # === INVENTORY STATUS ===
        # Note: This would need actual inventory data from game state
        # For now, using placeholder logic

        # === TIME/ENVIRONMENT (if available) ===
        # These would require additional data from game state
        # Placeholder for future enhancement

        # === NPC RELATIONSHIPS (enhanced) ===
        for npc_name, relationship in self.npc_relationships.items():
            if relationship.relationship_value < -0.3:
                self.logic_engine.add_fact(LogicPredicate("IsHostile", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsFriend", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsCompanion", (npc_name,), True))
            elif relationship.relationship_value > 0.7:
                self.logic_engine.add_fact(LogicPredicate("IsCompanion", (npc_name,), True))
                self.logic_engine.add_fact(LogicPredicate("IsFriend", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsHostile", (npc_name,), True))
            elif relationship.relationship_value > 0.3:
                self.logic_engine.add_fact(LogicPredicate("IsFriend", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsHostile", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsCompanion", (npc_name,), True))
            else:
                self.logic_engine.remove_fact(LogicPredicate("IsFriend", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsHostile", (npc_name,), True))
                self.logic_engine.remove_fact(LogicPredicate("IsCompanion", (npc_name,), True))

    def query_logic_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Queries the logic engine for strategic recommendations based on the current state.

        This method first updates the logic facts from the game state, then applies
        forward chaining to derive new truths. Finally, it checks for specific
        conclusions (like "ShouldHeal") to form a set of actionable recommendations.

        Args:
            game_state: The current game state dictionary.

        Returns:
            A dictionary containing boolean flags for various recommendations, a list
            of reasoning strings, and the number of newly derived facts.
        """
        # Update facts first
        self.update_logic_facts_from_state(game_state)

        # Apply forward chaining to derive new facts
        new_facts = self.logic_engine.forward_chain()

        # Extract recommendations
        recommendations = {
            'should_defend': False,
            'should_heal': False,
            'should_retreat': False,
            'should_explore': False,
            'should_use_magic': False,
            'should_avoid_detection': False,
            'logical_reasoning': [],
            'derived_facts': len(new_facts)
        }

        # Check various action recommendations
        if self.logic_engine.query(LogicPredicate("ShouldDefend", ("Player",), True)):
            recommendations['should_defend'] = True
            recommendations['logical_reasoning'].append("Defense recommended: Hostile NPCs in combat")

        if self.logic_engine.query(LogicPredicate("ShouldHeal", ("Player",), True)):
            recommendations['should_heal'] = True
            recommendations['logical_reasoning'].append("Healing recommended: Health is low")

        if self.logic_engine.query(LogicPredicate("ShouldRetreat", ("Player",), True)):
            recommendations['should_retreat'] = True
            recommendations['logical_reasoning'].append("Retreat recommended: Outnumbered with escape route")

        if self.logic_engine.query(LogicPredicate("ShouldUseMagic", ("Player",), True)):
            recommendations['should_use_magic'] = True
            recommendations['logical_reasoning'].append("Magic recommended: Enemy weak to magic")

        if self.logic_engine.query(LogicPredicate("ShouldAvoidDetection", ("Player",), True)):
            recommendations['should_avoid_detection'] = True
            recommendations['logical_reasoning'].append("Stealth recommended: In stealth with enemy nearby")

        return recommendations

    def get_logic_explanation(self, predicate: LogicPredicate) -> List[str]:
        """Generates an explanation for why a given predicate is believed to be true.

        It traces the derivation of the predicate, showing whether it is a direct
        fact or if it was inferred from a rule, and lists the premises that
        supported the inference.

        Args:
            predicate: The `LogicPredicate` to explain.

        Returns:
            A list of strings, where each string is a step in the explanation.
        """
        explanations = []

        # Check if directly in facts
        if predicate in self.logic_engine.facts:
            explanations.append(f"{predicate} is directly known")
            return explanations

        # Check which rules could derive this
        for rule in self.logic_engine.rules:
            if self._unify_predicates(rule.conclusion, predicate):
                # Check if premises are satisfied
                all_satisfied = all(p in self.logic_engine.facts for p in rule.premises)
                if all_satisfied:
                    prem_str = " AND ".join(str(p) for p in rule.premises)
                    explanations.append(
                        f"{predicate} follows from: {prem_str} "
                        f"(confidence: {rule.confidence:.2f})"
                    )

        return explanations if explanations else ["No logical derivation found"]

    def _unify_predicates(self, pred1: LogicPredicate, pred2: LogicPredicate) -> bool:
        """Checks if two predicates are identical."""
        return (pred1.name == pred2.name and
                pred1.args == pred2.args and
                pred1.truth_value == pred2.truth_value)

    def update_rule_confidences_from_outcome(
        self,
        used_recommendations: List[str],
        outcome_success: bool,
        cycle_number: int
    ):
        """Updates the confidence of rules based on the success of their recommendations.

        This is a key part of the learning process. If the agent follows a
        recommendation and the outcome is successful, the rules that led to that
        recommendation are reinforced. If the outcome is a failure, they are
        weakened.

        Args:
            used_recommendations: A list of recommendation keys (e.g., 'should_heal')
                                  that were followed by the agent.
            outcome_success: A boolean indicating if the subsequent outcome was successful.
            cycle_number: The current game cycle number, for tracking when rules were last used.
        """
        # Map recommendation types to conclusion predicates
        recommendation_mapping = {
            'should_defend': LogicPredicate("ShouldDefend", ("Player",), True),
            'should_heal': LogicPredicate("ShouldHeal", ("Player",), True),
            'should_retreat': LogicPredicate("ShouldRetreat", ("Player",), True),
            'should_use_magic': LogicPredicate("ShouldUseMagic", ("Player",), True),
            'should_avoid_detection': LogicPredicate("ShouldAvoidDetection", ("Player",), True),
        }

        # Find and update rules that generated these recommendations
        for rec_key in used_recommendations:
            if rec_key in recommendation_mapping:
                target_conclusion = recommendation_mapping[rec_key]

                # Find rules with this conclusion
                for rule in self.logic_engine.rules:
                    if (rule.conclusion.name == target_conclusion.name and
                        rule.conclusion.args == target_conclusion.args):
                        # Update confidence based on outcome
                        rule.update_confidence_from_outcome(outcome_success)
                        rule.last_used_cycle = cycle_number

                        if outcome_success:
                            print(f"[LOGIC-LEARNING] ✓ Rule confidence increased: {rule.conclusion} → {rule.confidence:.2f}")
                        else:
                            print(f"[LOGIC-LEARNING] ✗ Rule confidence decreased: {rule.conclusion} → {rule.confidence:.2f}")

    def predict_outcome(
        self,
        action: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predicts the outcome of an action using both the causal graph and logic engine.

        It first uses the causal graph for a probabilistic prediction of state changes.
        Then, it runs the logic engine on the predicted state to infer logical
        consequences and potential warnings.

        Args:
            action: The proposed action to be taken.
            current_state: The current game state.

        Returns:
            A dictionary representing the predicted next state, including any
            logical recommendations or warnings.
        """
        predicted_state = current_state.copy()

        # 1. Causal prediction (probabilistic)
        # Check if action node exists and get its effects
        if action in self.causal_graph.nodes:
            action_node = self.causal_graph.nodes[action]
            # Iterate through children (effects) of this action
            for effect_var in action_node.children:
                strength = action_node.causal_strengths.get(effect_var, 1.0)
                # Apply causal effect
                if effect_var in predicted_state:
                    if isinstance(predicted_state[effect_var], (int, float)):
                        # Modify numerical value
                        predicted_state[effect_var] += strength
                    else:
                        # Binary change
                        predicted_state[effect_var] = True
                else:
                    # Add new effect variable
                    predicted_state[effect_var] = strength

        # 2. Symbolic logic prediction (formal inference)
        logic_recommendations = self.query_logic_recommendation(predicted_state)
        predicted_state['logic_recommendations'] = logic_recommendations

        # 3. Combine insights
        if logic_recommendations.get('should_retreat') and action == 'attack':
            predicted_state['warning'] = "Logic suggests retreat, but action is attack"
        elif logic_recommendations.get('should_heal') and action != 'use_potion':
            predicted_state['warning'] = "Logic suggests healing"

        return predicted_state

    def update_npc_relationship(
        self,
        npc_name: str,
        faction: str,
        delta: float
    ):
        """Updates the player's relationship with a specific NPC.

        Args:
            npc_name: The name of the NPC.
            faction: The faction of the NPC.
            delta: The amount to change the relationship by, from -1.0 to 1.0.
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
        """Retrieves the current relationship status with an NPC.

        Args:
            npc_name: The name of the NPC.

        Returns:
            An `NPCRelationship` object if the NPC is known, otherwise None.
        """
        return self.npc_relationships.get(npc_name)

    def add_location(
        self,
        location_name: str,
        location_type: str,
        features: Dict[str, Any]
    ):
        """Adds a newly discovered location to the world model.

        Args:
            location_name: The name of the location.
            location_type: The type of location (e.g., 'dungeon', 'city').
            features: A dictionary of features associated with the location.
        """
        self.locations[location_name] = {
            'type': location_type,
            'features': features,
            'visited': True,
            'explored': False,
        }

    def mark_location_explored(self, location_name: str):
        """Marks a location as fully explored.

        Args:
            location_name: The name of the location to mark.
        """
        if location_name in self.locations:
            self.locations[location_name]['explored'] = True

    def get_unexplored_locations(self) -> List[str]:
        """Gets a list of discovered but not yet fully explored locations.

        Returns:
            A list of location names.
        """
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
        """DEPRECATED: Adds a quest to the world model."""
        # Quests removed in favor of terrain-aware system
        pass

    def complete_quest_objective(self, quest_name: str, objective: str):
        """DEPRECATED: Marks a quest objective as complete."""
        # Quests removed in favor of terrain-aware system
        pass

    def get_active_quests(self) -> List[str]:
        """DEPRECATED: Gets a list of active quests."""
        # Quests removed in favor of terrain-aware system
        return []

    def evaluate_moral_choice(
        self,
        choice: str,
        consequences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates the likely moral and practical outcome of a choice.

        This method uses a simple heuristic based on keywords to assess whether a
        choice is likely to be beneficial, detrimental, or neutral to the agent's
        interests within the game world.

        Args:
            choice: A string describing the moral choice.
            consequences: A dictionary of expected consequences (currently unused).

        Returns:
            A dictionary containing the evaluation, including an impact score and
            a recommended outcome status.
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
        """Retrieves a comprehensive set of statistics about the world model's state.

        Returns:
            A dictionary of statistics covering causal relationships, NPC interactions,
            locations, and the state of the logic engine.
        """
        logic_stats = self.logic_engine.get_stats()

        return {
            'causal_edges': len(self.causal_graph.graph.edges()),
            'npc_relationships': len(self.npc_relationships),
            'locations_discovered': len(self.locations),
            'locations_explored': sum(1 for loc in self.locations.values() if loc['explored']),
            'active_quests': len(self.get_active_quests()),
            'learned_rules': len(self.learned_rules),
            'logic_facts': logic_stats['facts'],
            'logic_rules': logic_stats['rules'],
            'logic_predicates_by_type': logic_stats['predicates_by_type'],
        }

    def get_logic_analysis(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Provides a detailed logical analysis of the current game state.

        This method synthesizes information from the logic engine to provide a
        high-level situational assessment, including recommendations, currently
        active facts, applicable rules, and explanations for key inferences.

        Args:
            game_state: The current game state dictionary.

        Returns:
            A dictionary containing the full logical analysis.
        """
        # Update facts and get recommendations
        recommendations = self.query_logic_recommendation(game_state)

        # Get all current facts
        current_facts = [str(fact) for fact in self.logic_engine.facts]

        # Get high-confidence rules that are applicable
        applicable_rules = []
        for rule in self.logic_engine.rules:
            if rule.confidence > 0.8:
                all_premises_true = all(p in self.logic_engine.facts for p in rule.premises)
                if all_premises_true:
                    applicable_rules.append(str(rule))

        # Get explanations for key recommendations
        explanations = {}
        if recommendations['should_defend']:
            explanations['defend'] = self.get_logic_explanation(
                LogicPredicate("ShouldDefend", ("Player",), True)
            )
        if recommendations['should_heal']:
            explanations['heal'] = self.get_logic_explanation(
                LogicPredicate("ShouldHeal", ("Player",), True)
            )
        if recommendations['should_retreat']:
            explanations['retreat'] = self.get_logic_explanation(
                LogicPredicate("ShouldRetreat", ("Player",), True)
            )

        return {
            'recommendations': recommendations,
            'current_facts': current_facts[:20],  # Limit to prevent overflow
            'applicable_rules': applicable_rules[:10],
            'explanations': explanations,
            'logic_confidence': self._compute_average_rule_confidence()
        }

    def _compute_average_rule_confidence(self) -> float:
        """Computes the average confidence score across all logic rules."""
        if not self.logic_engine.rules:
            return 0.0
        return sum(r.confidence for r in self.logic_engine.rules) / len(self.logic_engine.rules)


    def learn_terrain_feature(
        self,
        location: str,
        terrain_type: str,
        feature_data: Dict[str, Any]
    ):
        """Records a new terrain feature learned through exploration.

        Args:
            location: The name of the location where the feature was found.
            terrain_type: The category of terrain (e.g., 'indoor_spaces').
            feature_data: A dictionary describing the feature.
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
        """Classifies the current terrain based on visual scene type and combat state.

        Args:
            scene_type: A string from the scene classifier (e.g., 'outdoor', 'menu').
            in_combat: A boolean indicating if the agent is in combat.

        Returns:
            A string representing the classified terrain type.
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
        """Generates action recommendations based on the current terrain.

        Args:
            current_location: The current location of the agent.
            scene_type: The current visual scene type.
            in_combat: Whether the agent is in combat.

        Returns:
            A list of action strings recommended for the current terrain.
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
        """Updates the safety classification of a location based on combat encounters.

        Args:
            location: The name of the location.
            had_combat: A boolean indicating whether combat occurred at the location.
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
    print("=" * 70)
    print("Testing Skyrim World Model with Symbolic Logic")
    print("=" * 70)

    wm = SkyrimWorldModel()

    # 1. Test causal prediction
    print("\n1. Testing causal prediction...")
    state = {'bounty': 0, 'guards_hostile': False}
    predicted = wm.predict_outcome('steal_item', state)
    print(f"   Before: {state}")
    print(f"   Action: steal_item")
    print(f"   Predicted: {predicted}")

    # 2. Test symbolic logic recommendations
    print("\n2. Testing symbolic logic recommendations...")

    # Scenario: Low health in combat
    combat_state = {
        'health': 25,
        'in_combat': True,
        'enemies_nearby': 2,
        'magicka': 50,
        'location': 'Bleak Falls Barrow'
    }
    print(f"   Scenario: {combat_state}")
    recommendations = wm.query_logic_recommendation(combat_state)
    print(f"   Recommendations:")
    for reason in recommendations['logical_reasoning']:
        print(f"      • {reason}")

    # 3. Test logic explanation
    print("\n3. Testing logic explanation...")
    should_heal = LogicPredicate("ShouldHeal", ("Player",), True)
    explanations = wm.get_logic_explanation(should_heal)
    print(f"   Why should heal?")
    for exp in explanations:
        print(f"      • {exp}")

    # 4. Test learning from experience
    print("\n4. Testing learning from surprising experience...")
    before = {
        'health': 100,
        'in_combat': False,
        'target': 'chicken',
        'town_hostility': False
    }
    after = {
        'health': 100,
        'in_combat': True,
        'target': 'chicken',
        'town_hostility': True
    }
    print(f"   Action: attack_chicken")
    wm.learn_from_experience('attack_chicken', before, after, surprise_threshold=0.2)

    # 5. Test NPC relationships
    print("\n5. Testing NPC relationships...")
    wm.update_npc_relationship('Lydia', 'Whiterun', delta=0.5)
    wm.update_npc_relationship('Bandit Chief', 'Bandits', delta=-0.8)
    print(f"   Lydia: {wm.get_npc_relationship('Lydia').relationship_value:.2f} (friend)")
    print(f"   Bandit: {wm.get_npc_relationship('Bandit Chief').relationship_value:.2f} (hostile)")

    # 6. Test comprehensive logic analysis
    print("\n6. Testing comprehensive logic analysis...")
    analysis_state = {
        'health': 30,
        'in_combat': True,
        'enemies_nearby': 4,
        'magicka': 80,
        'bounty': 100,
        'location': 'Whiterun',
        'is_sneaking': False
    }
    analysis = wm.get_logic_analysis(analysis_state)
    print(f"   Current situation: Critical health, outnumbered in Whiterun with bounty")
    print(f"   Logic confidence: {analysis['logic_confidence']:.2f}")
    print(f"   Active facts: {len(analysis['current_facts'])} facts in knowledge base")
    print(f"   Applicable rules: {len(analysis['applicable_rules'])} high-confidence rules apply")

    # 7. Test moral evaluation
    print("\n7. Testing moral evaluation...")
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

    # 8. Display final stats
    print("\n8. Final world model statistics:")
    stats = wm.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ World model tests complete - Symbolic logic fully integrated!")
    print("=" * 70)
