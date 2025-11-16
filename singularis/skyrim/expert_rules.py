"""
Expert Rule System for Skyrim AGI

Implements a rule-based expert system for detecting problems and recommending actions.
Rules fire based on conditions and can:
- Set facts in working memory
- Recommend actions with priorities
- Block actions for N cycles
- Adjust system parameters

This provides fast, deterministic responses to known patterns before expensive LLM reasoning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time


class Priority(Enum):
    """Enumerates the priority levels for action recommendations."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class Fact:
    """Represents a fact asserted in the rule engine's working memory.

    Attributes:
        name: The unique name of the fact.
        confidence: The confidence level of the fact, from 0.0 to 1.0.
        timestamp: The time the fact was asserted.
        metadata: A dictionary for storing additional data related to the fact.
    """
    name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Represents an action recommended by a rule.

    Attributes:
        action: The name of the recommended action.
        priority: The priority level of the recommendation.
        reason: A string explaining why the action was recommended.
        confidence: The confidence level of the recommendation.
        timestamp: The time the recommendation was made.
    """
    action: str
    priority: Priority
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionBlock:
    """Represents a temporary block on a specific action.

    Attributes:
        action: The name of the action to block.
        cycles_remaining: The number of game cycles for which the block is active.
        reason: A string explaining why the action was blocked.
    """
    action: str
    cycles_remaining: int
    reason: str


@dataclass
class Rule:
    """Represents a single rule in the expert system.

    Attributes:
        name: The unique name of the rule.
        condition: A callable that takes the current context and returns True if
                   the rule should fire.
        consequences: A list of callables to be executed when the rule fires.
        description: A human-readable description of the rule's purpose.
        priority: The priority of the rule (higher values are evaluated first).
    """
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    consequences: List[Callable[[Dict[str, Any], 'RuleEngine'], None]]
    description: str
    priority: int = 1


class RuleEngine:
    """A rule-based expert system engine for making fast, deterministic decisions.

    This engine maintains a working memory of 'Facts' and evaluates a set of 'Rules'
    against the current game context. Firing rules can assert new facts, recommend
    actions, block actions, or adjust system parameters. It is designed to handle
    common, recognizable situations without needing to consult a more complex model.
    """
    
    def __init__(self):
        """Initializes the RuleEngine, setting up data structures and registering core rules."""
        self.rules: List[Rule] = []
        self.facts: Dict[str, Fact] = {}
        self.recommendations: List[Recommendation] = []
        self.blocked_actions: Dict[str, ActionBlock] = {}
        self.parameters: Dict[str, float] = {}
        
        # Statistics
        self.rules_fired_count = 0
        self.rules_fired_history: List[str] = []
        
        # Cycle tracking - external systems should call tick_cycle()
        self.last_evaluate_cycle = -1
        
        # Initialize with core rules
        self._register_core_rules()
    
    def _register_core_rules(self):
        """Registers the set of built-in expert rules for common Skyrim situations."""
        
        # Rule 1: Stuck in exploration loop
        self.add_rule(Rule(
            name="stuck_in_loop",
            description="Detect when stuck in repetitive exploration with high visual similarity",
            priority=10,
            condition=self._cond_stuck_in_loop,
            consequences=[
                self._cons_set_stuck_fact,
                self._cons_recommend_retreat,
                self._cons_recommend_activate,
                self._cons_block_explore
            ]
        ))
        
        # Rule 2: Scene classification mismatch
        self.add_rule(Rule(
            name="scene_mismatch",
            description="Detect when scene classification doesn't match visual description",
            priority=9,
            condition=self._cond_scene_mismatch,
            consequences=[
                self._cons_set_sensory_conflict,
                self._cons_recommend_activate_high,
                self._cons_increase_sensorimotor_authority
            ]
        ))
        
        # Rule 3: High visual similarity without action repetition (potential soft-lock)
        self.add_rule(Rule(
            name="visual_stasis",
            description="Detect when visuals don't change even with varied actions",
            priority=8,
            condition=self._cond_visual_stasis,
            consequences=[
                self._cons_set_visual_stasis_fact,
                self._cons_recommend_jump,
                self._cons_recommend_turn_around
            ]
        ))
        
        # Rule 4: Rapid action switching (indecision)
        self.add_rule(Rule(
            name="action_thrashing",
            description="Detect when actions change too rapidly without settling",
            priority=7,
            condition=self._cond_action_thrashing,
            consequences=[
                self._cons_set_indecision_fact,
                self._cons_recommend_commit_to_action
            ]
        ))
        
        # Rule 5: Low coherence with explore actions
        self.add_rule(Rule(
            name="unproductive_exploration",
            description="Detect exploration that doesn't improve coherence",
            priority=6,
            condition=self._cond_unproductive_exploration,
            consequences=[
                self._cons_set_unproductive_fact,
                self._cons_recommend_goal_change
            ]
        ))
    
    # ============================================================================
    # RULE CONDITIONS
    # ============================================================================
    
    def _cond_stuck_in_loop(self, context: Dict[str, Any]) -> bool:
        """Condition for the 'stuck_in_loop' rule.

        Fires if visual similarity is high and the agent has been repeatedly
        trying to 'explore'.
        """
        visual_similarity = context.get('visual_similarity', 0.0)
        recent_actions = context.get('recent_actions', [])
        
        if visual_similarity > 0.95:
            explore_count = sum(1 for a in recent_actions[-5:] if 'explore' in a.lower())
            if explore_count > 2:
                return True
        return False
    
    def _cond_scene_mismatch(self, context: Dict[str, Any]) -> bool:
        """Condition for the 'scene_mismatch' rule.

        Fires if the high-level scene classification (e.g., from an LLM)
        disagrees with the low-level visual scene type (e.g., from a CNN).
        """
        scene_classification = context.get('scene_classification')
        visual_scene_type = context.get('visual_scene_type')
        
        if scene_classification and visual_scene_type:
            # Compare, handling enum values
            scene_class_str = str(scene_classification).lower().replace('scenetype.', '')
            visual_str = str(visual_scene_type).lower().replace('scenetype.', '')
            
            if scene_class_str != visual_str and scene_class_str != 'unknown' and visual_str != 'unknown':
                return True
        return False
    
    def _cond_visual_stasis(self, context: Dict[str, Any]) -> bool:
        """Condition for the 'visual_stasis' rule.

        Fires if visual similarity is very high, even though the agent has
        been trying a variety of different actions. This might indicate a
        soft-lock or non-obvious obstacle.
        """
        visual_similarity = context.get('visual_similarity', 0.0)
        recent_actions = context.get('recent_actions', [])
        
        if visual_similarity > 0.97 and len(recent_actions) >= 4:
            # Check if actions are varied
            unique_actions = len(set(recent_actions[-4:]))
            if unique_actions >= 3:  # Different actions but same visuals
                return True
        return False
    
    def _cond_action_thrashing(self, context: Dict[str, Any]) -> bool:
        """Condition for the 'action_thrashing' rule.

        Fires if the agent is rapidly switching between many different actions,
        indicating indecision or planning instability.
        """
        recent_actions = context.get('recent_actions', [])
        
        if len(recent_actions) >= 5:
            # Check if last 5 actions are all different
            if len(set(recent_actions[-5:])) == 5:
                return True
        return False
    
    def _cond_unproductive_exploration(self, context: Dict[str, Any]) -> bool:
        """Condition for the 'unproductive_exploration' rule.

        Fires if the agent is repeatedly exploring but its internal coherence
        (a measure of understanding) is not improving, suggesting the exploration
        is not fruitful.
        """
        recent_actions = context.get('recent_actions', [])
        coherence_history = context.get('coherence_history', [])
        
        if len(recent_actions) >= 5 and len(coherence_history) >= 5:
            explore_count = sum(1 for a in recent_actions[-5:] if 'explore' in a.lower())
            
            if explore_count >= 3:
                # Check if coherence is stagnant or declining
                coherence_change = coherence_history[-1] - coherence_history[-5]
                if coherence_change < 0.01:  # Less than 1% improvement
                    return True
        return False
    
    # ============================================================================
    # RULE CONSEQUENCES
    # ============================================================================
    
    def _cons_set_stuck_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts a 'stuck_in_loop' fact into working memory."""
        engine.set_fact("stuck_in_loop", confidence=0.95, metadata={
            'visual_similarity': context.get('visual_similarity'),
            'explore_count': sum(1 for a in context.get('recent_actions', [])[-5:] if 'explore' in a.lower())
        })
    
    def _cons_recommend_retreat(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends moving backward to get unstuck."""
        engine.recommend("move_backward", Priority.HIGH, "Stuck in loop - retreat", confidence=0.9)
    
    def _cons_recommend_activate(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends using the 'activate' action to interact with a potential obstacle."""
        engine.recommend("activate", Priority.MEDIUM, "Try pressing/activating obstacle", confidence=0.8)
    
    def _cons_block_explore(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Blocks the 'explore' action for 3 cycles to prevent looping."""
        engine.block_action("explore", duration=3, reason="Stuck in exploration loop")
    
    def _cons_set_sensory_conflict(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts a 'sensory_conflict' fact into working memory."""
        engine.set_fact("sensory_conflict", confidence=0.90, metadata={
            'scene_classification': context.get('scene_classification'),
            'visual_scene_type': context.get('visual_scene_type')
        })
    
    def _cons_recommend_activate_high(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends 'activate' with high priority to resolve a sensory mismatch."""
        engine.recommend("activate", Priority.HIGH, "Resolve scene classification mismatch", confidence=0.85)
    
    def _cons_increase_sensorimotor_authority(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Increases the 'sensorimotor_authority' parameter to trust low-level vision more."""
        current = engine.parameters.get('sensorimotor_authority', 1.0)
        engine.set_parameter('sensorimotor_authority', current * 1.5)
    
    def _cons_set_visual_stasis_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts a 'visual_stasis' fact into working memory."""
        engine.set_fact("visual_stasis", confidence=0.85, metadata={
            'visual_similarity': context.get('visual_similarity')
        })
    
    def _cons_recommend_jump(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends jumping to try and break a soft-lock."""
        engine.recommend("jump", Priority.HIGH, "Break visual stasis", confidence=0.8)
    
    def _cons_recommend_turn_around(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends turning around to get a new perspective."""
        engine.recommend("turn_around", Priority.MEDIUM, "Change perspective", confidence=0.75)
    
    def _cons_set_indecision_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts an 'action_thrashing' fact into working memory."""
        engine.set_fact("action_thrashing", confidence=0.80)
    
    def _cons_recommend_commit_to_action(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Recommends repeating the last action to break indecision."""
        # Get most recent action and stick with it
        recent_actions = context.get('recent_actions', [])
        if recent_actions:
            action = recent_actions[-1]
            engine.recommend(action, Priority.MEDIUM, "Commit to action sequence", confidence=0.7)
    
    def _cons_set_unproductive_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts an 'unproductive_exploration' fact into working memory."""
        engine.set_fact("unproductive_exploration", confidence=0.75)
    
    def _cons_recommend_goal_change(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Consequence: Asserts a meta-fact suggesting the high-level goal should be revised."""
        # This is a meta-recommendation that should trigger goal replanning
        engine.set_fact("needs_goal_revision", confidence=0.8)
    
    # ============================================================================
    # PUBLIC API
    # ============================================================================
    
    def add_rule(self, rule: Rule):
        """Adds a new rule to the engine and sorts the rule list by priority.

        Args:
            rule: The Rule object to add.
        """
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def set_fact(self, name: str, confidence: float, metadata: Optional[Dict[str, Any]] = None):
        """Asserts a fact into the working memory, overwriting any existing fact with the same name.

        Args:
            name: The name of the fact.
            confidence: The confidence score for the fact (0.0 to 1.0).
            metadata: Optional dictionary of related data.
        """
        self.facts[name] = Fact(
            name=name,
            confidence=confidence,
            metadata=metadata or {}
        )
    
    def get_fact(self, name: str) -> Optional[Fact]:
        """Retrieves a fact from working memory by its name.

        Args:
            name: The name of the fact to retrieve.

        Returns:
            The Fact object, or None if it doesn't exist.
        """
        return self.facts.get(name)
    
    def has_fact(self, name: str, min_confidence: float = 0.5) -> bool:
        """Checks if a fact exists in working memory with at least a minimum confidence.

        Args:
            name: The name of the fact to check.
            min_confidence: The minimum required confidence level.

        Returns:
            True if the fact exists with sufficient confidence, False otherwise.
        """
        fact = self.facts.get(name)
        return fact is not None and fact.confidence >= min_confidence
    
    def recommend(self, action: str, priority: Priority, reason: str, confidence: float):
        """Adds an action recommendation to the current list of recommendations for this cycle.

        Args:
            action: The name of the action to recommend.
            priority: The priority of the recommendation.
            reason: The reason for the recommendation.
            confidence: The confidence score for the recommendation.
        """
        self.recommendations.append(Recommendation(
            action=action,
            priority=priority,
            reason=reason,
            confidence=confidence
        ))
    
    def is_context_appropriate(self, action: str, context: Dict[str, Any]) -> bool:
        """Checks if a given action is appropriate for the current game context.

        This crucial method prevents the system from recommending nonsensical
        actions, such as attacking while in a dialogue menu or trying to move
        while the inventory is open.

        Args:
            action: The action to check (e.g., 'attack').
            context: The current game state context, including scene type.

        Returns:
            True if the action is appropriate for the context, False otherwise.
        """
        scene_type = context.get('scene_classification') or context.get('visual_scene_type', '')
        in_combat = context.get('in_combat', False)
        in_menu = 'inventory' in str(scene_type).lower() or 'map' in str(scene_type).lower()
        in_dialogue = 'dialogue' in str(scene_type).lower()
        
        # Combat actions inappropriate in menus/dialogues
        combat_actions = ['attack', 'power_attack', 'block', 'bash', 'shout', 'cast_spell']
        if action in combat_actions and (in_menu or in_dialogue):
            return False
        
        # Healing/potion use inappropriate in menus (need to exit first)
        healing_actions = ['heal', 'use_potion', 'drink_potion', 'restore_health']
        if action in healing_actions and in_menu:
            return False
        
        # Movement inappropriate in dialogue
        movement_actions = ['move_forward', 'move_backward', 'strafe_left', 'strafe_right', 'sprint']
        if action in movement_actions and in_dialogue:
            return False
        
        # Interaction with objects inappropriate during combat
        interaction_actions = ['activate', 'take', 'open_container', 'read_book']
        if action in interaction_actions and in_combat:
            return False
        
        # Stealth actions inappropriate in menus/dialogues
        stealth_actions = ['sneak', 'pickpocket', 'lockpick']
        if action in stealth_actions and (in_menu or in_dialogue):
            return False
        
        return True
    
    def filter_recommendations_by_context(self, context: Dict[str, Any]):
        """Filters the current list of recommendations to remove any that are context-inappropriate.

        This should be called after `evaluate()` and before `get_top_recommendation()`.

        Args:
            context: The current game state and scene context.
        """
        original_count = len(self.recommendations)
        filtered = []
        removed = []
        
        for rec in self.recommendations:
            if self.is_context_appropriate(rec.action, context):
                filtered.append(rec)
            else:
                removed.append(rec)
        
        if removed:
            print(f"\n[RULES] Context filtering removed {len(removed)} inappropriate recommendations:")
            for rec in removed:
                scene = context.get('scene_classification', 'unknown')
                print(f"  ✗ {rec.action} (priority: {rec.priority.name})")
                print(f"    Reason: Inappropriate for scene '{scene}'")
        
        self.recommendations = filtered
        return len(removed)
    
    def block_action(self, action: str, duration: int, reason: str):
        """Blocks a specified action for a given number of cycles.

        Args:
            action: The name of the action to block.
            duration: The number of cycles to block the action for.
            reason: The reason for the block.
        """
        self.blocked_actions[action] = ActionBlock(
            action=action,
            cycles_remaining=duration,
            reason=reason
        )
    
    def is_action_blocked(self, action: str) -> bool:
        """Checks if an action is currently blocked.

        Args:
            action: The name of the action to check.

        Returns:
            True if the action is blocked, False otherwise.
        """
        block = self.blocked_actions.get(action)
        return block is not None and block.cycles_remaining > 0
    
    def set_parameter(self, name: str, value: float):
        """Sets a system-wide parameter that can be used by rules or other systems.

        Args:
            name: The name of the parameter.
            value: The float value to set.
        """
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: float = 1.0) -> float:
        """Retrieves a system parameter.

        Args:
            name: The name of the parameter.
            default: The default value to return if the parameter is not set.

        Returns:
            The value of the parameter or the default.
        """
        return self.parameters.get(name, default)
    
    def tick_cycle(self):
        """Advances the engine's internal cycle counter.

        This must be called once per game cycle to correctly manage time-based
        features like action blocks.
        """
        # Decrement blocked action counters
        expired_blocks = []
        for action, block in self.blocked_actions.items():
            block.cycles_remaining -= 1
            if block.cycles_remaining <= 0:
                expired_blocks.append(action)
        
        # Remove expired blocks
        for action in expired_blocks:
            del self.blocked_actions[action]
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates all registered rules against the current context.

        This method clears previous recommendations, iterates through all rules
        in priority order, and executes the consequences of any rule whose
        condition is met.

        Args:
            context: A dictionary representing the current game state and metrics.

        Returns:
            A dictionary summarizing the results of the evaluation, including
            which rules fired, the final list of recommendations, and current facts.
        """
        # Clear previous cycle's recommendations
        self.recommendations.clear()
        
        # Evaluate rules in priority order
        fired_rules = []
        
        for rule in self.rules:
            try:
                if rule.condition(context):
                    # Rule fires - execute consequences
                    for consequence in rule.consequences:
                        consequence(context, self)
                    
                    fired_rules.append(rule.name)
                    self.rules_fired_count += 1
                    self.rules_fired_history.append(rule.name)
                    
                    # Keep history bounded
                    if len(self.rules_fired_history) > 100:
                        self.rules_fired_history.pop(0)
                    
            except Exception as e:
                print(f"[RULE-ENGINE] Error evaluating rule '{rule.name}': {e}")
        
        return {
            'fired_rules': fired_rules,
            'recommendations': sorted(self.recommendations, key=lambda r: r.priority.value, reverse=True),
            'blocked_actions': list(self.blocked_actions.keys()),
            'facts': {name: {'confidence': f.confidence, 'metadata': f.metadata} for name, f in self.facts.items()},
            'parameters': self.parameters.copy(),
            'rules_fired_total': self.rules_fired_count
        }
    
    def get_top_recommendation(self, exclude_blocked: bool = True) -> Optional[Recommendation]:
        """Retrieves the single highest-priority recommendation from the last evaluation.

        Args:
            exclude_blocked: If True (default), blocked actions will not be returned.

        Returns:
            The highest-priority Recommendation object, or None if there are no
            valid recommendations.
        """
        recommendations = sorted(self.recommendations, key=lambda r: (r.priority.value, r.confidence), reverse=True)
        
        if exclude_blocked:
            recommendations = [r for r in recommendations if not self.is_action_blocked(r.action)]
        
        return recommendations[0] if recommendations else None
    
    def clear_facts(self, max_age_seconds: Optional[float] = None):
        """Clears facts from working memory.

        Args:
            max_age_seconds: If provided, only facts older than this age will be
                             cleared. If None, all facts are cleared.
        """
        if max_age_seconds is None:
            self.facts.clear()
        else:
            current_time = time.time()
            expired = [name for name, fact in self.facts.items() 
                      if current_time - fact.timestamp > max_age_seconds]
            for name in expired:
                del self.facts[name]
    
    def get_status_report(self) -> str:
        """Generates a human-readable string report of the engine's current state.

        Returns:
            A formatted string detailing active facts, recommendations, and blocks.
        """
        lines = [
            "═══════════════════════════════════════════════════════════",
            "                    RULE ENGINE STATUS                     ",
            "═══════════════════════════════════════════════════════════",
            f"Total Rules: {len(self.rules)}",
            f"Rules Fired (Total): {self.rules_fired_count}",
            f"Active Facts: {len(self.facts)}",
            f"Active Recommendations: {len(self.recommendations)}",
            f"Blocked Actions: {len(self.blocked_actions)}",
            ""
        ]
        
        if self.facts:
            lines.append("Active Facts:")
            for name, fact in self.facts.items():
                lines.append(f"  • {name} (confidence: {fact.confidence:.2f})")
            lines.append("")
        
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in sorted(self.recommendations, key=lambda r: r.priority.value, reverse=True):
                lines.append(f"  • [{rec.priority.name}] {rec.action} - {rec.reason} ({rec.confidence:.2f})")
            lines.append("")
        
        if self.blocked_actions:
            lines.append("Blocked Actions:")
            for action, block in self.blocked_actions.items():
                lines.append(f"  • {action} for {block.cycles_remaining} cycles - {block.reason}")
            lines.append("")
        
        if self.rules_fired_history:
            lines.append(f"Recent Rules Fired (last 10):")
            for rule_name in self.rules_fired_history[-10:]:
                lines.append(f"  • {rule_name}")
        
        lines.append("═══════════════════════════════════════════════════════════")
        return "\n".join(lines)
