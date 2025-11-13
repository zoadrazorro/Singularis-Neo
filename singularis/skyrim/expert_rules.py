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
    """Action priority levels."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class Fact:
    """A fact asserted in working memory."""
    name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """An action recommendation."""
    action: str
    priority: Priority
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionBlock:
    """Block an action for N cycles."""
    action: str
    cycles_remaining: int
    reason: str


@dataclass
class Rule:
    """A rule with condition and consequences."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    consequences: List[Callable[[Dict[str, Any], 'RuleEngine'], None]]
    description: str
    priority: int = 1  # Higher priority rules fire first


class RuleEngine:
    """
    Rule-based expert system engine.
    
    Maintains working memory (facts) and fires rules based on conditions.
    """
    
    def __init__(self):
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
        """Register core expert rules."""
        
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
        """
        Condition: visual_similarity > 0.95 AND recent_actions.count("explore") > 2
        """
        visual_similarity = context.get('visual_similarity', 0.0)
        recent_actions = context.get('recent_actions', [])
        
        if visual_similarity > 0.95:
            explore_count = sum(1 for a in recent_actions[-5:] if 'explore' in a.lower())
            if explore_count > 2:
                return True
        return False
    
    def _cond_scene_mismatch(self, context: Dict[str, Any]) -> bool:
        """
        Condition: scene_classification != visual_description.scene_type
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
        """
        Condition: High visual similarity despite varied actions
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
        """
        Condition: Actions changing every cycle
        """
        recent_actions = context.get('recent_actions', [])
        
        if len(recent_actions) >= 5:
            # Check if last 5 actions are all different
            if len(set(recent_actions[-5:])) == 5:
                return True
        return False
    
    def _cond_unproductive_exploration(self, context: Dict[str, Any]) -> bool:
        """
        Condition: Exploring but coherence not improving
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
        """Set stuck_in_loop fact."""
        engine.set_fact("stuck_in_loop", confidence=0.95, metadata={
            'visual_similarity': context.get('visual_similarity'),
            'explore_count': sum(1 for a in context.get('recent_actions', [])[-5:] if 'explore' in a.lower())
        })
    
    def _cons_recommend_retreat(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend retreat action."""
        engine.recommend("move_backward", Priority.HIGH, "Stuck in loop - retreat", confidence=0.9)
    
    def _cons_recommend_activate(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend activate action."""
        engine.recommend("activate", Priority.MEDIUM, "Try pressing/activating obstacle", confidence=0.8)
    
    def _cons_block_explore(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Block explore action for 3 cycles."""
        engine.block_action("explore", duration=3, reason="Stuck in exploration loop")
    
    def _cons_set_sensory_conflict(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Set sensory_conflict fact."""
        engine.set_fact("sensory_conflict", confidence=0.90, metadata={
            'scene_classification': context.get('scene_classification'),
            'visual_scene_type': context.get('visual_scene_type')
        })
    
    def _cons_recommend_activate_high(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend activate with high priority."""
        engine.recommend("activate", Priority.HIGH, "Resolve scene classification mismatch", confidence=0.85)
    
    def _cons_increase_sensorimotor_authority(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Increase sensorimotor authority weight."""
        current = engine.parameters.get('sensorimotor_authority', 1.0)
        engine.set_parameter('sensorimotor_authority', current * 1.5)
    
    def _cons_set_visual_stasis_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Set visual_stasis fact."""
        engine.set_fact("visual_stasis", confidence=0.85, metadata={
            'visual_similarity': context.get('visual_similarity')
        })
    
    def _cons_recommend_jump(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend jump to break stasis."""
        engine.recommend("jump", Priority.HIGH, "Break visual stasis", confidence=0.8)
    
    def _cons_recommend_turn_around(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend turning around."""
        engine.recommend("turn_around", Priority.MEDIUM, "Change perspective", confidence=0.75)
    
    def _cons_set_indecision_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Set action_thrashing fact."""
        engine.set_fact("action_thrashing", confidence=0.80)
    
    def _cons_recommend_commit_to_action(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend committing to one action."""
        # Get most recent action and stick with it
        recent_actions = context.get('recent_actions', [])
        if recent_actions:
            action = recent_actions[-1]
            engine.recommend(action, Priority.MEDIUM, "Commit to action sequence", confidence=0.7)
    
    def _cons_set_unproductive_fact(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Set unproductive_exploration fact."""
        engine.set_fact("unproductive_exploration", confidence=0.75)
    
    def _cons_recommend_goal_change(self, context: Dict[str, Any], engine: 'RuleEngine'):
        """Recommend changing goals."""
        # This is a meta-recommendation that should trigger goal replanning
        engine.set_fact("needs_goal_revision", confidence=0.8)
    
    # ============================================================================
    # PUBLIC API
    # ============================================================================
    
    def add_rule(self, rule: Rule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def set_fact(self, name: str, confidence: float, metadata: Optional[Dict[str, Any]] = None):
        """Assert a fact in working memory."""
        self.facts[name] = Fact(
            name=name,
            confidence=confidence,
            metadata=metadata or {}
        )
    
    def get_fact(self, name: str) -> Optional[Fact]:
        """Retrieve a fact from working memory."""
        return self.facts.get(name)
    
    def has_fact(self, name: str, min_confidence: float = 0.5) -> bool:
        """Check if fact exists with minimum confidence."""
        fact = self.facts.get(name)
        return fact is not None and fact.confidence >= min_confidence
    
    def recommend(self, action: str, priority: Priority, reason: str, confidence: float):
        """Add an action recommendation."""
        self.recommendations.append(Recommendation(
            action=action,
            priority=priority,
            reason=reason,
            confidence=confidence
        ))
    
    def block_action(self, action: str, duration: int, reason: str):
        """Block an action for N cycles."""
        self.blocked_actions[action] = ActionBlock(
            action=action,
            cycles_remaining=duration,
            reason=reason
        )
    
    def is_action_blocked(self, action: str) -> bool:
        """Check if action is currently blocked."""
        block = self.blocked_actions.get(action)
        return block is not None and block.cycles_remaining > 0
    
    def set_parameter(self, name: str, value: float):
        """Set a system parameter."""
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: float = 1.0) -> float:
        """Get a system parameter."""
        return self.parameters.get(name, default)
    
    def tick_cycle(self):
        """
        Advance the cycle counter - should be called once per game cycle.
        This ensures blocked actions are only decremented once per cycle.
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
        """
        Evaluate all rules against current context.
        
        Args:
            context: Current game state and metrics
            
        Returns:
            Dict with:
                - fired_rules: List of rules that fired
                - recommendations: List of action recommendations
                - blocked_actions: List of currently blocked actions
                - facts: Current facts in working memory
                - parameters: System parameters
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
        """
        Get the highest priority recommendation.
        
        Args:
            exclude_blocked: If True, skip blocked actions
            
        Returns:
            Highest priority recommendation, or None
        """
        recommendations = sorted(self.recommendations, key=lambda r: (r.priority.value, r.confidence), reverse=True)
        
        if exclude_blocked:
            recommendations = [r for r in recommendations if not self.is_action_blocked(r.action)]
        
        return recommendations[0] if recommendations else None
    
    def clear_facts(self, max_age_seconds: Optional[float] = None):
        """
        Clear facts from working memory.
        
        Args:
            max_age_seconds: If provided, only clear facts older than this
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
        """Get a human-readable status report."""
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
