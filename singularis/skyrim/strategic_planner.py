"""
Strategic Planner Neuron

A forward-looking planning system that:
1. Analyzes past experiences from memory
2. Predicts future states and outcomes
3. Plans multi-step action sequences
4. Adapts plans based on terrain and resources

Philosophical grounding:
- ETHICA: Adequate ideas enable prediction and planning
- Planning increases power (potentia agendi)
- Memory enables learning from experience
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class ActionPlan:
    """A planned sequence of actions."""
    goal: str
    steps: List[str]
    expected_outcome: Dict[str, Any]
    confidence: float
    priority: float
    terrain_context: str


@dataclass
class MemoryPattern:
    """A learned pattern from past experiences."""
    context: Dict[str, Any]
    action_sequence: List[str]
    outcome: Dict[str, Any]
    success_rate: float
    frequency: int


class StrategicPlannerNeuron:
    """
    Strategic planning neuron that learns from memory and plans ahead.
    
    Uses episodic memory to:
    - Identify successful action patterns
    - Predict outcomes of action sequences
    - Generate multi-step plans
    - Adapt to terrain and context
    """
    
    def __init__(self, memory_capacity: int = 100, rl_learner=None):
        """
        Initialize strategic planner.

        Args:
            memory_capacity: How many recent experiences to keep
            rl_learner: Optional ReinforcementLearner for Q-value integration
        """
        self.memory_capacity = memory_capacity
        self.rl_learner = rl_learner

        # Recent experiences (episodic memory)
        self.episodic_memory: deque = deque(maxlen=memory_capacity)

        # Learned patterns (semantic memory)
        self.learned_patterns: List[MemoryPattern] = []

        # Current active plan
        self.active_plan: Optional[ActionPlan] = None
        self.plan_step: int = 0

        # Success tracking
        self.plan_successes: int = 0
        self.plan_failures: int = 0

        print("[PLANNER] Strategic Planner Neuron initialized")

    def set_rl_learner(self, rl_learner):
        """
        Set RL learner for Q-value integration.

        Args:
            rl_learner: ReinforcementLearner instance
        """
        self.rl_learner = rl_learner
        print("[PLANNER] Integrated with RL learner")
    
    def record_experience(
        self,
        context: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any],
        success: bool
    ):
        """
        Record an experience in episodic memory.
        
        Args:
            context: Context before action (scene, health, location, etc.)
            action: Action taken
            outcome: Result after action
            success: Whether the action was successful
        """
        experience = {
            'context': context,
            'action': action,
            'outcome': outcome,
            'success': success,
            'timestamp': len(self.episodic_memory)
        }
        
        self.episodic_memory.append(experience)
        
        # Periodically learn patterns from memory
        if len(self.episodic_memory) % 10 == 0:
            self._learn_patterns()
    
    def _learn_patterns(self):
        """
        Analyze episodic memory to extract successful patterns.
        """
        if len(self.episodic_memory) < 5:
            return
        
        # Look for sequences of 2-3 actions that led to success
        for seq_length in [2, 3]:
            for i in range(len(self.episodic_memory) - seq_length):
                sequence = list(self.episodic_memory)[i:i+seq_length]
                
                # Check if sequence ended successfully
                if sequence[-1]['success']:
                    self._record_pattern(sequence)
    
    def _record_pattern(self, sequence: List[Dict[str, Any]]):
        """
        Record a successful action sequence as a pattern.
        
        Args:
            sequence: List of experiences forming a pattern
        """
        # Extract action sequence
        actions = [exp['action'] for exp in sequence]
        context = sequence[0]['context']
        outcome = sequence[-1]['outcome']
        
        # Check if pattern already exists
        for pattern in self.learned_patterns:
            if pattern.action_sequence == actions:
                # Update existing pattern
                pattern.frequency += 1
                pattern.success_rate = (
                    pattern.success_rate * (pattern.frequency - 1) + 1.0
                ) / pattern.frequency
                return
        
        # Create new pattern
        new_pattern = MemoryPattern(
            context=context,
            action_sequence=actions,
            outcome=outcome,
            success_rate=1.0,
            frequency=1
        )
        self.learned_patterns.append(new_pattern)
        
        print(f"[PLANNER] Learned pattern: {' → '.join(actions)}")
    
    def generate_plan(
        self,
        current_state: Dict[str, Any],
        goal: str,
        terrain_type: str
    ) -> Optional[ActionPlan]:
        """
        Generate a multi-step action plan based on memory and RL Q-values.

        Args:
            current_state: Current game state
            goal: High-level goal (e.g., "explore", "survive", "progress")
            terrain_type: Current terrain classification

        Returns:
            ActionPlan if one can be generated, None otherwise
        """
        # Find relevant patterns from memory
        relevant_patterns = self._find_relevant_patterns(
            current_state,
            terrain_type
        )

        # If RL learner is available, use Q-values to enhance pattern selection
        if self.rl_learner is not None and relevant_patterns:
            print("[PLANNER] Using RL Q-values for plan selection...")
            q_values = self.rl_learner.get_q_values(current_state)

            # Score patterns by combining success rate with RL Q-values
            best_pattern = None
            best_score = -float('inf')

            for pattern in relevant_patterns:
                # Pattern score: success rate * frequency
                pattern_score = pattern.success_rate * np.log1p(pattern.frequency)

                # RL score: average Q-value of actions in pattern
                rl_score = 0.0
                for action in pattern.action_sequence:
                    if action in q_values:
                        rl_score += q_values[action]
                rl_score /= len(pattern.action_sequence)

                # Combined score: 60% pattern, 40% RL
                combined_score = 0.6 * pattern_score + 0.4 * rl_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_pattern = pattern

            if best_pattern is None:
                print("[PLANNER] No suitable pattern found")
                return None
        elif relevant_patterns:
            # Select best pattern based on success rate and relevance only
            best_pattern = max(
                relevant_patterns,
                key=lambda p: p.success_rate * p.frequency
            )
        else:
            # No learned patterns, return None to let RL/LLM handle planning
            print("[PLANNER] No learned patterns, deferring to RL/LLM")
            return None

        # Create plan from pattern
        plan = ActionPlan(
            goal=goal,
            steps=best_pattern.action_sequence.copy(),
            expected_outcome=best_pattern.outcome,
            confidence=best_pattern.success_rate,
            priority=self._calculate_priority(goal, current_state),
            terrain_context=terrain_type
        )

        print(f"[PLANNER] Generated plan: {' → '.join(plan.steps)}")
        print(f"[PLANNER] Confidence: {plan.confidence:.2f}, Priority: {plan.priority:.2f}")

        return plan
    
    def _find_relevant_patterns(
        self,
        current_state: Dict[str, Any],
        terrain_type: str
    ) -> List[MemoryPattern]:
        """
        Find patterns relevant to current context.
        
        Args:
            current_state: Current state
            terrain_type: Terrain classification
            
        Returns:
            List of relevant patterns
        """
        relevant = []
        
        for pattern in self.learned_patterns:
            # Check context similarity
            similarity = self._context_similarity(
                pattern.context,
                current_state
            )
            
            if similarity > 0.5:  # Threshold for relevance
                relevant.append(pattern)
        
        return relevant
    
    def _context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two contexts.
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            Similarity score (0-1)
        """
        # Simple similarity based on matching keys
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _default_exploration_plan(self, terrain_type: str) -> ActionPlan:
        """
        Generate default exploration plan when no patterns exist.
        
        Args:
            terrain_type: Terrain classification
            
        Returns:
            Default exploration plan
        """
        if terrain_type == 'indoor_spaces':
            steps = ['interact', 'explore', 'navigate']
        elif terrain_type == 'outdoor_spaces':
            steps = ['explore', 'explore', 'navigate']
        elif terrain_type == 'danger_zones':
            steps = ['combat', 'explore']
        else:
            steps = ['explore', 'navigate']
        
        return ActionPlan(
            goal='explore',
            steps=steps,
            expected_outcome={'progress': True},
            confidence=0.5,
            priority=0.5,
            terrain_context=terrain_type
        )
    
    def _calculate_priority(
        self,
        goal: str,
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate priority of a goal based on current state.
        
        Args:
            goal: Goal string
            current_state: Current state
            
        Returns:
            Priority score (0-1)
        """
        priority = 0.5  # Base priority
        
        # Increase priority based on urgency
        health = current_state.get('health', 100)
        if health < 30:
            priority += 0.3  # Survival is high priority
        
        in_combat = current_state.get('in_combat', False)
        if in_combat:
            priority += 0.2  # Combat is urgent
        
        return min(1.0, priority)
    
    def execute_plan_step(self) -> Optional[str]:
        """
        Get next action from active plan.
        
        Returns:
            Next action string, or None if no active plan
        """
        if not self.active_plan or self.plan_step >= len(self.active_plan.steps):
            return None
        
        action = self.active_plan.steps[self.plan_step]
        self.plan_step += 1
        
        print(f"[PLANNER] Executing step {self.plan_step}/{len(self.active_plan.steps)}: {action}")
        
        return action
    
    def activate_plan(self, plan: ActionPlan):
        """
        Activate a plan for execution.
        
        Args:
            plan: Plan to activate
        """
        self.active_plan = plan
        self.plan_step = 0
        print(f"[PLANNER] Activated plan: {plan.goal}")
    
    def complete_plan(self, success: bool):
        """
        Mark current plan as complete.
        
        Args:
            success: Whether plan succeeded
        """
        if success:
            self.plan_successes += 1
            print(f"[PLANNER] Plan succeeded! ({self.plan_successes} total)")
        else:
            self.plan_failures += 1
            print(f"[PLANNER] Plan failed. ({self.plan_failures} total)")
        
        self.active_plan = None
        self.plan_step = 0
    
    def should_replan(self, current_state: Dict[str, Any]) -> bool:
        """
        Determine if we should abandon current plan and replan.
        
        Args:
            current_state: Current state
            
        Returns:
            True if should replan
        """
        if not self.active_plan:
            return True
        
        # Replan if context changed significantly
        if current_state.get('in_combat', False) and \
           self.active_plan.terrain_context != 'danger_zones':
            print("[PLANNER] Context changed - replanning")
            return True
        
        # Replan if plan is complete
        if self.plan_step >= len(self.active_plan.steps):
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get planner statistics.
        
        Returns:
            Dict of statistics
        """
        total_plans = self.plan_successes + self.plan_failures
        success_rate = (
            self.plan_successes / total_plans if total_plans > 0 else 0.0
        )
        
        return {
            'patterns_learned': len(self.learned_patterns),
            'experiences_recorded': len(self.episodic_memory),
            'plans_executed': total_plans,
            'success_rate': success_rate,
            'active_plan': self.active_plan.goal if self.active_plan else None
        }
