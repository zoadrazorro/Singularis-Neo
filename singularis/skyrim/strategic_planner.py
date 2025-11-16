"""
Strategic Planner Neuron

A forward-looking planning system that:
1. Analyzes past experiences from memory
2. Predicts future states and outcomes
3. Plans multi-step action sequences
4. Adapts plans based on terrain and resources

Design principles:
- Learning from past gameplay experiences
- Multi-step planning for complex objectives
- Context-aware adaptation to different situations
- Memory-based prediction of action outcomes
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class ActionPlan:
    """Represents a structured, multi-step plan to achieve a specific goal.

    Attributes:
        goal: A string describing the high-level objective of the plan.
        steps: A list of action strings to be executed in sequence.
        expected_outcome: A dictionary describing the anticipated state after the
                          plan is successfully executed.
        confidence: A float (0.0-1.0) indicating the planner's confidence in
                    the plan's success.
        priority: A float (0.0-1.0) representing the urgency or importance of the plan.
        terrain_context: A string classifying the terrain for which this plan is intended.
    """
    goal: str
    steps: List[str]
    expected_outcome: Dict[str, Any]
    confidence: float
    priority: float
    terrain_context: str


@dataclass
class MemoryPattern:
    """Represents a learned pattern of actions and outcomes from past experiences.

    These patterns form the basis of the strategic planner's knowledge, allowing it
    to construct new plans based on previously successful sequences.

    Attributes:
        context: The initial state or context in which the action sequence was performed.
        action_sequence: The list of actions that were taken.
        outcome: The resulting state or outcome of the action sequence.
        success_rate: The historical success rate of this pattern.
        frequency: The number of times this pattern has been observed.
    """
    context: Dict[str, Any]
    action_sequence: List[str]
    outcome: Dict[str, Any]
    success_rate: float
    frequency: int


class StrategicPlannerNeuron:
    """A strategic planning component that learns from memory to generate multi-step plans.

    This "neuron" analyzes its episodic memory of past actions and their outcomes
    to identify successful patterns. It uses these learned patterns to construct
    new `ActionPlan` objects tailored to the current state and high-level goals.
    It can integrate with a reinforcement learner to further refine its plan
    selection with Q-values.
    """
    
    def __init__(self, memory_capacity: int = 100, rl_learner=None):
        """Initializes the StrategicPlannerNeuron.

        Args:
            memory_capacity: The maximum number of recent experiences to store in
                             episodic memory.
            rl_learner: An optional instance of a reinforcement learner for integrating
                        Q-values into planning.
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
        
        # Cloud LLM integration
        self.hybrid_llm = None
        self.moe = None
        self.parallel_agi = None  # Reference to main AGI for parallel queries

        print("[PLANNER] Strategic Planner Neuron initialized")

    def set_rl_learner(self, rl_learner):
        """Sets the reinforcement learning (RL) learner for Q-value integration.

        Args:
            rl_learner: An instance of the `ReinforcementLearner`.
        """
        self.rl_learner = rl_learner
        print("[PLANNER] Integrated with RL learner")
    
    def set_hybrid_llm(self, hybrid_llm):
        """Sets the hybrid LLM for cloud-based planning.

        Args:
            hybrid_llm: An instance of the hybrid LLM client.
        """
        self.hybrid_llm = hybrid_llm
        print("[PLANNER] ✓ Hybrid LLM connected")
    
    def set_moe(self, moe):
        """Sets the Mixture of Experts (MoE) for expert consensus planning.

        Args:
            moe: An instance of the MoE orchestrator.
        """
        self.moe = moe
        print("[PLANNER] ✓ MoE connected")
    
    def set_parallel_agi(self, agi):
        """Sets the reference to the main AGI for parallel queries.

        Args:
            agi: The main AGI instance.
        """
        self.parallel_agi = agi
        print("[PLANNER] ✓ Parallel AGI reference connected")
    
    def record_experience(
        self,
        context: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any],
        success: bool
    ):
        """Records a single experience tuple in the planner's episodic memory.

        After recording, it periodically triggers the pattern learning process.

        Args:
            context: The state/context before the action was taken.
            action: The action that was performed.
            outcome: The resulting outcome or state change.
            success: A boolean indicating whether the action was successful.
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
        """Analyzes episodic memory to extract and record successful action sequences."""
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
        """Creates or updates a MemoryPattern from a successful sequence of experiences."""
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
        """Generates a multi-step action plan to achieve a goal.

        This method finds relevant learned patterns from memory, scores them based
        on historical success and (if available) RL Q-values, and constructs an
        `ActionPlan` from the best-scoring pattern.

        Args:
            current_state: The current game state.
            goal: A string describing the high-level goal (e.g., "explore").
            terrain_type: A string classifying the current terrain.

        Returns:
            An `ActionPlan` object if a suitable plan can be generated, otherwise None.
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
        """Finds learned patterns that are relevant to the current game context."""
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
        """Calculates a similarity score between two context dictionaries."""
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
        """Generates a default exploration plan for when no learned patterns are available."""
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
        """Calculates the priority of a goal based on the current state's urgency."""
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
        """Executes the next step in the currently active plan.

        Returns:
            The next action string from the plan, or None if the plan is complete
            or no plan is active.
        """
        if not self.active_plan or self.plan_step >= len(self.active_plan.steps):
            return None
        
        action = self.active_plan.steps[self.plan_step]
        self.plan_step += 1
        
        print(f"[PLANNER] Executing step {self.plan_step}/{len(self.active_plan.steps)}: {action}")
        
        return action
    
    def activate_plan(self, plan: ActionPlan):
        """Sets a new plan as the active plan and resets the step counter.

        Args:
            plan: The `ActionPlan` to activate.
        """
        self.active_plan = plan
        self.plan_step = 0
        print(f"[PLANNER] Activated plan: {plan.goal}")
    
    def complete_plan(self, success: bool):
        """Marks the current plan as complete and updates success/failure statistics.

        Args:
            success: A boolean indicating whether the plan's execution was successful.
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
        """Determines if the current plan should be abandoned in favor of replanning.

        Replanning is recommended if the context has changed significantly (e.g.,
        entering combat) or if the current plan is finished.

        Args:
            current_state: The current game state.

        Returns:
            True if the planner should generate a new plan, False otherwise.
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
        """Retrieves statistics about the planner's performance.

        Returns:
            A dictionary of statistics, including the number of learned patterns,
            executed plans, and the overall success rate.
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
