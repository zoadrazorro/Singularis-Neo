"""
Intrinsic Motivation System

Provides internal drives that power autonomous behavior.

Three core drives (inspired by Self-Determination Theory + Spinoza):
1. Curiosity: Reduce uncertainty, explore novel states
2. Competence: Increase capability, master skills
3. Coherence: Increase 𝒞 (coherentia - alignment with Being)

Key insight: Intelligence requires intrinsic motivation, not just
rewards from external tasks.

Philosophical grounding:
- ETHICA Part III, Prop VI: Conatus (striving) is essence of being
- Conatus = ∇𝒞 (gradient of coherence)
- Drive to understand = drive to increase adequacy = drive to be free
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class MotivationType(Enum):
    """Types of intrinsic motivations."""
    CURIOSITY = "curiosity"  # Information gain, novelty seeking
    COMPETENCE = "competence"  # Skill mastery, capability growth
    COHERENCE = "coherence"  # Increase 𝒞 (coherentia)
    AUTONOMY = "autonomy"  # Self-direction, freedom from external control


@dataclass
class MotivationState:
    """Current state of motivational drives."""
    curiosity: float = 0.5  # [0, 1]
    competence: float = 0.5
    coherence: float = 0.5
    autonomy: float = 0.5

    def total(self) -> float:
        """Total motivation level."""
        return (self.curiosity + self.competence + self.coherence + self.autonomy) / 4.0

    def dominant_drive(self) -> MotivationType:
        """Which drive is strongest?"""
        drives = {
            MotivationType.CURIOSITY: self.curiosity,
            MotivationType.COMPETENCE: self.competence,
            MotivationType.COHERENCE: self.coherence,
            MotivationType.AUTONOMY: self.autonomy,
        }
        return max(drives, key=drives.get)


class CuriosityDrive:
    """
    Curiosity: Drive to reduce uncertainty and explore.

    Mechanisms:
    - Information gain: Prefer actions that reduce uncertainty
    - Novelty seeking: Explore unfamiliar states
    - Prediction error: Seek surprises to learn from
    """

    def __init__(self, baseline: float = 0.5, decay_rate: float = 0.95):
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.current_level = baseline

        # Track what's been explored
        self.explored_states: set = set()
        self.uncertainty_estimates: Dict[str, float] = {}

    def compute_curiosity(
        self,
        state: Dict[str, Any],
        uncertainty: Optional[float] = None
    ) -> float:
        """
        Compute curiosity level for a state.

        High curiosity when:
        - State is novel (never seen)
        - Uncertainty is high
        - Prediction errors were large

        Args:
            state: Current state
            uncertainty: Epistemic uncertainty

        Returns:
            Curiosity level in [0, 1]
        """
        # Hash state for novelty check
        state_hash = self._hash_state(state)

        # Novelty component
        if state_hash in self.explored_states:
            novelty = 0.1  # Low curiosity for familiar
        else:
            novelty = 1.0  # High curiosity for novel

        # Uncertainty component
        if uncertainty is None:
            uncertainty = self.uncertainty_estimates.get(state_hash, 0.5)

        # Combine (average)
        curiosity = (novelty + uncertainty) / 2.0

        # Update current level
        self.current_level = curiosity

        return curiosity

    def mark_explored(self, state: Dict[str, Any]):
        """Mark state as explored (reduces future curiosity)."""
        state_hash = self._hash_state(state)
        self.explored_states.add(state_hash)

    def update_uncertainty(self, state: Dict[str, Any], uncertainty: float):
        """Update uncertainty estimate for state."""
        state_hash = self._hash_state(state)
        self.uncertainty_estimates[state_hash] = uncertainty

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Hash state for lookup."""
        # Simple hash (could be more sophisticated)
        return str(sorted(state.items()))

    def reset(self):
        """Reset curiosity drive."""
        self.current_level = self.baseline
        self.explored_states.clear()


class CompetenceDrive:
    """
    Competence: Drive to increase capabilities and master skills.

    Mechanisms:
    - Flow state: Prefer tasks at edge of ability
    - Progress tracking: Satisfaction from improvement
    - Mastery gradient: Focus on achievable challenges
    """

    def __init__(self, baseline: float = 0.5):
        self.baseline = baseline
        self.current_level = baseline

        # Track competence in different domains
        self.competence_levels: Dict[str, float] = {}

        # Track progress over time
        self.progress_history: List[Tuple[str, float]] = []

    def compute_competence_motivation(
        self,
        task: str,
        difficulty: float,
        current_ability: float
    ) -> float:
        """
        Compute motivation to attempt task.

        Highest motivation when task difficulty matches ability
        (flow state: not too easy, not too hard).

        Args:
            task: Task identifier
            difficulty: Task difficulty in [0, 1]
            current_ability: Current ability level in [0, 1]

        Returns:
            Competence motivation in [0, 1]
        """
        # Flow state: maximum motivation when difficulty ≈ ability
        # Use Gaussian centered at ability
        optimal_difficulty = current_ability + 0.1  # Slightly above ability

        # Compute distance from optimal
        distance = abs(difficulty - optimal_difficulty)

        # Gaussian falloff
        motivation = math.exp(-5 * distance**2)

        self.current_level = motivation
        return motivation

    def record_success(self, task: str, performance: float):
        """
        Record successful task completion.

        Updates competence level and generates satisfaction.

        Args:
            task: Task completed
            performance: How well (0-1)
        """
        # Update competence level for this task
        if task in self.competence_levels:
            # Exponential moving average
            self.competence_levels[task] = 0.9 * self.competence_levels[task] + 0.1 * performance
        else:
            self.competence_levels[task] = performance

        # Record progress
        self.progress_history.append((task, performance))

        # Trim history
        if len(self.progress_history) > 100:
            self.progress_history = self.progress_history[-100:]

    def get_competence(self, task: str) -> float:
        """Get current competence level for task."""
        return self.competence_levels.get(task, 0.0)

    def get_progress_rate(self, task: str, window: int = 10) -> float:
        """
        Compute rate of progress in task.

        Returns:
            Progress rate (positive = improving)
        """
        # Filter history for this task
        task_history = [(t, p) for t, p in self.progress_history if t == task]

        if len(task_history) < 2:
            return 0.0

        # Take recent window
        recent = task_history[-window:]

        # Compute linear trend
        times = list(range(len(recent)))
        perfs = [p for _, p in recent]

        # Simple linear regression slope
        mean_time = np.mean(times)
        mean_perf = np.mean(perfs)

        slope = np.sum((times - mean_time) * (perfs - mean_perf)) / (np.sum((times - mean_time)**2) + 1e-8)

        return slope


class CoherenceDrive:
    """
    Coherence: Drive to increase 𝒞 (coherentia).

    This is the CORE drive in Singularis philosophy:
    - Conatus = ∇𝒞 (striving = coherence gradient)
    - Ethics = Δ𝒞 > 0
    - Freedom ∝ Coherence

    Mechanisms:
    - Prefer actions that increase 𝒞
    - Avoid actions that decrease 𝒞
    - Long-term coherence optimization
    """

    def __init__(self, baseline: float = 0.5, discount_factor: float = 0.95):
        self.baseline = baseline
        self.current_level = baseline
        self.discount_factor = discount_factor  # γ for temporal discounting

        # Track coherence over time
        self.coherence_history: List[float] = []

    def compute_coherence_motivation(
        self,
        current_coherence: float,
        predicted_delta_coherence: float,
        time_horizon: int = 10
    ) -> float:
        """
        Compute motivation based on coherence dynamics.

        High motivation when:
        - Predicted Δ𝒞 > 0 (action increases coherence)
        - Current 𝒞 is low (need to increase)
        - Long-term coherence trajectory is positive

        Args:
            current_coherence: Current 𝒞 in [0, 1]
            predicted_delta_coherence: Expected Δ𝒞 from action
            time_horizon: Steps into future to consider

        Returns:
            Coherence motivation in [0, 1]
        """
        # Component 1: Immediate Δ𝒞
        immediate = predicted_delta_coherence

        # Component 2: Need (motivation higher when 𝒞 is low)
        need = 1.0 - current_coherence

        # Component 3: Long-term trajectory
        if len(self.coherence_history) >= 3:
            recent = self.coherence_history[-3:]
            trajectory = recent[-1] - recent[0]  # Positive if improving
        else:
            trajectory = 0.0

        # Combine (weighted sum)
        motivation = (
            0.5 * immediate +  # Immediate gain
            0.3 * need +  # Need to increase
            0.2 * trajectory  # Momentum
        )

        # Map to [0, 1]
        motivation = (motivation + 1.0) / 2.0
        motivation = np.clip(motivation, 0.0, 1.0)

        self.current_level = motivation
        return motivation

    def record_coherence(self, coherence: float):
        """Record coherence value."""
        self.coherence_history.append(coherence)

        # Trim history
        if len(self.coherence_history) > 100:
            self.coherence_history = self.coherence_history[-100:]

    def get_coherence_trend(self, window: int = 10) -> float:
        """
        Get coherence trend (improving or declining?).

        Returns:
            Trend: positive = improving, negative = declining
        """
        if len(self.coherence_history) < 2:
            return 0.0

        recent = self.coherence_history[-window:]
        if len(recent) < 2:
            return 0.0

        # Linear trend
        times = np.arange(len(recent))
        coherences = np.array(recent)

        # Regression slope
        mean_t = np.mean(times)
        mean_c = np.mean(coherences)

        slope = np.sum((times - mean_t) * (coherences - mean_c)) / (np.sum((times - mean_t)**2) + 1e-8)

        return slope


class IntrinsicMotivation:
    """
    Unified intrinsic motivation system.

    Combines all drives:
    - Curiosity (explore)
    - Competence (master)
    - Coherence (align with Being)
    - Autonomy (self-direct)
    """

    def __init__(
        self,
        curiosity_weight: float = 0.3,
        competence_weight: float = 0.2,
        coherence_weight: float = 0.4,  # Highest weight (core drive)
        autonomy_weight: float = 0.1
    ):
        """
        Initialize motivation system.

        Args:
            *_weight: Relative importance of each drive (should sum to 1.0)
        """
        # Normalize weights
        total = curiosity_weight + competence_weight + coherence_weight + autonomy_weight
        self.curiosity_weight = curiosity_weight / total
        self.competence_weight = competence_weight / total
        self.coherence_weight = coherence_weight / total
        self.autonomy_weight = autonomy_weight / total

        # Individual drives
        self.curiosity = CuriosityDrive()
        self.competence = CompetenceDrive()
        self.coherence = CoherenceDrive()

        # Current state
        self.state = MotivationState()

    def compute_motivation(
        self,
        state: Dict[str, Any],
        action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MotivationState:
        """
        Compute current motivation state.

        Args:
            state: Current world state
            action: Potential action to evaluate
            context: Additional context

        Returns:
            MotivationState with all drive levels
        """
        context = context or {}

        # Curiosity
        uncertainty = context.get('uncertainty', None)
        curiosity_level = self.curiosity.compute_curiosity(state, uncertainty)

        # Competence
        if action:
            difficulty = context.get('difficulty', 0.5)
            ability = self.competence.get_competence(action)
            competence_level = self.competence.compute_competence_motivation(
                action, difficulty, ability
            )
        else:
            competence_level = self.competence.current_level

        # Coherence
        current_coh = state.get('coherence', 0.5)
        predicted_delta = context.get('predicted_delta_coherence', 0.0)
        coherence_level = self.coherence.compute_coherence_motivation(
            current_coh, predicted_delta
        )

        # Autonomy (simplified: high when not externally controlled)
        external_control = context.get('external_control', 0.0)
        autonomy_level = 1.0 - external_control

        # Update state
        self.state = MotivationState(
            curiosity=curiosity_level,
            competence=competence_level,
            coherence=coherence_level,
            autonomy=autonomy_level
        )

        return self.state

    def select_action(
        self,
        available_actions: List[str],
        state: Dict[str, Any],
        action_contexts: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, float]:
        """
        Select action based on intrinsic motivation.

        Args:
            available_actions: List of possible actions
            state: Current state
            action_contexts: Context for each action

        Returns:
            (selected_action, motivation_score)
        """
        best_action = None
        best_score = -float('inf')

        for action in available_actions:
            context = action_contexts.get(action, {})

            # Compute motivation for this action
            mot_state = self.compute_motivation(state, action, context)

            # Weighted combination
            score = (
                self.curiosity_weight * mot_state.curiosity +
                self.competence_weight * mot_state.competence +
                self.coherence_weight * mot_state.coherence +
                self.autonomy_weight * mot_state.autonomy
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score

    def get_dominant_drive(self) -> MotivationType:
        """Get currently dominant drive."""
        return self.state.dominant_drive()

    def get_state(self) -> MotivationState:
        """Get current motivation state."""
        return self.state


# Example usage
if __name__ == "__main__":
    print("Testing Intrinsic Motivation System...")

    # Create motivation system
    motivation = IntrinsicMotivation(
        curiosity_weight=0.3,
        competence_weight=0.2,
        coherence_weight=0.4,
        autonomy_weight=0.1
    )

    # 1. Compute motivation for exploration
    print("\n1. Exploration scenario:")
    state = {'location': 'room_A', 'coherence': 0.6}
    context = {'uncertainty': 0.8, 'predicted_delta_coherence': 0.1}

    mot_state = motivation.compute_motivation(state, action='explore', context=context)
    print(f"   Curiosity: {mot_state.curiosity:.2f}")
    print(f"   Competence: {mot_state.competence:.2f}")
    print(f"   Coherence: {mot_state.coherence:.2f}")
    print(f"   Autonomy: {mot_state.autonomy:.2f}")
    print(f"   Dominant drive: {mot_state.dominant_drive().value}")

    # 2. Action selection
    print("\n2. Action selection:")
    actions = ['explore', 'practice', 'rest', 'reflect']
    action_contexts = {
        'explore': {'uncertainty': 0.9, 'difficulty': 0.7, 'predicted_delta_coherence': 0.05},
        'practice': {'uncertainty': 0.2, 'difficulty': 0.5, 'predicted_delta_coherence': 0.02},
        'rest': {'uncertainty': 0.1, 'difficulty': 0.1, 'predicted_delta_coherence': -0.01},
        'reflect': {'uncertainty': 0.3, 'difficulty': 0.6, 'predicted_delta_coherence': 0.15},
    }

    selected, score = motivation.select_action(actions, state, action_contexts)
    print(f"   Selected action: {selected} (score: {score:.3f})")

    # 3. Record progress
    print("\n3. Recording competence progress...")
    for i in range(5):
        motivation.competence.record_success('explore', 0.5 + i * 0.1)

    progress = motivation.competence.get_progress_rate('explore')
    print(f"   Progress rate: {progress:.3f}")

    # 4. Coherence tracking
    print("\n4. Tracking coherence...")
    for c in [0.5, 0.55, 0.6, 0.65, 0.7]:
        motivation.coherence.record_coherence(c)

    trend = motivation.coherence.get_coherence_trend()
    print(f"   Coherence trend: {trend:.3f} {'(improving)' if trend > 0 else '(declining)'}")

    print("\n✓ Intrinsic Motivation tests complete")
