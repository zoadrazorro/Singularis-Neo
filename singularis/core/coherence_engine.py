"""
CoherenceEngine - The One Function

Computes global coherence C from all subsystems.
This is the 'one thing' all learning and decision-making optimize.

The metaphysical "how well am I being?" made executable.
"""

from typing import Dict
import math
from .being_state import BeingState, LuminaState


class CoherenceEngine:
    """
    Computes global coherence C_global from BeingState.
    
    This is the metaphysical glue-function:
    - Integrates all subsystem coherences
    - Balances the Three Lumina
    - Produces one scalar everyone optimizes
    
    This is Spinoza's conatus (striving to persist in being),
    IIT's Φ (integrated information),
    and Lumen philosophy (balance of Being)
    compiled into one executable function.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initializes the CoherenceEngine.

        Args:
            verbose (bool, optional): If True, prints detailed coherence calculations
                                    during computation. Defaults to True.
        """
        self.verbose = verbose
        
        # Component weights - how much each aspect contributes to global coherence
        # These can be learned/tuned over time
        self.component_weights = {
            'lumina': 0.25,           # Balance of Three Lumina
            'consciousness': 0.20,    # Consciousness metrics (C, Phi, unity)
            'cognitive': 0.15,        # Mind system coherence
            'temporal': 0.10,         # Temporal binding coherence
            'rl': 0.10,               # RL performance
            'meta_rl': 0.08,          # Meta-learning quality
            'emotion': 0.07,          # Emotional coherence
            'voice': 0.05             # Voice-state alignment
        }
        
        # Lumina weights - how to balance the three
        self.lumina_weights = {
            'ontic': 0.33,
            'structural': 0.33,
            'participatory': 0.34
        }
        
        # Coherence history
        self.coherence_history = []
        self.max_history = 1000
        
        if verbose:
            print("[COHERENCE] CoherenceEngine initialized")
            print(f"[COHERENCE] Component weights: {self.component_weights}")
    
    def _lumina_coherence(self, lumina: LuminaState) -> float:
        """
        Computes the coherence score from the Three Lumina.

        This score is a combination of the geometric mean of the Lumina values
        (ensuring all are present) and a balance score that rewards equilibrium
        between them.

        Args:
            lumina (LuminaState): The current state of the Three Lumina.

        Returns:
            float: The coherence score for the Lumina, in the range [0, 1].
        """
        # Geometric mean - all three must be present
        vals = [
            max(1e-6, lumina.ontic),
            max(1e-6, lumina.structural),
            max(1e-6, lumina.participatory)
        ]
        geometric = (vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)
        
        # Balance bonus - reward balanced Lumina
        balance = lumina.balance_score()
        
        # Combine: 70% geometric mean, 30% balance
        return 0.7 * geometric + 0.3 * balance
    
    def _consciousness_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from consciousness-related metrics.

        This score is an average of the `coherence_C`, `unity_index`, and `phi_hat`
        values from the BeingState.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for consciousness, in the range [0, 1].
        """
        # Average of the three consciousness metrics
        return (state.coherence_C + state.unity_index + state.phi_hat) / 3.0
    
    def _cognitive_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from the cognitive system.

        The score is based on the cognitive coherence from the BeingState,
        penalized by the number of cognitive dissonances and bonused by
        the number of active heuristics.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for the cognitive system, in the range [0, 1].
        """
        base_coherence = state.cognitive_coherence
        
        # Penalty for dissonances
        dissonance_penalty = min(0.5, len(state.cognitive_dissonances) * 0.05)
        
        # Bonus for active heuristics (up to 0.1)
        heuristic_bonus = min(0.1, len(state.active_heuristics) * 0.02)
        
        return max(0.0, min(1.0, base_coherence - dissonance_penalty + heuristic_bonus))
    
    def _temporal_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from temporal binding.

        This score is based on the temporal coherence from the BeingState,
        penalized by the number of unclosed bindings and stuck loops.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for temporal binding, in the range [0, 1].
        """
        base_temporal = state.temporal_coherence
        
        # Penalty for unclosed bindings
        unclosed_penalty = min(0.3, state.unclosed_bindings * 0.03)
        
        # Strong penalty for stuck loops
        stuck_penalty = min(0.5, state.stuck_loop_count * 0.1)
        
        return max(0.0, min(1.0, base_temporal - unclosed_penalty - stuck_penalty))
    
    def _rl_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from reinforcement learning performance.

        The score is based on the average reward and the exploration rate,
        rewarding a balance between high rewards and a healthy level of exploration.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for reinforcement learning, in the range [0, 1].
        """
        # Normalize reward to [0, 1]
        # Assumes rewards are roughly in [-1, 1] range
        normalized_reward = (state.avg_reward + 1.0) / 2.0
        
        # Balance exploration - too much or too little is bad
        exploration_balance = 1.0 - abs(state.exploration_rate - 0.2)  # Ideal ~0.2
        
        # Combine: 80% reward, 20% exploration balance
        return 0.8 * normalized_reward + 0.2 * exploration_balance
    
    def _meta_rl_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from meta-reinforcement learning.

        This score is based on the meta-score from the BeingState, with a bonus
        for the number of meta-analyses performed.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for meta-reinforcement learning, in the range [0, 1].
        """
        # Meta score (if available)
        meta_score = state.meta_score
        
        # Bonus for having done meta-analyses
        analysis_bonus = min(0.2, state.total_meta_analyses * 0.01)
        
        return min(1.0, meta_score + analysis_bonus)
    
    def _emotion_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from the emotion system.

        This score considers the coherence of the emotion with the current situation
        and the balance of the emotion's intensity.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for the emotion system, in the range [0, 1].
        """
        # Get emotion coherence from state
        emotion_coh = state.emotion_state.get('coherence', 0.5)
        
        # Intensity should be moderate (not too high, not too low)
        intensity_balance = 1.0 - abs(state.emotion_intensity - 0.5)
        
        # Combine: 70% coherence, 30% intensity balance
        return 0.7 * emotion_coh + 0.3 * intensity_balance
    
    def _voice_coherence(self, state: BeingState) -> float:
        """
        Computes the coherence score from the voice system.

        This score measures the alignment between the being's inner state and its
        outer vocal expression.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The coherence score for the voice system, in the range [0, 1].
        """
        # Voice alignment (how well voice matches inner state)
        return state.voice_alignment
    
    def compute(self, state: BeingState) -> float:
        """
        Computes the single global coherence score (C_global) from the BeingState.

        This score represents the overall coherence of the being at a given moment.
        It is a weighted sum of the coherence scores from all major subsystems.
        The result is a single value that all learning and decision-making
        processes aim to optimize.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            float: The global coherence score, in the range [0, 1].
        """
        # Compute component coherences
        lumina_C = self._lumina_coherence(state.lumina)
        consciousness_C = self._consciousness_coherence(state)
        cognitive_C = self._cognitive_coherence(state)
        temporal_C = self._temporal_coherence(state)
        rl_C = self._rl_coherence(state)
        meta_rl_C = self._meta_rl_coherence(state)
        emotion_C = self._emotion_coherence(state)
        voice_C = self._voice_coherence(state)
        
        # Weighted sum
        C_global = (
            self.component_weights['lumina'] * lumina_C +
            self.component_weights['consciousness'] * consciousness_C +
            self.component_weights['cognitive'] * cognitive_C +
            self.component_weights['temporal'] * temporal_C +
            self.component_weights['rl'] * rl_C +
            self.component_weights['meta_rl'] * meta_rl_C +
            self.component_weights['emotion'] * emotion_C +
            self.component_weights['voice'] * voice_C
        )
        
        # Clamp to [0, 1]
        C_global = max(0.0, min(1.0, C_global))
        
        # Store in history
        self.coherence_history.append((state.timestamp, C_global))
        if len(self.coherence_history) > self.max_history:
            self.coherence_history.pop(0)
        
        # Verbose output
        if self.verbose and state.cycle_number % 10 == 0:
            print(f"\n[COHERENCE] Cycle {state.cycle_number}: C_global = {C_global:.3f}")
            print(f"  Lumina:        {lumina_C:.3f} (l_o={state.lumina.ontic:.3f}, l_s={state.lumina.structural:.3f}, l_p={state.lumina.participatory:.3f})")
            print(f"  Consciousness: {consciousness_C:.3f} (C={state.coherence_C:.3f}, Phi={state.phi_hat:.3f}, unity={state.unity_index:.3f})")
            print(f"  Cognitive:     {cognitive_C:.3f}")
            print(f"  Temporal:      {temporal_C:.3f}")
            print(f"  RL:            {rl_C:.3f}")
            print(f"  Meta-RL:       {meta_rl_C:.3f}")
            print(f"  Emotion:       {emotion_C:.3f}")
            print(f"  Voice:         {voice_C:.3f}")
        
        return C_global
    
    def get_component_breakdown(self, state: BeingState) -> Dict[str, float]:
        """
        Gets a breakdown of the coherence scores by individual component.

        This is useful for debugging and understanding which subsystems are
        contributing to or detracting from the global coherence.

        Args:
            state (BeingState): The current state of the being.

        Returns:
            Dict[str, float]: A dictionary mapping component names to their
                              coherence scores.
        """
        return {
            'lumina': self._lumina_coherence(state.lumina),
            'consciousness': self._consciousness_coherence(state),
            'cognitive': self._cognitive_coherence(state),
            'temporal': self._temporal_coherence(state),
            'rl': self._rl_coherence(state),
            'meta_rl': self._meta_rl_coherence(state),
            'emotion': self._emotion_coherence(state),
            'voice': self._voice_coherence(state)
        }
    
    def get_trend(self, window: int = 10) -> str:
        """
        Determines the trend of the coherence score over a recent window.

        Args:
            window (int, optional): The number of recent coherence samples to analyze.
                                  Defaults to 10.

        Returns:
            str: "increasing", "decreasing", "stable", or "insufficient_data".
        """
        if len(self.coherence_history) < window:
            return "insufficient_data"
        
        recent = [c for _, c in self.coherence_history[-window:]]
        
        # Simple linear trend
        first_half = sum(recent[:window//2]) / (window//2)
        second_half = sum(recent[window//2:]) / (window - window//2)
        
        diff = second_half - first_half
        
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the coherence scores.

        Returns:
            Dict[str, Any]: A dictionary containing the number of samples,
                            current coherence, average, min, max, and trend.
        """
        if not self.coherence_history:
            return {
                'samples': 0,
                'current': 0.0,
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'trend': 'no_data'
            }
        
        coherences = [c for _, c in self.coherence_history]
        
        return {
            'samples': len(coherences),
            'current': coherences[-1],
            'avg': sum(coherences) / len(coherences),
            'min': min(coherences),
            'max': max(coherences),
            'trend': self.get_trend()
        }
    
    def print_stats(self):
        """Prints a formatted summary of coherence statistics to the console."""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("COHERENCE ENGINE STATISTICS".center(80))
        print("="*80)
        print(f"Samples: {stats['samples']}")
        print(f"Current C_global: {stats['current']:.3f}")
        print(f"Average: {stats['avg']:.3f}")
        print(f"Min: {stats['min']:.3f}")
        print(f"Max: {stats['max']:.3f}")
        print(f"Trend: {stats['trend']}")
        print("="*80 + "\n")
