"""
Emotional Valence System for Skyrim AGI

Implements Spinozist affect theory from ETHICA UNIVERSALIS Part IV:
"An Affect is a modification of body and mind that increases or decreases
our power of acting, reflecting conatus encountering facilitation or obstruction."

This system computes emotional valence from:
1. Game events (combat, quests, social interactions)
2. State changes (health, progression, resources)
3. Coherence dynamics (Δ𝒞)
4. Adequacy of understanding (Adeq)

Key Concepts:
- Valence (Val): Emotional charge ∈ ℝ (unbounded affect index)
- Active Affect: Caused by understanding (Adeq ≥ θ, Δ𝒞 ≥ 0)
- Passive Affect: Caused by external forces (Adeq < θ)
- Affect Types: joy, sadness, fear, hope, love, hatred, etc.

From MATHEMATICA SINGULARIS D6:
- PASSIVE affect: Δ Valence caused by external necessity with Adeq(a) < θ
- ACTIVE affect: Δ Valence with Adeq(a) ≥ θ and Δ𝒞 ≥ 0
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np
from enum import Enum

from ..core.types import Affect


class AffectType(Enum):
    """
    Primary affect types from ETHICA Part IV.

    Basic affects:
    - JOY: Increase in power to act
    - SADNESS: Decrease in power to act
    - DESIRE: Drive toward something

    Derived affects:
    - HOPE: Joy mixed with doubt
    - FEAR: Sadness mixed with doubt
    - LOVE: Joy with external cause
    - HATRED: Sadness with external cause
    - COURAGE: Joy overcoming danger
    - DESPAIR: Overwhelming sadness
    """
    JOY = "joy"
    SADNESS = "sadness"
    DESIRE = "desire"
    HOPE = "hope"
    FEAR = "fear"
    LOVE = "love"
    HATRED = "hatred"
    COURAGE = "courage"
    DESPAIR = "despair"
    NEUTRAL = "neutral"


@dataclass
class ValenceState:
    """
    Complete emotional valence state.

    Tracks current affective state with all Spinozist components.
    """
    valence: float  # Current emotional charge (unbounded)
    valence_delta: float  # Recent change in valence
    affect_type: AffectType  # Dominant affect
    is_active: bool  # Active (understanding) vs Passive (external)
    adequacy: float  # Current adequacy of ideas (0-1)
    coherence_delta: float  # Associated Δ𝒞

    # Temporal tracking
    valence_history: List[float]  # Recent valence trajectory
    affect_stability: float  # How stable is current affect (0-1)

    # Component affects (for complex emotions)
    component_affects: Dict[str, float]  # Multiple simultaneous affects

    def __post_init__(self):
        """Ensure valence history is initialized."""
        if not self.valence_history:
            self.valence_history = [self.valence]

    def get_dominant_affect(self) -> AffectType:
        """Get the dominant affect type."""
        return self.affect_type

    def is_ethical_affect(self) -> bool:
        """
        Check if affect is ethically positive.

        From ETHICA: Active affects with Δ𝒞 > 0 are ethical.
        """
        return self.is_active and self.coherence_delta > 0

    def get_power_to_act(self) -> float:
        """
        Estimate current power to act from valence.

        From ETHICA: Joy increases power, sadness decreases it.
        Normalized to [0, 1] range.
        """
        # Transform unbounded valence to bounded power estimate
        # Use sigmoid to map ℝ → (0, 1)
        return 1.0 / (1.0 + np.exp(-self.valence))


class EmotionalValenceComputer:
    """
    Computes emotional valence from game events and state changes.

    Uses Spinozist affect theory to generate genuine emotional responses
    based on coherence dynamics, adequacy, and game events.
    """

    def __init__(self, adequacy_threshold: float = 0.70):
        """
        Initialize emotional valence computer.

        Args:
            adequacy_threshold: Threshold for active vs passive affects (θ)
        """
        self.adequacy_threshold = adequacy_threshold

        # Valence history (rolling window)
        self.valence_history: deque = deque(maxlen=100)
        self.affect_history: deque = deque(maxlen=50)

        # Current state
        self.current_valence: float = 0.0
        self.baseline_valence: float = 0.0  # Emotional baseline

        # Event weights for valence computation
        self.event_weights = {
            # Combat events
            'enemy_killed': 0.15,
            'took_damage': -0.10,
            'health_critical': -0.20,
            'combat_victory': 0.25,
            'combat_defeat': -0.30,
            'dodged_attack': 0.08,

            # Quest events
            'quest_started': 0.10,
            'quest_objective_complete': 0.12,
            'quest_completed': 0.30,
            'quest_failed': -0.25,

            # Social events
            'npc_dialogue_positive': 0.10,
            'npc_dialogue_negative': -0.10,
            'friendship_gained': 0.20,
            'enemy_made': -0.15,
            'trade_success': 0.08,

            # Progression events
            'level_up': 0.35,
            'skill_increased': 0.12,
            'new_spell_learned': 0.15,
            'new_item_acquired': 0.08,
            'treasure_found': 0.18,

            # Environmental events
            'discovered_location': 0.15,
            'entered_dangerous_area': -0.05,
            'escaped_danger': 0.20,
            'stuck_or_lost': -0.12,

            # Resource events
            'low_health_potion_used': 0.10,
            'ran_out_of_resources': -0.15,
            'gained_gold': 0.05,
            'lost_gold': -0.05,
        }

        print("[VALENCE] Emotional Valence Computer initialized")
        print(f"[VALENCE] Adequacy threshold (θ): {adequacy_threshold}")

    def compute_valence(
        self,
        game_state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]],
        coherence_delta: float,
        adequacy: float,
        events: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValenceState:
        """
        Compute emotional valence from current game state and events.

        Args:
            game_state: Current game state
            previous_state: Previous game state (for delta computation)
            coherence_delta: Change in coherence (Δ𝒞)
            adequacy: Current adequacy of ideas (Adeq)
            events: List of game events that occurred
            context: Optional additional context

        Returns:
            ValenceState with complete affective information
        """
        events = events or []
        context = context or {}

        # 1. Compute valence delta from events
        event_valence = self._compute_event_valence(events)

        # 2. Compute valence delta from state changes
        state_valence = self._compute_state_valence(game_state, previous_state)

        # 3. Compute valence from coherence dynamics (PRIMARY)
        coherence_valence = coherence_delta * 2.0  # Δ𝒞 directly affects valence

        # 4. Combine valence sources
        # Coherence is primary (40%), state changes (30%), events (30%)
        total_delta = (
            coherence_valence * 0.4 +
            state_valence * 0.3 +
            event_valence * 0.3
        )

        # 5. Update current valence
        self.current_valence += total_delta

        # Apply decay toward baseline (emotions fade over time)
        decay_rate = 0.05
        self.current_valence = (
            self.current_valence * (1 - decay_rate) +
            self.baseline_valence * decay_rate
        )

        # 6. Store in history
        self.valence_history.append(self.current_valence)

        # 7. Classify affect type
        affect_type = self._classify_affect_type(
            self.current_valence,
            total_delta,
            coherence_delta,
            game_state,
            adequacy
        )

        # 8. Determine active vs passive
        is_active = self._is_active_affect(adequacy, coherence_delta)

        # 9. Compute affect stability
        stability = self._compute_affect_stability()

        # 10. Extract component affects
        components = self._compute_component_affects(
            game_state,
            coherence_delta,
            events
        )

        # 11. Create valence state
        valence_state = ValenceState(
            valence=self.current_valence,
            valence_delta=total_delta,
            affect_type=affect_type,
            is_active=is_active,
            adequacy=adequacy,
            coherence_delta=coherence_delta,
            valence_history=list(self.valence_history),
            affect_stability=stability,
            component_affects=components
        )

        self.affect_history.append(affect_type)

        return valence_state

    def _compute_event_valence(self, events: List[str]) -> float:
        """Compute valence change from game events."""
        total = 0.0
        for event in events:
            if event in self.event_weights:
                total += self.event_weights[event]
        return total

    def _compute_state_valence(
        self,
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]]
    ) -> float:
        """Compute valence from state changes."""
        if not previous:
            return 0.0

        valence = 0.0

        # Health changes
        health_curr = current.get('health', 100)
        health_prev = previous.get('health', 100)
        health_delta = health_curr - health_prev
        valence += health_delta / 100.0 * 0.3  # Health is important

        # Combat state changes
        if current.get('in_combat') and not previous.get('in_combat'):
            valence -= 0.05  # Entering combat slightly negative
        elif not current.get('in_combat') and previous.get('in_combat'):
            # Exiting combat - depends on health
            if health_curr > 50:
                valence += 0.15  # Victory!
            else:
                valence += 0.05  # Survived, but barely

        # Enemy count changes
        enemies_curr = current.get('enemies_nearby', 0)
        enemies_prev = previous.get('enemies_nearby', 0)
        enemy_delta = enemies_curr - enemies_prev
        if enemy_delta < 0:
            valence += abs(enemy_delta) * 0.10  # Defeated enemies

        # Resource changes
        for resource in ['magicka', 'stamina']:
            curr_val = current.get(resource, 100)
            prev_val = previous.get(resource, 100)
            delta = curr_val - prev_val
            valence += delta / 200.0  # Less important than health

        # Gold changes
        gold_curr = current.get('gold', 0)
        gold_prev = previous.get('gold', 0)
        gold_delta = gold_curr - gold_prev
        valence += np.tanh(gold_delta / 100.0) * 0.05  # Diminishing returns

        # Level/skill changes
        level_curr = current.get('player_level', 1)
        level_prev = previous.get('player_level', 1)
        if level_curr > level_prev:
            valence += 0.30  # Level up is very positive

        return valence

    def _classify_affect_type(
        self,
        valence: float,
        valence_delta: float,
        coherence_delta: float,
        game_state: Dict[str, Any],
        adequacy: float
    ) -> AffectType:
        """
        Classify the dominant affect type.

        Uses Spinozist taxonomy of affects from ETHICA Part IV.
        """
        # Threshold for significant affect
        threshold = 0.10

        # Neutral if valence and delta are small
        if abs(valence) < threshold and abs(valence_delta) < threshold / 2:
            return AffectType.NEUTRAL

        # Check for complex/derived affects first
        in_combat = game_state.get('in_combat', False)
        enemies = game_state.get('enemies_nearby', 0)
        health = game_state.get('health', 100)

        # DESPAIR: Very negative valence + low health
        if valence < -0.40 and health < 30:
            return AffectType.DESPAIR

        # FEAR: Negative valence + danger present
        if valence < -0.10 and in_combat and enemies > 0:
            return AffectType.FEAR

        # COURAGE: Positive valence + danger present
        if valence > 0.10 and in_combat and enemies > 0:
            return AffectType.COURAGE

        # HOPE: Positive delta but still negative overall
        if valence_delta > 0.05 and valence < 0:
            return AffectType.HOPE

        # LOVE: Positive valence + external cause (NPC interaction)
        if valence > 0.15 and game_state.get('in_dialogue', False):
            return AffectType.LOVE

        # HATRED: Negative valence + external cause (combat)
        if valence < -0.15 and in_combat:
            return AffectType.HATRED

        # DESIRE: Positive coherence delta (seeking to increase coherence)
        if coherence_delta > 0.05:
            return AffectType.DESIRE

        # Basic affects: JOY vs SADNESS
        if valence > threshold:
            return AffectType.JOY
        elif valence < -threshold:
            return AffectType.SADNESS
        else:
            return AffectType.NEUTRAL

    def _is_active_affect(self, adequacy: float, coherence_delta: float) -> bool:
        """
        Determine if affect is active (from understanding) or passive (external).

        From MATHEMATICA D6:
        Active iff Adeq ≥ θ AND Δ𝒞 ≥ 0
        """
        return adequacy >= self.adequacy_threshold and coherence_delta >= 0

    def _compute_affect_stability(self) -> float:
        """
        Compute how stable the current affect is.

        High stability means emotion is consistent over time.
        Low stability means volatile/fluctuating emotions.
        """
        if len(self.valence_history) < 5:
            return 0.5  # Default

        recent = list(self.valence_history)[-10:]
        variance = np.var(recent)

        # Low variance = high stability
        stability = 1.0 / (1.0 + variance)
        return stability

    def _compute_component_affects(
        self,
        game_state: Dict[str, Any],
        coherence_delta: float,
        events: List[str]
    ) -> Dict[str, float]:
        """
        Compute multiple simultaneous affects (complex emotions).

        Humans experience multiple affects at once - this captures that.
        """
        components = {}

        # Joy component (from positive events)
        joy_score = 0.0
        for event in events:
            if event in self.event_weights and self.event_weights[event] > 0:
                joy_score += self.event_weights[event]
        if joy_score > 0:
            components['joy'] = min(joy_score, 1.0)

        # Fear component (from danger)
        if game_state.get('in_combat') and game_state.get('enemies_nearby', 0) > 0:
            health_factor = 1.0 - (game_state.get('health', 100) / 100.0)
            components['fear'] = health_factor * 0.5

        # Desire component (from coherence increase)
        if coherence_delta > 0:
            components['desire'] = min(coherence_delta * 2.0, 1.0)

        # Hope component (improving situation)
        if self.current_valence < 0 and len(self.valence_history) > 1:
            if self.valence_history[-1] > self.valence_history[-2]:
                components['hope'] = 0.3

        # Sadness component (from negative events)
        sadness_score = 0.0
        for event in events:
            if event in self.event_weights and self.event_weights[event] < 0:
                sadness_score += abs(self.event_weights[event])
        if sadness_score > 0:
            components['sadness'] = min(sadness_score, 1.0)

        return components

    def get_affect_summary(self) -> Dict[str, Any]:
        """Get summary of current affective state."""
        if not self.valence_history:
            return {
                'current_valence': 0.0,
                'avg_valence': 0.0,
                'affect_volatility': 0.0,
                'dominant_affects': []
            }

        valence_array = np.array(list(self.valence_history))

        return {
            'current_valence': self.current_valence,
            'avg_valence': np.mean(valence_array),
            'valence_std': np.std(valence_array),
            'valence_min': np.min(valence_array),
            'valence_max': np.max(valence_array),
            'affect_volatility': np.std(valence_array),
            'dominant_affects': [
                affect.value for affect in list(self.affect_history)[-5:]
            ],
            'active_affect_ratio': self._compute_active_ratio()
        }

    def _compute_active_ratio(self) -> float:
        """Compute ratio of active to passive affects in recent history."""
        # This requires tracking active/passive in history (future enhancement)
        # For now, estimate from adequacy trend
        return 0.5  # Placeholder

    def create_affect_object(self, valence_state: ValenceState) -> Affect:
        """
        Create an Affect object from ValenceState.

        Returns:
            Affect object from core types
        """
        return Affect(
            valence=valence_state.valence,
            valence_delta=valence_state.valence_delta,
            is_active=valence_state.is_active,
            adequacy_score=valence_state.adequacy,
            coherence_delta=valence_state.coherence_delta,
            affect_type=valence_state.affect_type.value
        )
