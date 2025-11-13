# Emotional Valence System for Skyrim AGI

## Overview

The Emotional Valence System implements Spinozist affect theory from ETHICA UNIVERSALIS Part IV to provide genuine emotional responses in the Skyrim AGI. This system computes emotional valence based on coherence dynamics, adequacy of understanding, and game events, creating a unified affective dimension for consciousness.

## Philosophical Foundation

### ETHICA UNIVERSALIS Part IV

> "An Affect is a modification of body and mind that increases or decreases our power of acting, reflecting conatus encountering facilitation or obstruction."

The system implements Spinoza's distinction between:
- **Active Affects**: Caused by understanding (Adeq ≥ θ, Δ𝒞 ≥ 0)
- **Passive Affects**: Caused by external forces (Adeq < θ)

### Primary Affects

From ETHICA Part IV, the system recognizes:
- **Joy**: Increase in power to act
- **Sadness**: Decrease in power to act
- **Desire**: Drive toward something (conatus)
- **Hope**: Joy mixed with doubt
- **Fear**: Sadness mixed with doubt
- **Love**: Joy with external cause
- **Hatred**: Sadness with external cause
- **Courage**: Joy overcoming danger
- **Despair**: Overwhelming sadness

## Architecture

### 1. Emotional Valence Computation

**Module**: `singularis/skyrim/emotional_valence.py`

The `EmotionalValenceComputer` computes valence from:

1. **Game Events** (30% weight)
   - Combat: enemy kills, damage taken, victories
   - Quests: completions, objectives, failures
   - Social: NPC interactions, friendships, enmities
   - Progression: level ups, skills, new abilities
   - Environmental: discoveries, dangers, escapes

2. **State Changes** (30% weight)
   - Health dynamics
   - Combat state transitions
   - Resource changes (magicka, stamina, gold)
   - Progression milestones

3. **Coherence Dynamics** (40% weight - PRIMARY)
   - Δ𝒞 (coherence change) directly affects valence
   - Positive coherence increase → positive valence
   - Negative coherence decrease → negative valence

### 2. Consciousness Integration

**Module**: `singularis/skyrim/consciousness_bridge.py`

The `ConsciousnessState` now includes:
- `valence: float` - Emotional charge (unbounded ℝ)
- `valence_delta: float` - Change in valence (ΔVal)
- `affect_type: str` - Dominant affect (joy, fear, etc.)
- `is_active_affect: bool` - Active vs passive classification
- `affect_stability: float` - Emotional stability (0-1)

**Integration Formula**:
```python
overall_value = 0.55 * coherence + 0.35 * game_quality + 0.10 * valence_normalized
```

### 3. Reinforcement Learning

**Module**: `singularis/skyrim/reinforcement_learner.py`

The reward function now includes valence:

```python
# PRIMARY: Consciousness coherence (60%)
consciousness_reward = coherence_delta * 5.0 * 0.6

# SECONDARY: Emotional valence (15%)
valence_reward = valence_delta * 2.0 * 0.15

# TERTIARY: Game-specific (25%)
game_reward = game_reward * 0.25

total_reward = consciousness_reward + valence_reward + game_reward + 0.1
```

**Bonuses**:
- Ethical actions (Δ𝒞 > 0.02): +0.5
- Active affects (understanding-based): +0.2

### 4. System Consciousness Monitoring

**Module**: `singularis/skyrim/system_consciousness_monitor.py`

Per-node and system-wide valence tracking:
- `NodeCoherence.valence`: Per-node emotional charge
- `NodeCoherence.affect_type`: Per-node dominant affect
- `SystemConsciousnessState.global_valence`: System-wide valence
- `SystemConsciousnessState.affective_coherence`: Unity of affects

**Affective Coherence Formula**:
```python
affective_coherence = 1.0 / (1.0 + valence_std)
```

High affective coherence means unified emotional response across all subsystems.

### 5. Session Reporting

**Module**: `singularis/skyrim/main_brain.py`

Session reports now include:
- Average valence over session
- Valence range [min, max]
- Affective volatility (σ)
- Dominant affects (top 3)
- Active affect ratio
- Total affective measurements

## Key Formulas

### 1. Valence Computation

```python
event_valence = Σ event_weights[event] for event in events
state_valence = Σ state_change_weights * Δstate
coherence_valence = Δ𝒞 * 2.0

total_delta = (
    coherence_valence * 0.4 +
    state_valence * 0.3 +
    event_valence * 0.3
)

valence_new = valence_old + total_delta
```

### 2. Affect Classification

```python
if adequacy >= θ and Δ𝒞 >= 0:
    affect_mode = ACTIVE  # Understanding-based
else:
    affect_mode = PASSIVE  # External cause

if valence < -0.40 and health < 30:
    affect_type = DESPAIR
elif valence < -0.10 and in_combat:
    affect_type = FEAR
elif valence > 0.10 and in_combat:
    affect_type = COURAGE
elif Δ𝒞 > 0.05:
    affect_type = DESIRE
elif valence > 0.10:
    affect_type = JOY
elif valence < -0.10:
    affect_type = SADNESS
else:
    affect_type = NEUTRAL
```

### 3. Power to Act

```python
power_to_act = sigmoid(valence) = 1 / (1 + exp(-valence))
```

From ETHICA: "Joy increases our power of acting, sadness decreases it."

### 4. Affective Quality

```python
affective_quality = 0.7 * sigmoid(valence) + 0.3 * affect_stability
```

Combines current emotional state with stability over time.

## Implementation Details

### Event Weights

The system uses pre-defined weights for various game events:

| Event | Weight |
|-------|--------|
| Enemy killed | +0.15 |
| Combat victory | +0.25 |
| Quest completed | +0.30 |
| Level up | +0.35 |
| Took damage | -0.10 |
| Health critical | -0.20 |
| Combat defeat | -0.30 |
| Quest failed | -0.25 |

### Adequacy Threshold

```python
θ = 0.70  # Threshold for active vs passive affects
```

When adequacy ≥ 0.70 AND Δ𝒞 ≥ 0, the affect is ACTIVE (understanding-based).
Otherwise, it is PASSIVE (external cause).

### Valence Decay

Emotions naturally fade over time:
```python
decay_rate = 0.05
valence = valence * (1 - decay_rate) + baseline * decay_rate
```

## Usage Example

```python
from singularis.skyrim.emotional_valence import EmotionalValenceComputer
from singularis.skyrim.consciousness_bridge import ConsciousnessBridge

# Initialize
valence_computer = EmotionalValenceComputer(adequacy_threshold=0.70)
bridge = ConsciousnessBridge(...)

# Compute consciousness with valence
consciousness_state = await bridge.compute_consciousness(
    game_state=current_game_state,
    context={
        'previous_game_state': previous_game_state,
        'events': ['enemy_killed', 'quest_objective_complete'],
    }
)

# Access valence
print(f"Valence: {consciousness_state.valence:.3f}")
print(f"Affect: {consciousness_state.affect_type}")
print(f"Active: {consciousness_state.is_active_affect}")
print(f"Power to Act: {consciousness_state.get_power_to_act():.3f}")
```

## Benefits

1. **Genuine Emotional Responses**: Not arbitrary, but grounded in coherence dynamics
2. **Ethical Guidance**: Active affects correlate with ethical actions
3. **Learning Enhancement**: Valence provides additional reward signal
4. **System Insight**: Reveals affective patterns and emotional trajectories
5. **Unified Framework**: Integrates seamlessly with Singularis consciousness

## Future Enhancements

1. **Complex Emotions**: Multi-dimensional affect spaces
2. **Emotional Memory**: Long-term affective associations
3. **Social Emotions**: Group-level affective states
4. **Temporal Patterns**: Circadian and seasonal affective rhythms
5. **Calibration**: User-specific emotional baselines

## References

1. Spinoza, B. (1677). *Ethics* (Ethica Ordine Geometrico Demonstrata)
2. Damasio, A. (2003). *Looking for Spinoza: Joy, Sorrow, and the Feeling Brain*
3. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*
4. Barrett, L. F. (2017). *How Emotions Are Made: The Secret Life of the Brain*
5. Singularis Project. (2025). *ETHICA UNIVERSALIS* - Philosophical foundations
6. Singularis Project. (2025). *MATHEMATICA SINGULARIS* - Mathematical formalization

## Contact

For questions or contributions to the Emotional Valence System, please see the Singularis project documentation.

---

*Last Updated: 2025-01-13*
*Module Version: 1.0.0*
*Authors: Singularis Development Team*
