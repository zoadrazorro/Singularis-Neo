# Skyrim-Specific Cognition Refactor

## Overview

This document describes the restructuring of the Skyrim integration to use **game-specific cognition** instead of abstract philosophical concepts from the general AGI framework.

## Motivation

The original Skyrim integration used the Singularis AGI framework's philosophical concepts (coherence Δ𝒞, Three Lumina, ETHICA) to make gameplay decisions. While this demonstrated the framework's flexibility, it was overly abstract for a game-playing agent. Game-specific concepts provide:

1. **Better interpretability**: "health at 30%" vs "coherence at 0.45"
2. **Clearer motivation**: "survive combat" vs "increase coherence"
3. **More relevant evaluation**: "quest completed" vs "coherence increased"
4. **Domain expertise**: Skyrim mechanics vs universal philosophy

## Changes Made

### 1. New Skyrim Cognition Module (`skyrim_cognition.py`)

Created a new module defining game-specific cognitive dimensions:

#### **SkyrimCognitiveState**
Evaluates game state across 5 dimensions (each 0.0-1.0):

- **Survival**: Health ratio, safety from threats
- **Progression**: Skills, level, quest completion
- **Resources**: Gold, equipment quality, inventory space
- **Knowledge**: Map exploration, NPCs met, mechanics learned
- **Effectiveness**: Combat/stealth/social success rates

Replaces abstract "coherence" with concrete game metrics.

#### **SkyrimMotivation**
Game-specific motivations:

- **Survival** (35%): Stay alive, avoid damage
- **Progression** (25%): Level up, complete quests
- **Exploration** (20%): Discover new areas
- **Wealth** (10%): Gather gold and loot
- **Mastery** (10%): Improve combat/stealth skills

Replaces abstract "intrinsic motivation" (curiosity, coherence, etc.) with concrete game goals.

#### **SkyrimActionEvaluator**
Evaluates action outcomes:

- Quality change: Overall improvement/degradation
- Dimension changes: Which aspects improved/degraded
- Assessment: BENEFICIAL, NEUTRAL, or DETRIMENTAL

Replaces philosophical "ethical evaluation" (Δ𝒞 > 0) with practical outcome assessment.

### 2. Reinforcement Learning Updates

#### **State Encoder** (`reinforcement_learner.py`)
Changed state features from:
```python
features[32] = state.get('curiosity', 0.0)
features[33] = state.get('competence', 0.0)
features[34] = state.get('coherence', 0.0)
features[35] = state.get('autonomy', 0.0)
```

To:
```python
features[32] = state.get('player_level', 1) / 81.0
features[33] = state.get('gold', 0) / 10000.0
features[34] = state.get('completed_quests', 0) / 100.0
features[35] = state.get('equipment_quality', 0.3)
```

#### **Reward Function** (`reinforcement_learner.py`)
Changed from:
```python
coherence_delta = state_after.get('coherence', 0.5) - state_before.get('coherence', 0.5)
reward += coherence_delta * 2.0
```

To:
```python
cognitive_before = SkyrimCognitiveState.from_game_state(state_before)
cognitive_after = SkyrimCognitiveState.from_game_state(state_after)
quality_delta = cognitive_after.quality_change(cognitive_before)
reward += quality_delta * 2.0
```

Now rewards based on concrete game state improvement (health, progression, etc.) rather than abstract coherence.

### 3. RL Reasoning Neuron Updates (`rl_reasoning_neuron.py`)

#### **Renamed Metrics**
- `coherence_score` → `tactical_score`
- Measures how well LLM reasoning aligns with learned Q-values
- "Tactical" is game-specific vs "coherence" which is philosophical

#### **System Prompt**
Changed from philosophical framing:
```
"Increase coherence by explaining the 'why' behind learned policies"
```

To game-specific framing:
```
"Recommend actions that balance learned experience with tactical insight
for Skyrim gameplay"
```

#### **Statistics**
- `avg_coherence` → `avg_tactical_score`

### 4. Documentation Updates

Removed philosophical references from all file headers:

**Before:**
```python
"""
Philosophical grounding:
- ETHICA: Conatus (∇𝒞) drives autonomous behavior
- Coherence emerges from integrating symbolic reasoning
"""
```

**After:**
```python
"""
Design principles:
- Game-specific cognition drives autonomous behavior
- Decisions based on survival, skill progression, and effectiveness
"""
```

Files updated:
- `__init__.py`
- `actions.py`
- `perception.py`
- `controller.py`
- `action_affordances.py`
- `strategic_planner.py`
- `skyrim_agi.py`
- `skyrim_world_model.py`

### 5. World Model Updates (`skyrim_world_model.py`)

#### **Moral Evaluation**
Renamed method `evaluate_moral_choice`:

Changed from:
```python
return {
    'delta_coherence': 0.1,
    'ethical_status': "ETHICAL"
}
```

To:
```python
return {
    'impact_score': 0.1,
    'outcome_status': "BENEFICIAL"
}
```

Now evaluates game impact (bounty, rewards, relationships) rather than abstract ethics.

### 6. Gameplay Cycle Updates (`skyrim_agi.py`)

#### **Statistics Tracking**
Changed from:
```python
self.stats = {
    'coherence_history': [],
}

current_coherence = 0.5 + mot_state.coherence * 0.5
self.stats['coherence_history'].append(current_coherence)
```

To:
```python
self.stats = {
    'game_state_quality_history': [],
}

cognitive_state = SkyrimCognitiveState.from_game_state(game_state.to_dict())
self.stats['game_state_quality_history'].append(cognitive_state.overall_quality)
```

#### **Stats Reporting**
All stats output changed from "Avg Coherence" to "Avg Game State Quality".

## What Was NOT Changed

The following still use the base AGI framework (intentionally):

1. **Base AGI Orchestrator**: Still uses abstract motivation system for high-level goals
2. **LLM Integration**: Still uses the consciousness engine and expert system
3. **World Model Base**: Still uses Pearl's causality framework
4. **Active Inference**: Still uses free energy minimization

These provide the foundational AGI capabilities. The Skyrim-specific cognition layer sits on top, translating between game state and AGI concepts where needed.

## Benefits of This Approach

### 1. **Interpretability**
Decisions are now explained in game terms:
- "Low health (30%), should retreat" 
- NOT "Coherence decreasing, action unethical"

### 2. **Debugging**
Easier to debug when things go wrong:
- "Why did it attack at low health?" → Check survival weight
- NOT "Why is coherence decreasing?" → Check philosophical axioms

### 3. **Tuning**
Clear parameters to adjust:
```python
SkyrimMotivation(
    survival_weight=0.35,  # Make it more/less cautious
    progression_weight=0.25,  # Prioritize quests more/less
    # etc.
)
```

### 4. **Domain Expertise**
Code reflects Skyrim expertise, not philosophy expertise:
- Combat tactics (block, dodge, retreat)
- Resource management (healing potions, equipment)
- Quest optimization (skill requirements, dependencies)

### 5. **Maintainability**
Future developers don't need to understand:
- Spinoza's Ethics
- Three Lumina theory
- Coherence mathematics

They just need to understand Skyrim!

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│         BASE AGI FRAMEWORK                       │
│  (Consciousness, LLM Experts, World Model)      │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│      SKYRIM-SPECIFIC COGNITION LAYER            │
│                                                  │
│  • SkyrimCognitiveState (5 dimensions)          │
│  • SkyrimMotivation (game goals)                │
│  • SkyrimActionEvaluator (outcomes)             │
│  • Reinforcement Learning (game rewards)        │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│           SKYRIM GAME                            │
│  (Perception → Actions → Game State)            │
└─────────────────────────────────────────────────┘
```

## Example Comparison

### Decision Making

**Before (Philosophical):**
```
Query: "What should I do?"
Consciousness Level: 0.68
Coherence: 0.52
Action: explore (increases coherence)
Reasoning: "Exploration increases understanding, which increases 
           coherence (Δ𝒞 > 0), which is ethically good per ETHICA"
```

**After (Game-Specific):**
```
Query: "What should I do?"
Game State Quality: 0.65
  - Survival: 0.85 (health: 85%)
  - Progression: 0.45 (level 10, few quests)
  - Knowledge: 0.40 (40% map explored)
Action: explore (improves progression & knowledge)
Reasoning: "Moderate health, low quest completion. Should explore
           to discover new quests and level up skills."
```

### Reward Signal

**Before (Philosophical):**
```
State Change:
  coherence: 0.52 → 0.55 (Δ𝒞 = +0.03)
Reward: +0.06 (2.0 × Δ𝒞)
Interpretation: "Action increased coherence, therefore good"
```

**After (Game-Specific):**
```
State Change:
  survival: 0.85 → 0.80 (took damage)
  progression: 0.45 → 0.48 (gained XP)
  knowledge: 0.40 → 0.42 (discovered location)
  Overall Quality: 0.65 → 0.66 (+0.01)
Reward: +0.02 (2.0 × quality change)
Interpretation: "Took some damage but made progress, net positive"
```

## Testing

All existing tests still pass (no Skyrim-specific tests yet). The changes:

1. ✅ Syntax checks pass
2. ✅ Module imports work
3. ✅ No breaking changes to APIs used by base AGI

## Future Enhancements

Possible improvements to the game-specific cognition:

1. **Dynamic Weights**: Adjust motivation weights based on situation
   - In combat: survival_weight = 0.8
   - In town: exploration_weight = 0.4
   
2. **Play Style Learning**: Learn player's preferred style
   - Stealth archer: mastery_weight higher for archery
   - Tank: survival_weight higher
   
3. **Quest-Specific Goals**: Generate goals based on active quests
   - "Kill the dragon" → combat mastery focus
   - "Find the artifact" → exploration focus
   
4. **NPC-Aware Decisions**: Factor relationships into evaluation
   - Helping faction member: +0.2 to social effectiveness
   - Attacking ally: -0.5 to survival (consequences)

## Conclusion

This refactor makes the Skyrim integration more maintainable, interpretable, and effective by using domain-specific concepts instead of abstract philosophy. The system is now a **game-playing agent** rather than a **philosophical agent playing a game**.

The foundational AGI capabilities remain intact, but the Skyrim-specific layer provides the concrete, practical evaluation and motivation needed for effective gameplay.
