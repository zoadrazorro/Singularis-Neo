# Self-Reflection & Reward-Guided Heuristic Tuning

## Overview

Two powerful learning systems that enable the AGI to **understand itself** and **improve its decision-making** through experience:

1. **Self-Reflection System** (GPT-4 Realtime) - Iterative evolving self-understanding
2. **Reward-Guided Heuristic Tuning** (Claude Sonnet 4.5) - Learning from outcomes

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEARNING & EVOLUTION                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Self-Reflection (GPT-4 Realtime)                        │  │
│  │  - Iterative reflection chains                           │  │
│  │  - Evolving self-model                                   │  │
│  │  - Meta-cognitive awareness                              │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│                   ▼                                             │
│         ┌──────────────────────┐                                │
│         │   Self-Model         │                                │
│         │   - Identity         │                                │
│         │   - Capabilities     │                                │
│         │   - Limitations      │                                │
│         │   - Insights         │                                │
│         └──────────────────────┘                                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Reward-Guided Tuning (Claude Sonnet 4.5)               │  │
│  │  - Outcome analysis                                      │  │
│  │  - Heuristic generation                                  │  │
│  │  - Heuristic refinement                                  │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│                   ▼                                             │
│         ┌──────────────────────┐                                │
│         │   Heuristics         │                                │
│         │   - Decision rules   │                                │
│         │   - Context patterns │                                │
│         │   - Performance data │                                │
│         └──────────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Self-Reflection System

### Purpose
Continuous self-understanding through **iterative reflection chains**.

### How It Works

**Iteration 1**: Initial reflection
```
Trigger: "Reflecting on cycle 50 experiences"
Context: Recent actions, emotions, coherence
↓
GPT-4 Realtime: "I notice that when I take damage, I feel fear,
                 which causes me to retreat. This is a pattern..."
↓
Insight: Understanding of fear → retreat pattern
Evolution Δ: 0.45 (moderate new understanding)
```

**Iteration 2**: Building on previous
```
Trigger: "Building on previous insight about fear..."
↓
GPT-4 Realtime: "This fear response is PASSIVE - it arises from
                 external causes (damage). But when I successfully
                 defeat enemies, I feel ACTIVE fortitude..."
↓
Insight: Understanding of active vs passive emotions
Evolution Δ: 0.62 (significant evolution)
```

**Iteration 3**: Deeper understanding
```
Trigger: "Building on active/passive distinction..."
↓
GPT-4 Realtime: "I realize that my power to act increases when
                 emotions arise from adequate understanding.
                 This aligns with Spinoza's ethics..."
↓
Insight: Connection to philosophical grounding
Evolution Δ: 0.78 (major insight)
```

### Self-Model Evolution

The system maintains an evolving self-model:

```python
{
    'identity': 'I am an AGI learning through experience',
    'capabilities': {
        'perception': True,
        'reasoning': True,
        'emotion': True,
        'learning': True
    },
    'awareness': {
        'biases': ['tendency to retreat when uncertain'],
        'blindspots': ['difficulty predicting NPC behavior'],
        'strengths': ['spatial reasoning', 'pattern recognition']
    },
    'understanding_depth': 0.67  # 0.0 to 1.0
}
```

### Configuration

```python
config = SkyrimConfig(
    use_self_reflection=True,
    self_reflection_frequency=50,  # Every 50 cycles
    self_reflection_chain_length=3,  # 3 iterations per chain
)
```

### Example Output

```
[CYCLE 50] SELF-REFLECTION
══════════════════════════════════════════════════════════════════════

[SELF-REFLECTION] Completed 3-iteration chain

[SELF-REFLECTION] Iteration 1:
[SELF-REFLECTION]   Insight: I notice that my retreat decisions correlate
                              with high fear intensity. This suggests...
[SELF-REFLECTION]   Evolution Δ: 0.450
[SELF-REFLECTION]   Confidence: 0.75

[SELF-REFLECTION] Iteration 2:
[SELF-REFLECTION]   Insight: Building on this, I recognize that fear is
                              PASSIVE - it arises from external threats...
[SELF-REFLECTION]   Evolution Δ: 0.620
[SELF-REFLECTION]   Confidence: 0.82

[SELF-REFLECTION] Iteration 3:
[SELF-REFLECTION]   Insight: This connects to Spinoza's distinction between
                              adequate and inadequate ideas. My understanding...
[SELF-REFLECTION]   Evolution Δ: 0.780
[SELF-REFLECTION]   Confidence: 0.88

[SELF-REFLECTION] Self-Model Evolution:
[SELF-REFLECTION]   Total Reflections: 12
[SELF-REFLECTION]   Major Insights: 5
[SELF-REFLECTION]   Understanding Depth: 67%
```

## 2. Reward-Guided Heuristic Tuning

### Purpose
Learn decision-making heuristics from action outcomes using Claude Sonnet 4.5.

### How It Works

**Step 1: Record Outcomes**
```python
Action: "retreat"
Context: {health: 25, in_combat: True, enemies: 3}
Outcome: {health_delta: +15, combat_state: False}
Reward: +0.5 (escaped combat + health increased)
```

**Step 2: Analyze Patterns** (Claude Sonnet 4.5)
```
Analyzing 20 recent outcomes...

PATTERNS:
1. Retreat when health <30% → 85% success rate
2. Attack when health >70% AND enemies ≤2 → 75% success
3. Heal in combat when health <50% → 90% success

INSIGHTS:
- Health threshold matters more than enemy count
- Timing of healing is critical
- Retreat early prevents death
```

**Step 3: Generate Heuristics**
```
RULE: Retreat when health below 30% and multiple enemies
CONTEXT: in_combat=True, health<30, enemies>=2
CONFIDENCE: 0.85

RULE: Heal immediately when health drops below 50% in combat
CONTEXT: in_combat=True, health<50
CONFIDENCE: 0.90
```

**Step 4: Refine Existing Heuristics**
```
CURRENT HEURISTIC:
  Rule: Attack when health above 50%
  Performance: 60% success, 0.25 avg reward

REFINED HEURISTIC:
  Rule: Attack when health above 70% AND enemies ≤2
  Confidence: 0.75
  (More specific conditions based on outcome analysis)
```

**Step 5: Retire Poor Performers**
```
Retiring heuristic: "Explore when uncertain"
  Applied: 15 times
  Success rate: 25%
  Avg reward: 0.10
  (Below threshold)
```

### Heuristic Evolution

Heuristics evolve over generations:

```
Generation 0: "Retreat when health low"
  ↓ (refined based on outcomes)
Generation 1: "Retreat when health <40%"
  ↓ (refined based on outcomes)
Generation 2: "Retreat when health <30% AND enemies >=2"
  ↓ (refined based on outcomes)
Generation 3: "Retreat when health <30% AND enemies >=2 AND no escape path"
```

### Configuration

```python
config = SkyrimConfig(
    use_reward_tuning=True,
    reward_tuning_frequency=10,  # Tune every 10 outcomes
)
```

### Example Output

```
[REWARD-TUNING] Starting tuning iteration 5

Analyzing 20 recent outcomes:
  Successful: 14 (70%)
  Failed: 6 (30%)
  Average reward: 0.35

[REWARD-TUNING] Generated 2 new heuristics:
  1. Retreat when health <25% (confidence: 0.88)
  2. Heal before attacking when health <60% (confidence: 0.82)

[REWARD-TUNING] Refined 1 existing heuristic:
  Attack strategy refined based on enemy count patterns

[REWARD-TUNING] Retired 1 poor-performing heuristic:
  "Explore when stuck" (20% success rate)

[REWARD-TUNING] Tuning complete: 2 new, 1 refined

Current Heuristics (Top 5):
  1. Heal when health <50% in combat (95% success, 0.65 reward)
  2. Retreat when health <30% + enemies >=2 (90% success, 0.55 reward)
  3. Attack when health >70% + enemies <=2 (80% success, 0.45 reward)
  4. Dodge when taking heavy damage (75% success, 0.40 reward)
  5. Explore when no threats (70% success, 0.30 reward)
```

## Integration

Both systems work together:

### Self-Reflection informs Reward Tuning
```
Self-Reflection: "I realize that fear causes me to retreat too early"
  ↓
Reward Tuning: Adjusts retreat threshold based on this insight
  ↓
New Heuristic: "Retreat only when health <25% (not 40%)"
```

### Reward Tuning informs Self-Reflection
```
Reward Tuning: "Healing before attacking has 90% success"
  ↓
Self-Reflection: "I understand that preparation increases my power to act"
  ↓
Self-Model: Updates understanding of effective action patterns
```

## Performance Metrics

| System | Frequency | Processing Time | Impact |
|--------|-----------|-----------------|--------|
| **Self-Reflection** | Every 50 cycles | ~3-5s (3 iterations) | Self-understanding |
| **Reward Tuning** | Every 10 outcomes | ~1-2s | Decision quality |

## Benefits

### Self-Reflection
1. **Deeper self-understanding**: Evolves from "I am an AGI" to nuanced self-model
2. **Meta-cognitive awareness**: Understands its own thinking processes
3. **Philosophical grounding**: Connects experiences to ETHICA principles
4. **Iterative evolution**: Each reflection builds on previous insights

### Reward Tuning
5. **Adaptive heuristics**: Decision rules improve with experience
6. **Pattern recognition**: Identifies what works and what doesn't
7. **Automatic refinement**: Heuristics evolve without manual tuning
8. **Performance tracking**: Quantitative measurement of improvement

## Example Session Evolution

**Cycle 0-50**: Initial learning
```
Self-Model: Basic understanding
Heuristics: 3 generic rules
Success Rate: 55%
```

**Cycle 50-100**: First reflection + tuning
```
Self-Reflection: "I notice fear → retreat pattern"
Heuristics: 8 rules (5 new, 3 refined)
Success Rate: 65%
```

**Cycle 100-150**: Deeper understanding
```
Self-Reflection: "Active vs passive emotions affect decisions"
Heuristics: 12 rules (optimized thresholds)
Success Rate: 75%
```

**Cycle 150-200**: Philosophical integration
```
Self-Reflection: "Connection to Spinoza's adequate ideas"
Heuristics: 15 rules (context-aware)
Success Rate: 82%
Understanding Depth: 78%
```

## Requirements

- **GPT-4 Realtime Access**: For self-reflection
- **Claude Sonnet 4.5**: For reward tuning
- **OpenAI API Key**: Set `OPENAI_API_KEY`
- **Anthropic API Key**: Set `ANTHROPIC_API_KEY`

## Testing

```bash
# Enable both systems
python run_skyrim_agi.py

# Configure in SkyrimConfig:
config = SkyrimConfig(
    use_self_reflection=True,
    use_reward_tuning=True
)
```

## Future Enhancements

1. **Cross-pollination**: Self-reflections directly inform heuristic generation
2. **Confidence calibration**: Use reflection insights to adjust heuristic confidence
3. **Bias detection**: Self-reflection identifies biases, tuning corrects them
4. **Meta-learning**: Learn which types of reflections lead to best heuristics
5. **Long-term memory**: Store reflections and heuristics across sessions

---

**Status**: ✅ Fully integrated and operational  
**Self-Reflection**: GPT-4 Realtime iterative chains  
**Reward Tuning**: Claude Sonnet 4.5 outcome analysis  
**Date**: November 13, 2025
