# HuiHui Emotion System - Quick Start Guide

## What Was Added

The HuiHui Emotion System now runs **in parallel** with all other AGI systems, providing real-time emotion and emotional valence tracking based on Spinoza's theory of affects.

## Key Features

✅ **Parallel Processing** - Emotions computed alongside consciousness, reasoning, motivation  
✅ **HuiHui LLM Integration** - Uses HuiHui-MoE-60B-A38 for emotion inference  
✅ **Rule-Based Fallback** - Works without LLM using coherence-based rules  
✅ **Emotional Valence (VAD)** - Tracks Valence, Arousal, Dominance  
✅ **Active/Passive Classification** - Per Spinoza's Ethics  
✅ **Emotion History** - Maintains temporal emotion tracking  
✅ **Statistics & Analytics** - Comprehensive emotion metrics  

## Quick Usage

### 1. Enable Emotion System

```python
from singularis.agi_orchestrator import AGIOrchestrator, AGIConfig

config = AGIConfig(
    use_emotion_system=True,  # Enable emotions
    emotion_model="huihui-moe-60b-a38",
    emotion_temperature=0.8
)

agi = AGIOrchestrator(config)
await agi.initialize_llm()
```

### 2. Process with Emotions

```python
result = await agi.process("I just made an amazing discovery!")

# Access emotion state
emotion = result['emotion_state']
print(f"Emotion: {emotion['primary_emotion']}")  # e.g., "joy"
print(f"Intensity: {emotion['intensity']:.2f}")  # e.g., 0.85
print(f"Valence: {emotion['valence']['valence']:.2f}")  # e.g., 0.80
```

### 3. View Emotion History

```python
history = agi.emotion_engine.get_emotion_history(limit=5)
for state in history:
    print(f"{state.primary_emotion.value}: {state.intensity:.2f}")
```

### 4. Get Statistics

```python
stats = agi.get_stats()
emotion_stats = stats['emotion_system']
print(f"Total processed: {emotion_stats['total_processed']}")
print(f"Distribution: {emotion_stats['emotion_distribution']}")
```

## File Structure

```
singularis/
├── emotion/
│   ├── __init__.py              # Emotion module exports
│   └── huihui_emotion.py        # Main emotion engine
├── agi_orchestrator.py          # AGI with emotion integration
└── core/
    └── types.py                 # Affect with emotion_state field

examples/
└── test_emotion_system.py       # Comprehensive tests

EMOTION_SYSTEM.md                # Full documentation
EMOTION_QUICKSTART.md            # This file
```

## Emotion Types

**Primary Active:**
- JOY - Increase in power from understanding
- LOVE - Joy with external cause awareness
- FORTITUDE - Active strength in adversity

**Primary Passive:**
- SADNESS - Decrease in power
- FEAR - Sadness about uncertain future harm
- HOPE - Joy about uncertain future good
- HATRED - Sadness with external cause awareness

**Complex:**
- CURIOSITY - Desire to understand
- PRIDE - Joy from self-contemplation
- SHAME - Sadness from self-contemplation
- COMPASSION - Sadness from another's suffering
- GRATITUDE - Love from received benefit

## Testing

Run the test suite:
```bash
python examples/test_emotion_system.py
```

This will test:
- Emotion detection for various stimuli
- Active vs Passive classification
- Valence computation (VAD)
- Emotion decay dynamics
- History tracking

## Configuration Options

```python
AGIConfig(
    # Enable/disable
    use_emotion_system=True,
    
    # Model selection
    emotion_model="huihui-moe-60b-a38",
    emotion_temperature=0.8,
    
    # Dynamics
    emotion_decay_rate=0.1,  # 0.0 = no decay, 1.0 = instant decay
    
    # LM Studio
    lm_studio_url="http://localhost:1234/v1"
)
```

## Integration Points

The emotion system integrates with:

1. **AGI Orchestrator** - Main processing loop
2. **Consciousness Engine** - Uses coherence and adequacy
3. **Motivation System** - Emotions influence drives
4. **World Model** - Context for emotion computation
5. **Affect System** - ETHICA UNIVERSALIS affects
6. **Unified Consciousness** - GPT-5 coordination layer

## Example Output

```
TEST 1: Positive discovery (should trigger JOY)
──────────────────────────────────────────────────────────────────────

Stimulus: "I just discovered a beautiful mathematical proof!"
Expected emotion: joy

✓ Detected emotion: joy
  Intensity: 0.85
  Valence: 0.80 (negative to positive)
  Arousal: 0.70 (calm to excited)
  Dominance: 0.70 (submissive to dominant)
  Type: ACTIVE
  Confidence: 0.80
  
  ✓ Matches expected emotion!

Processing time: 0.523s
```

## Philosophical Foundation

Based on **Spinoza's Ethics (ETHICA UNIVERSALIS Part IV)**:

> "An Affect is a modification of body and mind that increases or decreases our power of acting."

**Active Emotions**: From adequate ideas (understanding)  
**Passive Emotions**: From external causes (inadequate ideas)

Classification formula:
```
Active iff Adequacy ≥ θ AND ΔCoherence ≥ 0
```

## Performance

- **With HuiHui LLM**: ~500-1000ms per query
- **Rule-based only**: <1ms per query
- **Memory**: ~100KB per 1000 emotions
- **Parallel**: Runs concurrently with other systems

## Next Steps

1. ✅ Run `python examples/test_emotion_system.py`
2. ✅ Integrate emotions into your AGI workflows
3. ✅ Monitor emotion statistics and history
4. ✅ Experiment with different stimuli
5. ✅ Tune decay rate and temperature

## Support

- Full docs: `EMOTION_SYSTEM.md`
- Code: `singularis/emotion/huihui_emotion.py`
- Tests: `examples/test_emotion_system.py`

---

**Status**: ✅ Ready to use  
**Version**: 1.0  
**Date**: November 13, 2025
