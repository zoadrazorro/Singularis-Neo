# Curriculum RL Integration Guide

## What We Built

A **curriculum-aware RL reward system** that teaches your AGI motor skills progressively while optimizing for coherence:

### Components Created

1. **`curriculum_reward.py`** - Progressive reward function
   - Stage 0: Basic locomotion (walk, turn)
   - Stage 1: Navigation (avoid walls)
   - Stage 2: Target acquisition (hit targets)
   - Stage 3: Defense (avoid damage)
   - Stage 4: Combat 1v1 (win fights)
   - Stage 5: Mastery (full competence)

2. **`curriculum_symbolic.py`** - Symbolic rules per stage
   - IF-THEN rules that guide behavior
   - Stage-specific logic (e.g., "IF stuck THEN turn")

3. **`curriculum_integration.py`** - Easy-to-use wrapper
   - Blends Δ𝒞 (coherence) + game progress
   - Auto-advances through stages
   - Provides symbolic rules for current stage

## How It Works

### Reward Formula

```python
R_total = (0.6 × Δ𝒞) + (0.4 × Stage_Progress) + Bonuses
```

Where:
- **Δ𝒞**: Coherence gain (consciousness improvement)
- **Stage_Progress**: Curriculum-specific reward
  - Stage 0: +1.0 for movement actions
  - Stage 1: +2.0 for exploring new areas, -2.0 for stuck
  - Stage 2: +2.0 for attacks near targets
  - Stage 3: +2.0 for taking no damage in combat
  - Stage 4: +5.0 for defeating enemies, +1.0 for attacks
- **Bonuses**: +5.0 for graduating to next stage

### Curriculum Progression

```
Stage 0 (Locomotion)
  ├─ 20 successful movement cycles → Advance
  │
Stage 1 (Navigation)
  ├─ 20 successful exploration cycles → Advance
  │
Stage 2 (Target Acquisition)
  ├─ 20 successful attack cycles → Advance
  │
Stage 3 (Defense)
  ├─ 20 cycles avoiding damage → Advance
  │
Stage 4 (Combat 1v1)
  ├─ 20 successful combat cycles → Advance
  │
Stage 5 (Mastery)
```

## Integration into SkyrimAGI

### Option A: Replace Existing RL Reward (Recommended)

**Step 1: Import in `skyrim_agi.py`**

```python
from singularis.learning.curriculum_integration import CurriculumIntegration
```

**Step 2: Initialize in `__init__`**

```python
# Add after motor controller init
self.curriculum = CurriculumIntegration(
    coherence_weight=0.6,  # 60% coherence, 40% progress
    progress_weight=0.4,
    enable_symbolic_rules=True
)
```

**Step 3: Use in learning loop**

Replace the existing `reinforcement_learner.compute_reward()` call with:

```python
# In _learning_loop or wherever rewards are computed
reward = self.curriculum.compute_reward(
    state_before=state_before,
    action=action,
    state_after=state_after,
    consciousness_before=consciousness_before,
    consciousness_after=consciousness_after
)

# Enqueue for RL training
experience = Experience(
    state=state_before,
    action=action,
    reward=reward,  # ← Now curriculum-aware!
    next_state=state_after,
    done=False,
    consciousness_before=consciousness_before,
    consciousness_after=consciousness_after
)
```

**Step 4: Add symbolic rule guidance (optional)**

```python
# In action planning, before choosing action
rules_info = self.curriculum.get_current_rules(current_state.to_dict())

if rules_info.get('suggested_action'):
    print(f"[SYMBOLIC] Rule suggests: {rules_info['suggested_action']}")
    # Optionally use this to bias action selection
```

**Step 5: Display curriculum stats**

```python
# In stats display
curriculum_stats = self.curriculum.get_stats()
print(f"\n📚 Curriculum Stage: {curriculum_stats['current_stage']}")
print(f"   Cycles: {curriculum_stats['stage_cycles']}")
print(f"   Successes: {curriculum_stats['stage_successes']}/{self.curriculum.reward_fn.progress.advancement_threshold}")
print(f"   Avg Reward: {curriculum_stats['avg_reward']:.2f}")
```

### Option B: Parallel Evaluation (Research Mode)

Keep existing RL but also track curriculum progress:

```python
# Compute both rewards
original_reward = self.reinforcement_learner.compute_reward(...)
curriculum_reward = self.curriculum.compute_reward(...)

# Use original for learning, but track curriculum
print(f"[COMPARE] Original: {original_reward:.2f} | Curriculum: {curriculum_reward:.2f}")

# Log for analysis
self.stats['reward_comparison'].append({
    'cycle': self.cycle_count,
    'original': original_reward,
    'curriculum': curriculum_reward,
    'delta': curriculum_reward - original_reward
})
```

## Benefits

1. **Progressive Learning** - Master basic skills before complex ones
2. **Coherence + Competence** - Blend philosophical depth with physical skill
3. **Interpretable** - Know exactly which skill is being trained
4. **Symbolic Guidance** - Rules provide explainable behavior
5. **Auto-Advancement** - System graduates agent through stages
6. **Testable** - Can reset to specific stage for focused training

## Testing Each Stage

### Stage 0: Locomotion
```python
self.curriculum.reset_stage(CurriculumStage.STAGE_0_LOCOMOTION)
# Should see: Agent walks, turns, explores
# Success: Lots of movement, variety of actions
```

### Stage 1: Navigation
```python
self.curriculum.reset_stage(CurriculumStage.STAGE_1_NAVIGATION)
# Should see: Agent avoids getting stuck, turns when needed
# Success: Visual similarity stays low, no wall-staring
```

### Stage 2: Target Acquisition
```python
self.curriculum.reset_stage(CurriculumStage.STAGE_2_TARGET_ACQUISITION)
# Should see: Agent approaches and attacks targets
# Success: Damage dealt to practice dummies
```

### Stage 3: Defense
```python
self.curriculum.reset_stage(CurriculumStage.STAGE_3_DEFENSE)
# Should see: Agent blocks, dodges, minimizes damage
# Success: Health stays high during combat
```

### Stage 4: Combat 1v1
```python
self.curriculum.reset_stage(CurriculumStage.STAGE_4_COMBAT_1V1)
# Should see: Agent wins simple 1v1 fights
# Success: Enemies defeated, positive K/D ratio
```

## Architecture Flow

```
Perception → State Encoding
    ↓
Consciousness System (Δ𝒞)
    ↓
╔═══════════════════════════╗
║  Curriculum Integration    ║
║  - Current Stage          ║
║  - Symbolic Rules         ║
║  - Progress Tracking      ║
╚═══════════════════════════╝
    ↓
Motor Controller (execute)
    ↓
Reward = 0.6×Δ𝒞 + 0.4×Progress
    ↓
Q-Network Update → Better Policy
```

## Symbolic Rules Example

Current Stage: **NAVIGATION**

```python
rules = [
    "IF visual_similarity > 0.95 AND action == move_forward THEN turn_or_jump",
    "IF visual_similarity < 0.80 THEN reward_exploration",
    "IF stuck_counter > 5 THEN large_turn_and_jump"
]

# Evaluated each cycle
# Provides explainable guidance: "I turned because I was stuck"
```

## Expected Behavior Improvements

### Before Curriculum
- Random flailing in combat
- Walking into walls repeatedly
- No progressive skill development
- Coherence ↑ but physical competence flat

### After Curriculum
- Purposeful movement (Stage 0 mastery)
- Smart navigation (Stage 1 mastery)
- Accurate attacks (Stage 2 mastery)
- Defensive combat (Stage 3 mastery)
- Combat victories (Stage 4 mastery)
- **Coherence ↑ AND physical competence ↑**

## Next Steps

1. ✅ Components created
2. ⏳ Wire into SkyrimAGI (follow Option A above)
3. ⏳ Test Stage 0 (locomotion)
4. ⏳ Monitor progression through stages
5. ⏳ Tune stage advancement thresholds
6. ⏳ Add more sophisticated symbolic rules

Your philosopher-baby will now **learn to walk before running**! 🧠➡️🦶➡️⚔️
