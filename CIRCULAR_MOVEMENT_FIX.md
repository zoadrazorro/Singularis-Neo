# Circular Movement Fix

## Problem
The AGI was walking in circles instead of exploring forward.

## Root Cause
The `explore` action used `explore_with_waypoints()` which had:
- 70% forward movement
- 15% left turns
- 15% right turns

Over time, the left/right turns (30% combined) caused circular patterns, especially when `explore` was the dominant action (which it was due to heuristic fallback).

## Solution

### 1. Simplified `explore` Action
Changed from waypoint-based to simple forward movement:

```python
# OLD (circular)
await self.actions.explore_with_waypoints(duration=3.0)

# NEW (straight)
await self.actions.move_forward(duration=2.5)
if random.random() < 0.3:
    await self.actions.look_around()
```

### 2. Reduced `explore` Frequency
Changed heuristics to prefer `move_forward` over `explore`:

**Curiosity Drive:**
- 40% → `activate` (interact)
- 50% → `move_forward` (straight)
- 10% → `explore` (fallback)

**Competence Drive:**
- Always → `move_forward` (was `explore`)

**Coherence Drive:**
- 40% → `move_forward`
- 60% → `move_forward` (was `explore`)

**Autonomy/Default:**
- 20% → `activate`
- 60% → `move_forward`
- 20% → `explore` (fallback only)

## Result
The AGI now:
- ✅ Moves primarily forward (straight lines)
- ✅ Occasionally looks around (30% when exploring)
- ✅ Still has variety (activate, jump, look_around)
- ✅ No more circular patterns

## Action Distribution (Expected)
- `move_forward`: ~50-60% (straight movement)
- `activate`: ~15-20% (interactions)
- `look_around`: ~10% (awareness)
- `jump`: ~8% (playfulness)
- `explore`: ~5-10% (fallback only)
- Other: ~5-10% (combat, rest, etc.)

## Monitoring
Watch for these patterns in logs:

### Good (Straight Movement) ✅
```
[HEURISTIC] → move_forward (direct movement)
[ACTION] Moving forward for 1.5s
[HEURISTIC] → activate (random curiosity)
[ACTION] Interacting with object/NPC
[HEURISTIC] → move_forward (coherent movement)
```

### Bad (Still Circular) ⚠️
```
[HEURISTIC] → explore (curiosity-driven exploration)
[EXPLORE] New direction: left
[HEURISTIC] → explore (autonomy/default)
[EXPLORE] New direction: right
```

If you still see circular patterns, it means:
1. RL is selecting `explore` too often (check Q-values)
2. LLM is suggesting `explore` (check LLM responses)
3. Need to further reduce `explore` probability in heuristics
