# PHASE 1: EMERGENCY STABILIZATION (1 Day)

**Goal**: Stop competing action executors and add basic validation

---

## ✅ Step 1.1: Disable Competing Action Executors

**File**: `singularis/skyrim/skyrim_agi.py` (line 3801-3812)

**Change**: Comment out fast reactive loop and auxiliary exploration loop

```python
tasks = [perception_task, reasoning_task, action_task, learning_task]

# DISABLED: Causes action conflicts
# if self.config.enable_fast_loop:
#     fast_loop_task = asyncio.create_task(self._fast_reactive_loop(...))
print("[ASYNC] Fast reactive loop DISABLED (architecture fix)")

# DISABLED: Overrides planned actions  
# aux_exploration_task = asyncio.create_task(self._auxiliary_exploration_loop(...))
print("[ASYNC] Auxiliary exploration loop DISABLED (architecture fix)")
```

**Test**: Run for 10 minutes, verify no `[AUX-EXPLORE]` or `[FAST-LOOP]` logs

---

## ✅ Step 1.2: Add Perception Timestamp Validation

**File**: `singularis/skyrim/skyrim_agi.py`

**Add methods** (line ~2550):
```python
def _is_perception_fresh(self, perception_timestamp: float, max_age_seconds: float = 2.0) -> bool:
    age = time.time() - perception_timestamp
    if age > max_age_seconds:
        print(f"[VALIDATION] Perception stale: {age:.1f}s old")
        return False
    return True

def _validate_action_context(self, action: str, perception_timestamp: float, 
                             original_scene: str, original_health: float) -> Tuple[bool, str]:
    # Check freshness
    if not self._is_perception_fresh(perception_timestamp):
        return (False, f"Perception too old")
    
    # Check if scene/health changed significantly
    if self.current_perception:
        current_game_state = self.current_perception.get('game_state')
        if current_game_state:
            current_scene = str(self.current_perception.get('scene_type'))
            if current_scene != original_scene:
                return (False, f"Scene changed: {original_scene} → {current_scene}")
    
    return (True, "Valid")
```

**Modify action loop** (line ~6100):
```python
# Before executing action, validate context
is_valid, reason = self._validate_action_context(...)
if not is_valid:
    print(f"[ACTION] ❌ Rejected: {reason}")
    self.stats['action_rejected_count'] += 1
    continue
```

**Test**: Verify stale actions (>2s) are rejected

---

## ✅ Step 1.3: Single-Threaded Control Flow Test

**File**: `tests/test_skyrim_single_control.py` (NEW)

Create test that:
1. Disables fast/auxiliary loops
2. Runs core loops (perception → reasoning → action)
3. Measures override rate (should be <5%)
4. Measures latency (should be <5s)

See full test code in main implementation doc.

**Test**: `pytest tests/test_skyrim_single_control.py -v`

**Success**: Override rate <5%, latency <5s
