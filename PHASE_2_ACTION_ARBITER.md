# PHASE 2: ACTION ARBITER (2-3 Days)

**Goal**: Create single point of action execution with priority system and validation

---

## ✅ Step 2.1: Implement ActionArbiter Class

**File**: `singularis/skyrim/action_arbiter.py` (NEW FILE)

**Create class with**:
- Priority enum (CRITICAL, HIGH, NORMAL, LOW)
- ActionRequest/ActionResult dataclasses
- Main ActionArbiter class

**Key methods**:
```python
async def request_action(action, priority, source, context, callback)
    → ActionResult
```

**Features**:
- Single action execution at a time
- Higher priority can preempt lower
- Validation before execution
- Statistics tracking
- Callback notifications

**Test**: Create `tests/test_action_arbiter.py` to verify priority preemption

---

## ✅ Step 2.2: Add Action Validation Logic

**In**: `action_arbiter.py` → `_validate_request` method

**Validations**:
1. ✅ Perception freshness (2s normal, 5s critical)
2. ✅ Scene consistency
3. ✅ Action availability (e.g., can't move in menu)
4. ✅ Health-based (no offense when health <25)
5. ✅ Combat context (no menus unless critical)
6. ✅ Repeated action detection (no 5x same action)

**Test**: Unit tests for each validation rule

---

## ✅ Step 2.3: Route All Actions Through Arbiter

**File**: `singularis/skyrim/skyrim_agi.py`

**Changes**:

1. **Initialize** (line ~1200):
```python
from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
self.action_arbiter = ActionArbiter(self)
```

2. **Update reasoning loop** (line ~5920):
```python
# Replace: self.action_queue.put_nowait(...)
# With:
result = await self.action_arbiter.request_action(
    action=action,
    priority=ActionPriority.NORMAL,
    source='reasoning_loop',
    context={...},
    callback=self._handle_action_result
)
```

3. **Update rule engine** (line ~7696):
```python
result = await self.action_arbiter.request_action(
    action=top_recommendation.action,
    priority=ActionPriority.HIGH,
    source='rule_engine',
    context={...}
)
```

**Test**: All actions now logged with `[ARBITER]` prefix

---

## ✅ Step 2.4: Add Source Tracking and Metrics

**File**: `singularis/skyrim/skyrim_agi.py`

**Add stats** (line ~1080):
```python
self.stats.update({
    'arbiter_total_requests': 0,
    'arbiter_executed': 0,
    'arbiter_rejected': 0,
    'arbiter_overridden': 0,
    'arbiter_by_source': {},
    'arbiter_by_priority': {},
    'perception_to_action_latency': [],
    'action_freshness_violations': 0,
    'action_context_mismatches': 0,
})
```

**Add logging** (every 20 cycles):
```python
if cycle_count % 20 == 0:
    arbiter_stats = self.action_arbiter.get_stats()
    print(f"\nACTION ARBITER STATS")
    print(f"Total: {arbiter_stats['total_requests']}")
    print(f"Executed: {arbiter_stats['executed']}")
    print(f"Rejection Rate: {arbiter_stats['rejection_rate']:.1%}")
    print(f"Override Rate: {arbiter_stats['override_rate']:.1%}")
```

**Add callback**:
```python
async def _handle_action_result(self, result: ActionResult):
    if not result.executed:
        self.stats['action_rejected_count'] += 1
        if 'too old' in result.reason.lower():
            self.stats['action_freshness_violations'] += 1
```

**Test**: Run for 30min, monitor rejection/override rates
**Success**: Rejection rate 5-10%, Override rate <1%
