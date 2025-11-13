# Expert Rule System for Skyrim AGI

## Overview

The Expert Rule System provides **fast, deterministic problem detection and response** before expensive LLM reasoning. It implements a rule-based expert system that can:

- **Detect known patterns** (stuck loops, scene mismatches, etc.)
- **Assert facts** in working memory
- **Recommend actions** with priorities
- **Block actions** for N cycles
- **Adjust system parameters** dynamically

This is inspired by classic expert systems but integrated into a modern AGI architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Rule Engine                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐      ┌──────────────┐                  │
│  │   Rules     │ ───▶ │  Conditions  │                  │
│  │  (Priority) │      │  (Evaluate)  │                  │
│  └─────────────┘      └──────────────┘                  │
│         │                     │                          │
│         │                     ▼                          │
│         │              ┌──────────────┐                  │
│         └────────────▶ │ Consequences │                  │
│                        │  (Execute)   │                  │
│                        └──────────────┘                  │
│                               │                          │
│         ┌─────────────────────┼────────────┐            │
│         ▼                     ▼            ▼            │
│  ┌──────────┐         ┌──────────┐  ┌──────────┐       │
│  │  Facts   │         │  Recs    │  │  Blocks  │       │
│  │ (Memory) │         │(Actions) │  │(Actions) │       │
│  └──────────┘         └──────────┘  └──────────┘       │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Implemented Rules

### Rule 1: Stuck in Loop
**Condition:**
```python
IF visual_similarity > 0.95 AND recent_actions.count("explore") > 2:
```

**Consequences:**
- Set fact: `stuck_in_loop` (confidence: 0.95)
- Recommend: `move_backward` (priority: HIGH)
- Recommend: `activate` (priority: MEDIUM)
- Block action: `explore` for 3 cycles

**Purpose:** Detect when the agent is stuck in a repetitive exploration pattern with no visual progress.

---

### Rule 2: Scene Classification Mismatch
**Condition:**
```python
IF scene_classification != visual_description.scene_type:
```

**Consequences:**
- Set fact: `sensory_conflict` (confidence: 0.90)
- Recommend: `activate` (priority: HIGH)
- Increase parameter: `sensorimotor_authority` to 1.5x

**Purpose:** Detect when the scene classifier disagrees with visual perception, suggesting the agent needs to interact to resolve state ambiguity.

---

### Rule 3: Visual Stasis
**Condition:**
```python
IF visual_similarity > 0.97 AND len(unique_actions[-4:]) >= 3:
```

**Consequences:**
- Set fact: `visual_stasis` (confidence: 0.85)
- Recommend: `jump` (priority: HIGH)
- Recommend: `turn_around` (priority: MEDIUM)

**Purpose:** Detect when the agent tries different actions but the screen doesn't change (soft-lock).

---

### Rule 4: Action Thrashing
**Condition:**
```python
IF len(set(recent_actions[-5:])) == 5:  # All different
```

**Consequences:**
- Set fact: `action_thrashing` (confidence: 0.80)
- Recommend: most recent action (priority: MEDIUM) to commit

**Purpose:** Detect rapid action switching (indecision) and encourage commitment to action sequences.

---

### Rule 5: Unproductive Exploration
**Condition:**
```python
IF recent_actions.count("explore") >= 3 AND coherence_delta < 0.01:
```

**Consequences:**
- Set fact: `unproductive_exploration` (confidence: 0.75)
- Set fact: `needs_goal_revision` (confidence: 0.80)

**Purpose:** Detect when exploration doesn't improve coherence and suggest goal replanning.

---

## Integration Points

### 1. Planning Phase (Early)
```python
# In _plan_action(), BEFORE expensive LLM operations
rule_context = {
    'visual_similarity': perception.get('visual_similarity', 0.0),
    'recent_actions': self.action_history[-10:],
    'scene_classification': scene_type,
    'visual_scene_type': perception.get('scene_type'),
    'coherence_history': self.coherence_history[-10:],
    # ... more context
}

rule_results = self.rule_engine.evaluate(rule_context)

# Check for immediate high-priority recommendations
top_rec = self.rule_engine.get_top_recommendation(exclude_blocked=True)
if top_rec and top_rec.priority.value >= 3:  # HIGH or CRITICAL
    return top_rec.action  # Use immediately, skip LLM
```

### 2. Action Filtering
```python
# Filter blocked actions from available actions
available_actions = [
    a for a in game_state.available_actions 
    if not self.rule_engine.is_action_blocked(a)
]
```

### 3. Cycle Management
```python
# At start of each game cycle
self.rule_engine.tick_cycle()  # Decrements block counters
```

## Usage Example

```python
from singularis.skyrim.expert_rules import RuleEngine, Priority

# Initialize
engine = RuleEngine()

# Each cycle:
context = {
    'visual_similarity': 0.97,
    'recent_actions': ['explore', 'explore', 'explore'],
    'scene_classification': SceneType.OUTDOOR_WILDERNESS,
    'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
    'coherence_history': [0.75, 0.76, 0.75],
    'health': 100,
    'in_combat': False,
    'enemies_nearby': 0,
}

results = engine.evaluate(context)

# Check results
if results['fired_rules']:
    print(f"Rules fired: {results['fired_rules']}")
    
if results['recommendations']:
    top = results['recommendations'][0]
    print(f"Top recommendation: {top.action} (priority: {top.priority.name})")
    
if results['blocked_actions']:
    print(f"Blocked: {results['blocked_actions']}")

# Get immediate action if high priority
action = engine.get_top_recommendation(exclude_blocked=True)
if action and action.priority.value >= 3:
    # Use this action immediately
    return action.action

# Advance cycle
engine.tick_cycle()
```

## Performance Benefits

### Speed
- **< 1ms** rule evaluation vs **2-30 seconds** for LLM
- Rules fire before expensive operations start
- No API calls, no rate limits

### Reliability
- **Deterministic** responses to known patterns
- No hallucination, no reasoning failures
- **100% reproducible** behavior

### Complementarity
- Rules handle **known problems** (fast path)
- LLMs handle **novel situations** (slow path)
- Working together, not competing

## Monitoring

### Status Report
```python
report = engine.get_status_report()
print(report)
```

Output:
```
═══════════════════════════════════════════════════════════
                    RULE ENGINE STATUS
═══════════════════════════════════════════════════════════
Total Rules: 5
Rules Fired (Total): 127
Active Facts: 2
Active Recommendations: 3
Blocked Actions: 1

Active Facts:
  • stuck_in_loop (confidence: 0.95)
  • sensory_conflict (confidence: 0.90)

Recommendations:
  • [HIGH] move_backward - Stuck in loop - retreat (0.90)
  • [MEDIUM] activate - Try pressing/activating obstacle (0.80)
  • [MEDIUM] jump - Break visual stasis (0.75)

Blocked Actions:
  • explore for 2 cycles - Stuck in exploration loop

Recent Rules Fired (last 10):
  • stuck_in_loop
  • scene_mismatch
  • visual_stasis
═══════════════════════════════════════════════════════════
```

### Dashboard Integration
The rule engine status should be displayed in the web dashboard:
- Active facts
- Current recommendations
- Blocked actions
- Rules fired this cycle

## Adding New Rules

```python
# 1. Define condition
def _cond_my_rule(self, context: Dict[str, Any]) -> bool:
    # Return True if rule should fire
    return context.get('some_metric') > threshold

# 2. Define consequences
def _cons_my_action(self, context: Dict[str, Any], engine: 'RuleEngine'):
    engine.set_fact("my_fact", confidence=0.9)
    engine.recommend("my_action", Priority.HIGH, "reason")

# 3. Register rule
self.add_rule(Rule(
    name="my_rule",
    description="Detects X and does Y",
    priority=8,
    condition=self._cond_my_rule,
    consequences=[self._cons_my_action]
))
```

## Testing

Run the test suite:
```bash
python test_expert_rules.py
```

Tests cover:
- All rule conditions
- Action blocking over multiple cycles
- Fact assertion
- Recommendation generation
- Parameter adjustment
- No false positives

## Future Extensions

### Possible Additions
1. **Combat rules** - Detect disadvantageous combat situations
2. **Inventory rules** - Detect when healing items should be used
3. **Quest rules** - Detect when objectives can be completed
4. **Social rules** - Detect NPC interaction opportunities
5. **Temporal rules** - Time-based patterns (e.g., "stuck for 10 cycles")

### Learning Integration
Rules could be **learned** from experience:
- Extract patterns from successful/failed episodes
- Use LLM to propose new rule conditions
- A/B test rules and keep effective ones
- Evolve rule priorities based on outcomes

### Fuzzy Logic
Current rules are binary (fire/don't fire). Could extend to:
- **Fuzzy conditions** - partial truth values
- **Confidence propagation** - combine multiple weak signals
- **Temporal reasoning** - rules that build up evidence over time

## References

- Classic Expert Systems (MYCIN, CLIPS)
- Production Rule Systems
- Forward Chaining Inference
- RETE Algorithm (for optimization)
