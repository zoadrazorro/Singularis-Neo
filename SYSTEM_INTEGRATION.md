# System Integration: "Neurons that Fire Together, Wire Together"

## Overview
All AGI systems are connected through Hebbian learning, creating an adaptive neural network where successful collaborations are reinforced and systems learn to work together seamlessly.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Hebbian Integrator                         │
│              (Central Coordination Hub)                      │
│                                                              │
│  • Tracks temporal co-activation (30s window)               │
│  • Measures success correlations                            │
│  • Adapts system weights                                    │
│  • Applies synaptic decay                                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Visual       │   │ Reasoning    │   │ Strategic    │
│ Systems      │   │ Systems      │   │ Systems      │
└──────────────┘   └──────────────┘   └──────────────┘
│                   │                   │
│ • Gemini Vision   │ • Cloud LLM      │ • Sensorimotor
│ • Local Qwen3-VL  │ • Local MoE      │ • Singularis
│                   │ • Phi-4          │ • Huihui
│                   │                   │
└───────────────────┴───────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  Unified Action │
    │   (Coherent)    │
    └─────────────────┘
```

## Tracked System Pairs and Their Synergies

### 1. **Sensorimotor ↔ Gemini Vision**
**Why They Work Together:**
- Gemini provides detailed visual scene descriptions
- Sensorimotor uses this to reason about spatial navigation
- Both active every 5 cycles

**Hebbian Learning:**
```python
# When both succeed:
sensorimotor_claude45: weight += 0.1 × 1.0 (success) × recency
gemini_vision: weight += 0.1 × 0.8 (success) × recency
correlation(sensorimotor, gemini) += 0.1 × recency × 0.8

# Typical learned correlation: 1.5-2.0 (very strong)
```

**Example Integration:**
```
Cycle 5:
  [SENSORIMOTOR] Getting Gemini visual analysis...
  → Gemini: "Stone corridor, door ahead, pillar on right"
  [SENSORIMOTOR] Claude reasons about navigation
  → Recommendation: "Move forward, door is clear path"
  
  Hebbian Update:
  ✓ gemini_vision: success=True, strength=0.8
  ✓ sensorimotor_claude45: success=True, strength=1.0
  → Correlation += 0.08 (strengthened)
```

### 2. **Cloud LLM ↔ Phi-4 Planner**
**Why They Work Together:**
- Both compete in the parallel race
- When Cloud wins, Phi-4 still provided fast heuristic baseline
- Phi-4 validates Cloud's recommendations

**Hebbian Learning:**
```python
# Parallel race scenario:
# Both active within temporal window (racing)
# Cloud wins → both marked as active + successful
cloud_llm_hybrid: weight += 0.1 × 1.0 × 1.0 (immediate)
phi4_planner: weight += 0.1 × 0.8 × 0.9 (provided baseline)
correlation(cloud, phi4) += 0.1 × 1.0 × 0.72
```

**Example Integration:**
```
Cycle 10:
  [HEURISTIC-FAST] Phi-4 quick: "move_forward"
  [CLOUD-LLM] Claude reasoning: "sneak" (wins race)
  
  Hebbian Update:
  ✓ phi4_planner: success=True, strength=0.8 (provided baseline)
  ✓ cloud_llm_hybrid: success=True, strength=1.0 (won race)
  → Correlation += 0.08 (both contributed)
```

### 3. **Singularis ↔ Huihui Dialectical**
**Why They Work Together:**
- Singularis orchestrator uses Huihui for dialectical synthesis
- Always co-active (every 15 cycles)
- Deep strategic reasoning requires both

**Hebbian Learning:**
```python
# Dialectical reasoning:
huihui_dialectical: weight += 0.1 × 0.9 × 1.0
singularis_orchestrator: weight += 0.1 × 1.0 × 1.0
correlation(singularis, huihui) += 0.1 × 1.0 × 0.9 = 0.09

# Typical learned correlation: 1.6-1.9 (very strong)
```

### 4. **Local MoE ↔ Cloud LLM** (Fallback Synergy)
**Why They Work Together:**
- Run in parallel as backup
- When Cloud fails, Local MoE takes over
- Complementary strengths

**Hebbian Learning:**
```python
# Scenario 1: Both fail → both weakened
# Scenario 2: Cloud succeeds → Cloud strengthened, MoE neutral
# Scenario 3: MoE succeeds → MoE strengthened significantly
# Result: System learns when to rely on each
```

## Learning Dynamics

### Success Reinforcement
```python
def on_success(system_name, strength):
    # Increase system weight
    system_weight[system_name] += learning_rate × strength
    
    # Strengthen correlations with recently active systems
    for other_system in recent_activations:
        if time_diff < temporal_window:
            recency = 1.0 - (time_diff / window)
            delta = learning_rate × recency × strength_product
            correlation[(system, other)] += delta
```

### Failure Weakening
```python
def on_failure(system_name, strength):
    # Decrease system weight (smaller decrease)
    system_weight[system_name] -= learning_rate × 0.5 × strength
    
    # Weaken correlations
    for other_system in recent_activations:
        correlation[(system, other)] -= learning_rate × 0.3
```

### Synaptic Decay (Every 30 Cycles)
```python
def apply_decay():
    # Gradual weakening of unused connections
    for correlation in all_correlations:
        correlation.strength *= (1.0 - decay_rate)  # 0.99
    
    # Prevent unbounded weight growth
    for weight in system_weights:
        weight *= (1.0 - decay_rate)
```

## Real-World Adaptation Examples

### Example 1: Discovering Gemini-Sensorimotor Synergy

**Initial State (Cycle 0-10):**
```
gemini_vision: weight=1.0
sensorimotor_claude45: weight=1.0
correlation=0.0
```

**After 10 Successful Co-activations:**
```
gemini_vision: weight=1.35
sensorimotor_claude45: weight=1.48
correlation=1.65

Interpretation: System learned that Gemini's visual analysis 
                significantly enhances sensorimotor reasoning
```

**System Response:**
- Prioritizes Gemini visual analysis for sensorimotor cycles
- If Gemini fails, correlation doesn't grow
- Eventually learns optimal integration pattern

### Example 2: Cloud vs Local Tradeoff

**Cloud Reliable (80% success rate):**
```
cloud_llm_hybrid: weight=1.52
local_moe: weight=0.95
correlation=0.42 (moderate - both run, cloud wins more)
```

**Cloud Unreliable (40% success rate):**
```
cloud_llm_hybrid: weight=0.88
local_moe: weight=1.38
correlation=0.68 (stronger - local saves cloud failures)
```

**System Adaptation:**
- Learns to trust Local MoE more when Cloud is unreliable
- Correlation strengthens because they work together (fallback)
- Weights automatically adjust to current reliability

### Example 3: Exploration Synergy Discovery

**Discovered Pattern:**
```
Top Synergistic Pairs After 100 Cycles:
  1. sensorimotor_claude45 ↔ gemini_vision: 1.85
  2. gemini_vision ↔ local_vision_qwen: 1.42
  3. cloud_llm_hybrid ↔ phi4_planner: 1.38
```

**Interpretation:**
1. Visual systems work together for comprehensive scene understanding
2. Sensorimotor benefits most from Gemini (learned through success)
3. Cloud and Phi-4 have moderate synergy (racing scenario)

## Emergent Behaviors

### 1. **Automatic Specialization**
Systems naturally specialize based on success rates:
- Sensorimotor becomes expert at spatial reasoning
- Cloud LLM becomes go-to for complex decisions
- Local MoE becomes reliable fallback

### 2. **Resilient Fallback Chains**
System learns fallback preferences:
```
Primary: cloud_llm_hybrid (weight=1.45)
   ↓ (if fails)
Backup: local_moe (weight=1.28)
   ↓ (if fails)
Tertiary: phi4_planner (weight=1.15)
```

### 3. **Temporal Coordination**
Systems learn which combinations work well:
- Gemini → Sensorimotor (strong correlation)
- Singularis → Huihui → Claude (orchestrated)
- Fast reactive → Auxiliary exploration (complementary)

### 4. **Self-Optimization**
Through success/failure feedback:
- Weak systems get less weight over time
- Strong pairs get reinforced
- Unused connections decay away
- System adapts to changing conditions

## Monitoring and Diagnostics

### Status Report (Every 30 Cycles)
```
═══════════════════════════════════════════════════════════
HEBBIAN INTEGRATION STATUS
═══════════════════════════════════════════════════════════
Total Activations: 245
Successful Integrations: 198
Success Rate: 80.8%
Active Systems: 8
Correlations Tracked: 28

Strongest System: cloud_llm_hybrid (weight: 1.45)

Top Synergistic System Pairs:
  1. sensorimotor_claude45 ↔ gemini_vision: 1.85
  2. cloud_llm_hybrid ↔ phi4_planner: 1.62
  3. singularis_orchestrator ↔ huihui_dialectical: 1.58
  4. gemini_vision ↔ local_vision_qwen: 1.42
  5. local_moe ↔ cloud_llm_hybrid: 1.25

System Importance Weights:
  cloud_llm_hybrid: 1.45
  sensorimotor_claude45: 1.38
  gemini_vision: 1.25
  singularis_orchestrator: 1.18
  phi4_planner: 1.15
  huihui_dialectical: 1.12
  local_moe: 1.08
  local_vision_qwen: 0.98
═══════════════════════════════════════════════════════════
```

### Interpretation Guide

**High Correlation (>1.5):**
- Systems work together extremely well
- Should be co-activated when possible
- Example: sensorimotor + gemini_vision

**Moderate Correlation (0.8-1.5):**
- Systems complement each other
- Useful for fallback scenarios
- Example: cloud_llm + local_moe

**Low Correlation (<0.8):**
- Systems don't benefit from co-activation
- May work independently
- Example: auxiliary_exploration + singularis (different timescales)

**High Weight (>1.3):**
- System is very reliable
- Should be prioritized
- Example: cloud_llm_hybrid, sensorimotor

**Low Weight (<0.9):**
- System is underperforming
- May need tuning or replacement
- Naturally deprioritized

## Benefits of Hebbian Integration

1. **Automatic Optimization:** System learns optimal configurations
2. **Adaptive Resilience:** Adjusts to failures automatically
3. **Synergy Discovery:** Finds beneficial system combinations
4. **Resource Efficiency:** Prioritizes high-value systems
5. **Self-Healing:** Weak connections fade, strong ones grow
6. **Interpretability:** Status reports show what's working
7. **Continuous Learning:** Always adapting to new patterns

## Future Enhancements

- Use learned correlations to pre-activate complementary systems
- Dynamic timeout adjustment based on system weights
- Automatic fallback chain construction
- Cross-session learning (persist Hebbian weights)
- Visual network graph of system connections
- Predictive activation (activate systems before needed based on correlations)
