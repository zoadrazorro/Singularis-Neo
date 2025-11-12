# Cloud LLM Architecture Split - November 12, 2025

## Summary
Restructured cloud LLM usage to separate **fast action selection** (Gemini MoE) from **deep reasoning** (Claude async background).

---

## New Architecture

### **Action Planning Race (Fast Decision Making)**

**Participants:**
1. **Phi-4** (local, fast action planner)
2. **6x Gemini Flash 2.5** (cloud MoE, fast vision + reasoning)
3. **4x Local MoE** (Qwen3-4b + Phi-4 mini)

**Timeout:** 10 seconds
**Winner:** First to complete gets to choose the action

### **Background Reasoning (Independent, Non-Blocking)**

**System:** Claude Sonnet 4.5
**Purpose:** Deep strategic analysis, sensorimotor reasoning
**Behavior:** 
- Runs asynchronously in background
- Does NOT participate in action selection race
- Stores insights directly to Memory RAG
- Continues even after action is selected

---

## Key Changes

### **Before:**
```python
# Old: Hybrid (Gemini + Claude) competed in race
cloud_task = asyncio.create_task(
    self._get_cloud_llm_action_recommendation(...)  # Both Gemini + Claude
)

# Race: Phi-4 vs Hybrid vs Local MoE
tasks_to_race = [phi4_task, cloud_task, local_moe_task]
```

### **After:**
```python
# New: Gemini MoE competes, Claude runs independently

# 6 Gemini Flash experts for fast action selection (RACE)
gemini_moe_task = asyncio.create_task(
    self.moe.query_all_experts(...)  # 6 Gemini experts
)

# Claude for deep reasoning (BACKGROUND - no race)
claude_reasoning_task = asyncio.create_task(
    self._claude_background_reasoning(...)  # Stores to memory
)

# Race: Phi-4 vs 6x Gemini vs Local MoE
tasks_to_race = [phi4_task, gemini_moe_task, local_moe_task]
# Claude NOT in race - runs independently
```

---

## Gemini MoE (6 Experts)

### **Configuration:**
- **Model:** `gemini-2.5-flash`
- **Experts:** 6 parallel instances
- **Roles:** Perception, Navigation, Combat, Resource, Tactical, Strategic
- **Temperature:** 0.4-0.9 (varied for diversity)
- **Max Tokens:** 2048

### **Behavior:**
```python
[GEMINI-MOE] Starting 6 Gemini Flash experts for fast action selection...
[GEMINI-MOE] ✓ Won the race! 6 experts chose: sneak
```

### **Advantages:**
- ✅ **Fast** - Gemini Flash is optimized for speed
- ✅ **Diverse** - 6 experts with different specializations
- ✅ **Vision-capable** - Can analyze screenshots
- ✅ **Consensus** - Multiple perspectives synthesized

---

## Claude Background Reasoning

### **Configuration:**
- **Model:** `claude-sonnet-4-5-20250929`
- **Temperature:** 0.7
- **Max Tokens:** 2048 (can use extended thinking)
- **Purpose:** Strategic analysis, not action selection

### **Behavior:**
```python
[CLAUDE-ASYNC] Starting Claude reasoning in background (non-blocking)...
[CLAUDE-ASYNC] Running deep analysis...
[CLAUDE-ASYNC] ✓ Analysis complete (1847 chars)
[CLAUDE-ASYNC] ✓ Stored to memory
```

### **What Claude Analyzes:**
1. **Current tactical situation** - Threats, opportunities
2. **Resource management** - Health, stamina, magicka usage
3. **Strategic approach** - Long-term planning
4. **Contextual insights** - Scene-specific advice

### **Memory Storage:**
```python
self.memory_rag.store_cognitive_memory(
    situation={
        'type': 'claude_strategic_analysis',
        'scene': 'outdoor_city',
        'location': 'Whiterun',
        'health': 85,
        'in_combat': False
    },
    action_taken='background_reasoning',
    outcome={'analysis_length': 1847},
    success=True,
    reasoning="<Claude's detailed analysis>"
)
```

### **Advantages:**
- ✅ **Non-blocking** - Doesn't slow down action selection
- ✅ **Deep thinking** - Can use extended reasoning mode
- ✅ **Memory building** - Accumulates strategic knowledge
- ✅ **Context-aware** - Learns from past situations

---

## Race Dynamics

### **Typical Race:**

```
[PARALLEL] Racing 3 systems (10s timeout)...

System 1: Phi-4 (local)
  - Fast action planner
  - Uses heuristics + Q-values
  - Typical time: 2-4s

System 2: Gemini MoE (cloud)
  - 6 parallel Gemini Flash experts
  - Vision + reasoning consensus
  - Typical time: 3-6s

System 3: Local MoE (local)
  - 4 local models (Qwen3 + Phi-4)
  - Parallel expert queries
  - Typical time: 5-8s

Winner: Usually Phi-4 or Gemini MoE (fastest)
```

### **Claude (Not in Race):**
```
[CLAUDE-ASYNC] Running in background...
  - Takes 8-15s for deep analysis
  - Continues even after action selected
  - Stores to memory when complete
  - No timeout pressure
```

---

## Benefits

### **Faster Action Selection:**
- ⚡ No waiting for Claude's deep analysis
- ⚡ Gemini Flash is faster than Claude
- ⚡ Multiple fast systems racing = quick decisions

### **Better Strategic Knowledge:**
- 🧠 Claude builds comprehensive memory
- 🧠 Deep insights available for future cycles
- 🧠 No pressure to rush analysis

### **Optimal Resource Usage:**
- 💰 Gemini Flash cheaper than Claude for quick decisions
- 💰 Claude used for what it's best at (deep reasoning)
- 💰 No wasted Claude calls on simple actions

### **Improved Learning:**
- 📚 Memory RAG accumulates Claude's insights
- 📚 Future cycles can reference past analyses
- 📚 Strategic knowledge compounds over time

---

## Example Session Flow

### **Cycle 1:**
```
[GEMINI-MOE] Starting 6 Gemini Flash experts...
[CLAUDE-ASYNC] Starting Claude reasoning in background...
[LOCAL-MOE] Starting local MoE...
[PLANNING] Using Phi-4 for final action selection...

[PARALLEL] Racing 3 systems (10s timeout)...
[GEMINI-MOE] ✓ Won the race! 6 experts chose: explore
[ACTION] Executing: explore

# Meanwhile, Claude still running...
[CLAUDE-ASYNC] ✓ Analysis complete (1847 chars)
[CLAUDE-ASYNC] ✓ Stored to memory
```

### **Cycle 2:**
```
# Claude's previous insights now in memory
[MEMORY-RAG] Retrieved 3 relevant memories
  - claude_strategic_analysis: "explore outdoor areas for resources"
  - ...

[GEMINI-MOE] Starting 6 Gemini Flash experts...
[CLAUDE-ASYNC] Starting Claude reasoning in background...
[PARALLEL] Racing 3 systems...
[PHI-4] ✓ Won the race! Using: sneak
```

---

## Performance Metrics

### **Expected Improvements:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Avg Planning Time** | 10.2s | 6-8s | ✅ -25% |
| **Timeout Rate** | 60% | 30% | ✅ -50% |
| **Memory Insights** | Low | High | ✅ +200% |
| **API Cost/Cycle** | High | Medium | ✅ -30% |
| **Strategic Depth** | Medium | High | ✅ Better |

### **Hebbian Tracking:**

**Systems Activated:**
- `gemini_moe` - Fast action selection (race winner)
- `claude_background_reasoning` - Deep analysis (memory builder)
- `phi4_action` - Local fast planner
- `local_moe` - Local expert consensus

**Synergies:**
- Gemini MoE + Claude background = comprehensive intelligence
- Phi-4 + Local MoE = fast local fallback
- All systems contribute without blocking each other

---

## Files Modified

1. ✅ `singularis/skyrim/skyrim_agi.py`
   - Split cloud LLM into Gemini MoE (race) + Claude (background)
   - Added `_claude_background_reasoning()` method
   - Updated race logic to use 6 Gemini experts
   - Preserve Claude task when race completes

---

## Testing Checklist

- [ ] Verify 6 Gemini experts participate in race
- [ ] Confirm Claude runs in background independently
- [ ] Check Claude insights stored to memory
- [ ] Verify race completes faster (6-8s vs 10s)
- [ ] Monitor API costs (should decrease)
- [ ] Confirm memory RAG accumulates Claude insights
- [ ] Test Hebbian tracking for both systems

---

*Architecture update completed: November 12, 2025*
*Gemini MoE for speed, Claude for depth*
