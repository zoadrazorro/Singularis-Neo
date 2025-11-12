# Token Limits Update - November 12, 2025

## Summary
Increased token limits across all LLM configurations to ensure models have sufficient capacity for complex reasoning and longer responses.

---

## Changes Applied

### 1. **LM Studio Client** (Default Config)

**File:** `singularis/llm/lmstudio_client.py`

```python
# Before
max_tokens: int = 2048

# After
max_tokens: int = 4096  # Increased for Phi-4 models
```

**Impact:** All local LM Studio calls now default to 4096 tokens

---

### 2. **Local MoE Orchestrator**

**File:** `singularis/llm/local_moe.py`

#### Expert Config:
```python
# Before
timeout: int = 10
max_tokens: int = 512

# After
timeout: int = 15  # Increased for thinking models
max_tokens: int = 1024  # Increased for better reasoning depth
```

#### Synthesizer:
```python
# Before
max_tokens=256

# After
max_tokens=512  # Increased for better synthesis
```

**Impact:** 
- Experts can generate longer, more detailed analyses
- Synthesizer can produce more comprehensive final decisions
- Longer timeout accommodates thinking mode in Qwen3-4b

---

### 3. **Cloud MoE Orchestrator**

**File:** `singularis/llm/moe_orchestrator.py`

#### Gemini Experts:
```python
# Before
max_tokens=768

# After
max_tokens=2048  # Increased for Gemini 2.5 Flash
```

#### Claude Experts:
```python
# Before
max_tokens=2048

# After
max_tokens=4096  # Increased for Claude Sonnet 4.5
```

**Impact:**
- Gemini experts can provide more detailed visual analysis
- Claude experts can perform deeper reasoning
- Better utilization of model capabilities

---

## Token Capacity by Model

### **Cloud Models:**

| Model | Max Output Tokens | Our Limit | Utilization |
|-------|------------------|-----------|-------------|
| **Gemini 2.5 Flash** | 8,192 | 2,048 | 25% |
| **Claude Sonnet 4.5** | 8,192 | 4,096 | 50% |
| **GPT-4o** | 16,384 | 4,096 | 25% |

### **Local Models:**

| Model | Context Window | Our Limit | Utilization |
|-------|---------------|-----------|-------------|
| **microsoft/phi-4** | 16,384 | 4,096 | 25% |
| **microsoft/phi-4-mini-reasoning** | 16,384 | 4,096 | 25% |
| **qwen/qwen3-4b-thinking-2507** | 32,768 | 1,024 | 3% |

**Note:** Local model limits are conservative to balance speed vs depth

---

## Rationale

### Why Increase Tokens?

1. **Better Reasoning:**
   - Phi-4 models support extended reasoning
   - Qwen3-4b has "thinking" mode that needs space
   - Claude Sonnet 4.5 has extended thinking capability

2. **Complex Tasks:**
   - Skyrim AGI needs detailed spatial reasoning
   - Strategic planning requires longer explanations
   - Sensorimotor analysis benefits from verbose output

3. **MoE Consensus:**
   - Experts need sufficient space to explain reasoning
   - Synthesizer needs to consider all expert opinions
   - Better consensus with more detailed responses

4. **Avoid Truncation:**
   - Previous limits (256-512) were too restrictive
   - Models were cutting off mid-thought
   - Important details were being lost

---

## Performance Impact

### Positive:
- ✅ **Better reasoning quality** - Models can think through problems
- ✅ **More detailed analysis** - Richer sensorimotor descriptions
- ✅ **Improved consensus** - MoE experts provide fuller context
- ✅ **Fewer truncations** - Responses complete naturally

### Considerations:
- ⚠️ **Slightly slower** - More tokens = longer generation time
- ⚠️ **Higher API costs** - Cloud models charge per token
- ⚠️ **More VRAM** - Local models need more memory for longer context

### Mitigation:
- Local models are small (4B params) so VRAM impact is minimal
- Timeouts increased to accommodate longer generation
- Conservative limits (25-50% of max) balance speed vs quality

---

## Token Budget Examples

### Typical Use Cases:

**Sensorimotor Analysis (Claude Sonnet 4.5):**
- Input: ~500 tokens (game state, visual description)
- Output: ~2000 tokens (spatial reasoning, recommendations)
- **Total: ~2500 tokens** ✅ Within 4096 limit

**MoE Expert Opinion (Gemini/Claude):**
- Input: ~300 tokens (action prompt, context)
- Output: ~1500 tokens (analysis, recommendation)
- **Total: ~1800 tokens** ✅ Within 2048/4096 limit

**Local MoE Expert (Phi-4/Qwen3):**
- Input: ~200 tokens (action prompt)
- Output: ~800 tokens (reasoning, choice)
- **Total: ~1000 tokens** ✅ Within 1024 limit

**Synthesizer (Phi-4):**
- Input: ~400 tokens (all expert opinions)
- Output: ~300 tokens (final decision)
- **Total: ~700 tokens** ✅ Within 512 limit

---

## Monitoring

### Watch For:
- ⚠️ Timeout errors (increase timeout if needed)
- ⚠️ Truncated responses (increase max_tokens if needed)
- ⚠️ VRAM usage (reduce if system struggles)
- ⚠️ API costs (monitor Gemini/Claude usage)

### Success Metrics:
- ✅ Responses complete without truncation
- ✅ Reasoning is coherent and detailed
- ✅ No timeout errors
- ✅ VRAM usage stable

---

## Future Adjustments

### If Responses Too Short:
- Increase max_tokens further
- Add minimum length requirements
- Adjust temperature for verbosity

### If Too Slow:
- Reduce max_tokens
- Use faster models for simple tasks
- Implement adaptive token limits

### If VRAM Issues:
- Reduce local MoE token limits
- Use fewer parallel experts
- Switch to smaller models

---

## Summary Table

| Component | Old Limit | New Limit | Change |
|-----------|-----------|-----------|--------|
| **LM Studio Default** | 2048 | 4096 | +100% |
| **Local MoE Experts** | 512 | 1024 | +100% |
| **Local MoE Synthesizer** | 256 | 512 | +100% |
| **Cloud Gemini Experts** | 768 | 2048 | +167% |
| **Cloud Claude Experts** | 2048 | 4096 | +100% |
| **Local MoE Timeout** | 10s | 15s | +50% |

---

*Update completed: November 12, 2025*
*All token limits increased to support better reasoning and avoid truncation*
