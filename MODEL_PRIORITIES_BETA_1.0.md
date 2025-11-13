# Model Priorities - Singularis Neo Beta 1.0

**Optimized for UNLIMITED models - Gemini Flash 2.5 Lite as PRIMARY**

---

## 🎯 Primary Models (UNLIMITED - No Rate Limits)

### 1. **Google Gemini Flash 2.5 Lite** - PRIMARY ⭐
- **Model**: `gemini-2.5-flash-lite`
- **Usage**: Main vision, reasoning, perception
- **Rate Limit**: **UNLIMITED** (optimized for high throughput)
- **Speed**: Fastest Gemini model
- **Status**: ✅ ENABLED (PRIMARY)

### 2. **Google Gemini 2.5 Flash Live** - Alternative
- **Model**: `gemini-2.5-flash-live`
- **Usage**: Real-time streaming
- **Rate Limit**: **UNLIMITED**
- **Status**: ✅ AVAILABLE

---

## 🎯 Secondary Models (High Limits)

### 1. **OpenAI GPT-5** - Meta-Cognitive Orchestrator
- **Model**: `gpt-5`
- **Usage**: Central coordination, meta-cognition
- **Rate Limit**: High (enterprise tier)
- **Status**: ✅ ENABLED

### 2. **Anthropic Claude** - Secondary Reasoning
- **Models**:
  - `claude-3-5-haiku-20241022` - Fast reasoning
  - `claude-sonnet-4-5-20250929` - Critical sensorimotor tasks
- **Usage**: Secondary reasoning, action planning, sensorimotor
- **Rate Limit**: 100 RPM (very high)
- **Status**: ✅ ENABLED (SECONDARY)

### 3. **Local Models** - Fast Fallback
- **Models**:
  - `microsoft/phi-4-mini-reasoning` - Action planning
  - `qwen/qwen3-vl-30b` - Vision/perception
- **Usage**: Fallback when APIs busy, fast local inference
- **Rate Limit**: None (local)
- **Status**: ✅ ENABLED (FALLBACK)

### 4. **Hyperbolic** - Additional Fallback
- **Usage**: Alternative cloud provider
- **Rate Limit**: High
- **Status**: ✅ ENABLED (if API key available)

---

## ✅ Optional Models (Can Enable)

### 1. **Google Gemini Voice/Video** - OPTIONAL
- **Models**:
  - `gemini-2.5-pro-preview-tts` (Voice)
  - `gemini-2.5-flash-native-audio-preview-09-2025` (Video)
- **Rate Limit**: Check your tier (may have limits)
- **Status**: ⚠️ OPTIONAL (disabled by default)

**Systems Affected:**
- Voice System (uses Gemini TTS) - Can enable if needed
- Video Interpreter (uses Gemini Flash) - Can enable if needed

**Note**: Main vision uses Flash 2.5 Lite (unlimited), so these are optional enhancements

---

## 🔧 Configuration

### Beta 1.0 Runner (`run_beta_skyrim_agi.py`)

```python
config = SkyrimConfig(
    # Primary: Claude
    use_hybrid_llm=True,
    use_claude_reasoning=True,
    claude_model="claude-3-5-haiku-20241022",
    claude_sensorimotor_model="claude-sonnet-4-5-20250929",
    
    # Orchestrator: GPT-5
    use_gpt5_orchestrator=True,
    
    # Fallback: Local
    use_local_fallback=True,
    
    # DISABLED: Gemini (rate limits)
    use_gemini_vision=False,
    enable_voice=False,
    enable_video_interpreter=False,
)
```

---

## 📊 Model Usage Breakdown

### Per Cycle (3 seconds)

| System | Model | Calls/Cycle | RPM Equivalent |
|--------|-------|-------------|----------------|
| **GPT-5 Orchestrator** | GPT-5 | 1-2 | ~20-40 RPM |
| **Claude Reasoning** | Claude Haiku | 2-3 | ~40-60 RPM |
| **Sensorimotor** | Claude Sonnet | 1 | ~20 RPM |
| **Local Fallback** | Phi-4/Qwen3 | As needed | ∞ |
| ~~Gemini Vision~~ | ~~Gemini Flash~~ | ~~2~~ | ~~40 RPM~~ ❌ |
| ~~Voice~~ | ~~Gemini TTS~~ | ~~1~~ | ~~20 RPM~~ ❌ |
| ~~Video~~ | ~~Gemini Flash~~ | ~~1~~ | ~~20 RPM~~ ❌ |

**Total API Calls**: ~4-6 per cycle (all within limits)

---

## ✅ Benefits of Current Configuration

### 1. **No Rate Limit Issues**
- Claude: 100 RPM limit (using ~60 RPM)
- GPT-5: High enterprise limit (using ~30 RPM)
- Local: Unlimited

### 2. **Fast Fallback Chain**
```
Primary: Claude Haiku (fast, reliable)
    ↓
Fallback 1: Local Phi-4 (instant, unlimited)
    ↓
Fallback 2: Local Qwen3-VL (vision, unlimited)
```

### 3. **Cost Effective**
- Claude Haiku: Very cheap
- Local models: Free
- GPT-5: Only for orchestration (minimal usage)

### 4. **Stable Long-Running**
- No 429 errors
- No rate limit cascades
- Can run 24+ hours continuously

---

## 🔮 Future: When Gemini Rate Limits Increase

Once Gemini rate limits are higher (paid tier or API improvements):

### Re-enable Voice System
```python
enable_voice=True,
voice_type="NOVA",
```

### Re-enable Video Interpreter
```python
enable_video_interpreter=True,
video_interpretation_mode="COMPREHENSIVE",
```

### Re-enable Gemini Vision
```python
use_gemini_vision=True,
gemini_model="gemini-2.5-flash",
```

---

## 🎯 Current Model Priorities (Summary)

**Tier 1 (Primary):**
- ✅ Claude Haiku/Sonnet (Anthropic)
- ✅ GPT-5 (OpenAI)

**Tier 2 (Fallback):**
- ✅ Phi-4 (Local)
- ✅ Qwen3-VL (Local)
- ✅ Hyperbolic (if available)

**Tier 3 (Disabled):**
- ❌ Gemini (rate limits)

---

## 📝 Notes

### Why Claude is Primary
1. **Fast**: Haiku is extremely fast
2. **Reliable**: 100 RPM limit (very high)
3. **Quality**: Sonnet 4.5 for critical tasks
4. **Cost**: Haiku is very cheap

### Why Local is Important
1. **Unlimited**: No rate limits
2. **Fast**: Runs on local GPU
3. **Privacy**: No API calls
4. **Fallback**: Always available

### Why Gemini is Disabled
1. **Rate Limits**: 30 RPM too low
2. **System Needs**: 120+ RPM required
3. **429 Errors**: Constant failures
4. **Stability**: Causes cascading failures

---

## 🚀 Running Beta 1.0

```bash
python run_beta_skyrim_agi.py
```

**You'll see:**
```
Core Systems:
  LLM:               ✓
  GPT-5 Orchestrator: ✓ (OpenAI)
  Claude Reasoning:  ✓ (Anthropic - primary)
  Local Fallback:    ✓ (Qwen3-VL, Phi-4)
  Voice System:      ✗ (Disabled - Gemini rate limits)
  Video Interpreter: ✗ (Disabled - Gemini rate limits)
  Double Helix:      ✓
```

**Result**: Stable, fast, no rate limit issues! ✅

---

**Singularis Neo Beta 1.0 - Optimized for Stability** 🚀
