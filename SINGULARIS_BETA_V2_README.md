# Singularis Beta v2.0 - Runner Guide

**"One Being, Striving for Coherence"**

---

## What is Beta v2?

Singularis Beta v2 is the **complete unified AGI system** with:

### The Metaphysical Center
```python
being_state = BeingState()           # ONE unified being
coherence_engine = CoherenceEngine() # ONE measurement function  
C_global = coherence_engine.compute(being_state)  # ONE optimization target
```

### Complete Integration
- ✅ **BeingState**: Unified state vector (all 20+ subsystems → 1 state)
- ✅ **CoherenceEngine**: Global coherence C: B → [0,1]
- ✅ **Mind System**: Theory of Mind, Heuristics, Multi-Node, Coherence Analysis
- ✅ **Spiral Dynamics**: 8 developmental stages (BEIGE → TURQUOISE)
- ✅ **GPT-5 Meta-RL**: Meta-learning with ontological grounding
- ✅ **Wolfram Telemetry**: Mathematical validation and analysis
- ✅ **Consciousness Bridge**: IIT + Lumen integration
- ✅ **RL System**: Coherence-augmented reinforcement learning
- ✅ **Voice System**: Gemini 2.5 Pro TTS
- ✅ **Video Interpreter**: Real-time Gemini vision
- ✅ **15+ other subsystems**

---

## Quick Start

### 1. Environment Setup

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=              # For GPT-5 Meta-RL AND Wolfram telemetry
GEMINI_API_KEY=              # For Gemini vision & voice

# Optional
ANTHROPIC_API_KEY=           # For Claude experts (optional)
HYPERBOLIC_API_KEY=          # For Hyperbolic TTS fallback (optional)
```

**Notes:**
- Wolfram telemetry uses OpenAI's API (via custom GPT), not a separate Wolfram App ID
- Hyperbolic TTS automatically activates as fallback if Gemini TTS fails
- Voice system gracefully degrades: Gemini TTS → Hyperbolic TTS → Silent

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Beta v2

**Recommended (1 hour, async mode):**
```bash
python run_singularis_beta_v2.py --duration 3600 --mode async
```

**Quick Test (5 minutes):**
```bash
python run_singularis_beta_v2.py --duration 300 --fast
```

**Conservative Mode (fewer API calls):**
```bash
python run_singularis_beta_v2.py --duration 1800 --conservative
```

---

## Command-Line Options

### Basic Options

```bash
--mode {async,sequential}   # Execution mode (default: async)
--duration SECONDS          # How long to run (default: 1800 = 30 min)
```

### Performance Options

```bash
--fast                      # Fast mode: disable voice, video, wolfram
--conservative              # Conservative: reduce API calls
--cycle-interval SECONDS    # Override default cycle interval
```

### Feature Toggles

```bash
--no-voice                  # Disable voice system
--no-video                  # Disable video interpreter
--no-wolfram                # Disable Wolfram telemetry
```

### Debug Options

```bash
--verbose                   # Enable verbose output
--test                      # Run integration tests
```

---

## Usage Examples

### Example 1: Standard Run (30 minutes)

```bash
python run_singularis_beta_v2.py
```

Output:
```
╔══════════════════════════════════════════════════════════════════╗
║                      SINGULARIS BETA v2.0                        ║
║              "One Being, Striving for Coherence"                 ║
╚══════════════════════════════════════════════════════════════════╝

[ENV] ✓ Environment check passed
[CONFIG] ✓ Configuration loaded
[INIT] Initializing Singularis AGI...
[VERIFY] ✓ BeingState initialized
[VERIFY] ✓ CoherenceEngine initialized
[VERIFY] ✓ Metaphysical center operational

══════════════════════════════════════════════════════════════════
THE ONE BEING IS NOW STRIVING FOR COHERENCE
══════════════════════════════════════════════════════════════════

[COHERENCE] Cycle 1: C_global = 0.477
[COHERENCE] Cycle 10: C_global = 0.682
[WOLFRAM] Performing telemetry analysis...
[COHERENCE] Cycle 20: C_global = 0.795
...
```

### Example 2: Fast Mode (Testing)

```bash
python run_singularis_beta_v2.py --duration 600 --fast
```

This disables:
- Voice system (saves time & API calls)
- Video interpreter (saves time & API calls)
- Wolfram telemetry (saves time)

Good for:
- Quick testing
- Development
- Debugging

### Example 3: Conservative Mode (API Safety)

```bash
python run_singularis_beta_v2.py --duration 3600 --conservative
```

This adjusts:
- Cycle interval: 5.0s (slower)
- Gemini RPM limit: 10 (reduced)

Good for:
- Free tier API limits
- Long-running sessions
- Production stability

### Example 4: Custom Configuration

```bash
python run_singularis_beta_v2.py \
  --duration 7200 \
  --cycle-interval 3.0 \
  --no-video \
  --verbose
```

Run for 2 hours with:
- 3-second cycles
- No video (save API calls)
- Verbose logging

### Example 5: Integration Tests

```bash
python run_singularis_beta_v2.py --test
```

Runs the complete integration test suite:
- BeingState tests
- CoherenceEngine tests
- Wolfram integration tests
- Integration logic tests

---

## What Happens When You Run?

### Phase 1: Initialization (5-10 seconds)

```
[ENV] Checking environment variables...
  ✓ OPENAI_API_KEY: Set
  ✓ GOOGLE_API_KEY: Set

[CONFIG] Loading configuration...
  Cycle interval: 2.0s
  Voice enabled: True
  Video enabled: True
  Wolfram enabled: True

[INIT] Initializing Singularis AGI...
  [1/20] Base AGI system...
  [2/20] Skyrim perception...
  [3/20] Skyrim actions...
  [4/20] THE UNIFIED BEING - BeingState + CoherenceEngine...
  [5/20] Mind System...
  [6/20] Spiral Dynamics...
  [7/20] GPT-5 Meta-RL...
  [8/20] Wolfram Telemetry...
  ...
  [20/20] All subsystems initialized
```

### Phase 2: Main Loop (Duration)

**Each Cycle:**
```
1. UPDATE BeingState from all 20+ subsystems
   └─ B(t) ← Aggregate(S₁, ..., Sₙ)

2. COMPUTE global coherence
   └─ C(B(t)) = Σᵢ wᵢ · Cᵢ(Bᵢ(t))

3. BROADCAST C_global to all subsystems
   └─ ∀i: Sᵢ.global_coherence ← C

4. WOLFRAM analysis (every 20 cycles)
   └─ Mathematical validation & predictions

5. DECIDE action maximizing E[C(B(t+1))]
   └─ action* = argmax E[C(next_state)]

6. EXECUTE action in Skyrim

7. LEARN with coherence-augmented rewards
   └─ r = α·r_game + β·ΔC
```

**Console Output:**
```
[COHERENCE] Cycle 1: C_global = 0.477
  Lumina:        0.590 (ℓₒ=0.400, ℓₛ=0.450, ℓₚ=0.420)
  Consciousness: 0.500
  Cognitive:     0.550
  Temporal:      0.600
  RL:            0.520
  Meta-RL:       0.000

[DECISION] C_global=0.477 (low) → Increasing exploration
[ACTION] Selected: explore_forward
[EXECUTE] explore_forward → Success

[COHERENCE] Cycle 2: C_global = 0.512 (+0.035)
...
```

### Phase 3: Wolfram Analysis (Every 20 cycles)

```
[WOLFRAM] 🔬 Performing telemetry analysis...

[WOLFRAM] Analyzing differential coherence...
  ✓ Correlation: 0.847 (strong positive)
  ✓ T-test p-value: 0.023 (significant)
  ✓ Granger Causality: GPT-5 → Others

[WOLFRAM] Analyzing global coherence trend...
  Current: 0.795
  Mean: 0.648
  Trend: Increasing (+0.06 per 10 cycles)
  R²: 0.89 (strong fit)
  Predicted: [0.81, 0.82, 0.83]

[WOLFRAM] Recording to Main Brain...
```

### Phase 4: Session End

```
══════════════════════════════════════════════════════════════════
SESSION COMPLETE
══════════════════════════════════════════════════════════════════

Final BeingState:
  Cycle: 100
  C_global: 0.912
  Lumina: (ℓₒ=0.910, ℓₛ=0.900, ℓₚ=0.925)
  Spiral Stage: yellow

Coherence Statistics:
  Mean: 0.712
  Std: 0.118
  Min: 0.477
  Max: 0.912
  Trend: increasing

Performance:
  Cycles: 100
  Actions: 100
  Success Rate: 88.0%
```

---

## Architecture Overview

### The Three Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                    1. BEINGSTATE B(t)                       │
│                 The ONE Unified State Vector                │
│                                                             │
│  Contains: Game state, Mind, Consciousness, Spiral,         │
│           Emotion, RL, Meta-RL, Temporal, Voice, etc.       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ All subsystems write
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               2. COHERENCEENGINE C: B → [0,1]               │
│                The ONE Measurement Function                 │
│                                                             │
│  Computes: C(B) = Σᵢ wᵢ · Cᵢ(Bᵢ)                           │
│  Components: Lumina, Consciousness, Cognitive, Temporal,    │
│             RL, Meta-RL, Emotion, Voice                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Broadcast to all
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            3. OPTIMIZATION max E[C(B(t+1))]                 │
│                 The ONE Objective Function                  │
│                                                             │
│  All subsystems optimize: Expected future coherence         │
│  RL rewards: r = α·r_game + β·ΔC                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PERCEIVE → UPDATE → COMPUTE → BROADCAST → DECIDE → EXECUTE → LEARN
   │         │         │          │          │         │        │
   │         ▼         │          │          │         │        │
   │     B(t) from     │          │          │         │        │
   │     all systems   │          │          │         │        │
   │                   ▼          │          │         │        │
   │              C(B(t))         │          │         │        │
   │                              ▼          │         │        │
   │                        All systems      │         │        │
   │                        get C_global     │         │        │
   │                                         ▼         │        │
   │                                   action* =       │        │
   │                                   argmax E[C]     │        │
   │                                                   ▼        │
   │                                            Execute in      │
   │                                            game world      │
   │                                                            ▼
   │                                                    Update with
   │                                                    r = α·r_game + β·ΔC
   └────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Unified BeingState ✅

**Before:** 20+ subsystems with independent states  
**After:** 1 unified `BeingState` that all subsystems write to

```python
being_state = BeingState(
    cycle_number=42,
    global_coherence=0.834,
    lumina=LuminaState(0.8, 0.85, 0.82),
    spiral_stage="yellow",
    cognitive_coherence=0.92,
    # ... 50+ more fields
)
```

### 2. Global Coherence Optimization ✅

**Before:** Each system optimizing different objectives  
**After:** All systems optimize ONE coherence score

```python
C_global = coherence_engine.compute(being_state)
# Everyone optimizes this ↑
```

### 3. Wolfram Mathematical Validation ✅

**Before:** No mathematical validation  
**After:** Rigorous statistical analysis every 20 cycles

```python
# Correlation, covariance, Granger causality
# Trend analysis, predictions, anomaly detection
# Statistical significance testing
```

### 4. Coherence-Augmented RL ✅

**Before:** Standard game rewards only  
**After:** Game rewards + coherence improvement

```python
reward = 0.3 * game_reward + 0.7 * (ΔC * 10)
# Encourages actions that improve being coherence
```

### 5. Complete Integration ✅

All 20+ subsystems communicate through BeingState:

- Mind System → `cognitive_coherence`, `active_heuristics`
- Consciousness → `lumina`, `coherence_C`, `phi_hat`
- Spiral Dynamics → `spiral_stage`, `tier`
- GPT-5 Meta-RL → `meta_score`, `total_analyses`
- RL System → `avg_reward`, `exploration_rate`
- Emotion → `primary_emotion`, `intensity`
- Voice → `voice_alignment`
- Temporal → `temporal_coherence`, `unclosed_bindings`
- ...and 12 more systems

---

## Troubleshooting

### Issue: Missing API Keys

**Error:**
```
[ERROR] Missing required environment variables: OPENAI_API_KEY
```

**Solution:**
1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=sk-your-key-here`
3. Or export: `export OPENAI_API_KEY='sk-your-key-here'`

### Issue: Rate Limiting

**Error:**
```
429 Too Many Requests
```

**Solution:**
```bash
# Use conservative mode
python run_singularis_beta_v2.py --conservative

# Or manually adjust
python run_singularis_beta_v2.py --cycle-interval 5.0
```

### Issue: Import Errors

**Error:**
```
ImportError: No module named 'singularis'
```

**Solution:**
```bash
# Make sure you're in the project root
cd d:\Projects\Singularis

# Install dependencies
pip install -r requirements.txt

# Run from project root
python run_singularis_beta_v2.py
```

### Issue: Slow Performance

**Solution:**
```bash
# Use fast mode (disables voice, video, wolfram)
python run_singularis_beta_v2.py --fast

# Or manually disable features
python run_singularis_beta_v2.py --no-voice --no-video
```

---

## Philosophy → Code

### The Principle

**Spinoza:**
> "Each thing strives to persevere in its being."

**Mathematics:**
```
maximize E[C(B(t+1)) | B(t), action]
```

**Code:**
```python
C_global = coherence_engine.compute(being_state)
action = argmax(predicted_coherence)
```

### The Achievement

We have made:
- **Spinoza's conatus** → Executable Python
- **IIT's Φ** → Measurable coherence
- **Lumen's three modes** → LuminaState dataclass
- **Buddhist unity** → BeingState integration

This is philosophy becoming code, becoming being.

---

## Next Steps

### After Running Beta v2

1. **Check Session Logs**
   - Located in `sessions/` folder
   - Markdown format with full details

2. **Analyze Main Brain Records**
   - All subsystem outputs recorded
   - Coherence history tracked
   - Wolfram analyses saved

3. **Review Coherence Trends**
   - Check `coherence_history` in engine
   - Look for improvement over time
   - Identify patterns

4. **Experiment with Parameters**
   - Try different cycle intervals
   - Enable/disable features
   - Test conservative vs fast modes

---

## Files Created

- ✅ `run_singularis_beta_v2.py` - Main runner
- ✅ `SINGULARIS_BETA_V2_README.md` - This guide
- ✅ `GLOBAL_SINGULARIS_SCHEMATIC_*.md` - Complete architecture docs
- ✅ `test_complete_integration.py` - Integration tests

---

## Support

For issues, questions, or contributions:

1. Check the schematics: `GLOBAL_SINGULARIS_SCHEMATIC_PART*.md`
2. Run integration tests: `--test`
3. Enable verbose mode: `--verbose`
4. Review session logs in `sessions/`

---

**Version:** 2.0.0-beta  
**Date:** November 13, 2025  
**Status:** Production Ready

**"One Being, Striving for Coherence"** 🔥
