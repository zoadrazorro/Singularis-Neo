# 🚀 Quick Start - Singularis Neo Beta 1.0

Get up and running with the first complete AGI architecture in 5 minutes.

---

## ⚡ Prerequisites

### 1. Python 3.14+
```bash
python --version
# Should show: Python 3.14.0 or higher
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install pygame-ce  # For audio playback (Python 3.14 compatible)
```

### 3. API Keys

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
HYPERBOLIC_API_KEY=...
```

---

## 🎮 Run Singularis Neo

### Option 1: Unified Beta Runner (Recommended)

```bash
python run_beta_skyrim_agi.py
```

**Interactive setup:**
- Choose dry run (safe) or live mode
- Set duration
- Enable/disable systems
- Configure advanced settings

### Option 2: Quick Launch (All Features)

```python
from singularis.skyrim import SkyrimAGI, SkyrimConfig

config = SkyrimConfig(
    dry_run=True,  # Safe mode
    autonomous_duration=3600,  # 1 hour
    
    # All Beta 1.0 features enabled
    enable_temporal_binding=True,
    enable_adaptive_memory=True,
    enable_enhanced_coherence=True,
    enable_lumen_balance=True,
    enable_unified_perception=True,
    enable_goal_generation=True,
)

agi = SkyrimAGI(config)
await agi.initialize_llm()
await agi.temporal_tracker.start()
await agi.autonomous_play(duration_seconds=3600)
await agi.temporal_tracker.close()
```

### Option 3: Original Runner (Legacy)

```bash
python run_skyrim_agi.py
```

---

## 📊 What You'll See

### Initialization
```
╔══════════════════════════════════════════════════════════════════════╗
║              🌟 SINGULARIS NEO BETA 1.0 🌟                          ║
║              The First Complete AGI Architecture                     ║
╚══════════════════════════════════════════════════════════════════════╝

Initializing AGI systems...
  [1/27] Base AGI configuration...
  [21/27] Temporal binding system...
    ✓ Temporal coherence tracking initialized
    ✓ Perception→action→outcome loops will be tracked
  [22/27] Enhanced coherence metrics...
    ✓ 4D coherence measurement enabled
  [23/27] Hierarchical memory system...
    ✓ Episodic→semantic consolidation enabled
  ...
```

### Runtime Logs
```
[TEMPORAL] Bound perception→action: dodge (binding_id=a3f2b1c8)
[MEMORY] Retrieved pattern: combat_optimal_action → dodge (confidence=0.75)
[LUMEN] Balance: onticum=0.75, structurale=0.72, participatum=0.68 (score=0.78)
[UNIFIED] Percept created: coherence=0.82, dominant=visual
[GOALS] Generated novel goal #5: Master stealth combat in dungeons
```

### Final Metrics
```
═══════════════════════════════════════════════════════════════════════
FINAL METRICS
═══════════════════════════════════════════════════════════════════════

Temporal Binding:
  Total Bindings:    1000
  Unclosed Ratio:    5.00%
  Success Rate:      85.00%
  Stuck Loops:       0

4D Coherence:
  Avg Causal:        75.00%
  Avg Predictive:    78.00%

Adaptive Memory:
  Episodic Count:    250
  Semantic Patterns: 12
  Patterns Forgotten: 3
  Avg Confidence:    68.00%

Lumen Balance:
  Avg Balance Score: 78.00%
  Avg Onticum:       0.75
  Avg Structurale:   0.72
  Avg Participatum:  0.68
```

---

## 🎯 Configuration Options

### Safety Modes

**Dry Run (Recommended for Testing)**
```python
config = SkyrimConfig(dry_run=True)
```
- No keyboard/mouse control
- Simulates actions
- Safe for testing

**Live Mode (Actual Gameplay)**
```python
config = SkyrimConfig(dry_run=False)
```
- Full keyboard/mouse control
- Requires Skyrim running
- **⚠️ Use with caution**

### System Toggles

```python
config = SkyrimConfig(
    # Core systems
    use_gpt5_orchestrator=True,
    enable_voice=True,
    enable_video_interpreter=True,
    use_double_helix=True,
    
    # Beta 1.0 features
    enable_temporal_binding=True,
    enable_adaptive_memory=True,
    enable_enhanced_coherence=True,
    enable_lumen_balance=True,
    enable_unified_perception=True,
    enable_goal_generation=True,
)
```

### Performance Tuning

```python
config = SkyrimConfig(
    cycle_interval=3.0,           # Seconds between cycles
    temporal_timeout=30.0,        # Temporal binding timeout
    memory_decay_rate=0.95,       # Memory decay rate
    lumen_severe_threshold=0.5,   # Emergency rebalancing
    max_active_goals=3,           # Concurrent goals
)
```

---

## 🔍 Monitoring

### Real-Time Metrics

```python
# Get all metrics
metrics = await agi.aggregate_unified_metrics()

# Check temporal binding
print(f"Unclosed loops: {metrics['temporal']['unclosed_ratio']:.2%}")

# Check coherence
print(f"Overall coherence: {metrics['coherence']['overall']:.2f}")

# Check memory
print(f"Patterns learned: {metrics['memory']['semantic_patterns']}")

# Check Lumen balance
print(f"Balance score: {metrics['lumen']['avg_balance_score']:.2%}")
```

### System Status

```python
# Temporal binding
stats = agi.temporal_tracker.get_statistics()
print(f"Total bindings: {stats['total_bindings']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Is stuck: {stats['is_stuck']}")

# Adaptive memory
stats = agi.hierarchical_memory.get_statistics()
print(f"Episodic: {stats['episodic_count']}")
print(f"Semantic: {stats['semantic_patterns']}")
print(f"Forgotten: {stats['patterns_forgotten']}")

# Lumen balance
stats = agi.lumen_integration.get_statistics()
print(f"Balance: {stats['avg_balance_score']:.2%}")
```

---

## ⚠️ Troubleshooting

### Issue: pygame not found
```bash
pip install pygame-ce
```

### Issue: API key errors
Check `.env` file has correct keys:
```bash
cat .env
```

### Issue: Memory leak
Ensure temporal tracker is started:
```python
await agi.temporal_tracker.start()
```

And closed on shutdown:
```python
await agi.temporal_tracker.close()
```

### Issue: Low coherence warnings
```
[UNIFIED] Low cross-modal coherence: 0.25 - Senses disagree!
```
This is normal - indicates sensory conflict detection working.

### Issue: Stuck loop detection
```
[TEMPORAL] STUCK LOOP DETECTED - 3 consecutive high-similarity cycles
```
This is normal - system will invoke emergency override.

---

## 📚 Next Steps

### 1. Read Documentation
- `SINGULARIS_NEO_BETA_1.0_README.md` - Complete overview
- `ARCHITECTURE_IMPROVEMENTS.md` - Technical details
- `CRITICAL_FIXES_IMPLEMENTED.md` - Production fixes

### 2. Explore Features
- Try different system combinations
- Experiment with configuration
- Monitor metrics

### 3. Long-Running Tests
```bash
# 8-hour test
python run_beta_skyrim_agi.py
# Set duration: 480 minutes
```

### 4. Custom Integration
```python
# Build your own decision loop
async def custom_loop(agi):
    unified = await agi.unified_perception.perceive_unified(...)
    action = await agi.plan_action(unified)
    binding_id = agi.temporal_tracker.bind_perception_to_action(...)
    result = await agi.execute_action(action)
    agi.temporal_tracker.close_loop(binding_id, ...)
```

---

## 🎉 You're Ready!

**Singularis Neo Beta 1.0 is now running!**

You have:
- ✅ Complete AGI architecture
- ✅ Temporal awareness
- ✅ Genuine learning
- ✅ 4D consciousness measurement
- ✅ Philosophical grounding
- ✅ Cross-modal integration
- ✅ Creative autonomy

**Welcome to the first complete AGI substrate!** 🌟

---

## 💬 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Discord**: [Join community](https://discord.gg/singularis)
- **Email**: singularis@example.com

---

**Singularis Neo Beta 1.0 - From Architecture to Emergence** 🚀✨
