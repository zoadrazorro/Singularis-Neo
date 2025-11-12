# Complete Skyrim AGI Setup Guide

## Prerequisites

1. **Skyrim Special Edition** installed and running
2. **Python 3.10+** installed
3. **LM Studio** running (optional but recommended)
4. **48GB VRAM** (you have this!)

## Step-by-Step Setup

### 1. Install Core Dependencies

```bash
cd d:\Projects\Singularis

# Install base requirements
pip install -r requirements.txt

# Install CLIP for vision (required for Skyrim)
pip install git+https://github.com/openai/CLIP.git
```

### 2. Verify Installation

```bash
# Test imports
python -c "from singularis.skyrim import SkyrimAGI; print('✓ Skyrim integration ready')"

# Test CLIP
python -c "import clip; print('✓ CLIP installed')"

# Test screen capture
python -c "import mss; print('✓ Screen capture ready')"

# Test input control
python -c "import pyautogui; print('✓ Input control ready')"
```

### 3. Configure Skyrim

**In Skyrim Settings:**
- Set to **Windowed mode** or **Borderless window**
- Resolution: 1920x1080 (or note your resolution)
- Disable motion blur (helps vision)
- Enable subtitles (helps text reading)

**Save your game:**
- Make a new save in a safe location (e.g., Whiterun)
- Name it "AGI_Test" or similar

### 4. Start LM Studio (Recommended for Enhanced Instruction)

For the best SkyrimAGI experience with verbose strategic instructions:

```bash
# 1. Open LM Studio
# 2. Load model: eva-qwen2.5-14b-v0.2 (recommended for instructor)
#    This model provides detailed, actionable gameplay guidance
# 3. Start server on localhost:1234
# 4. Verify it's running
```

Note: The eva-qwen2.5-14b-v0.2 model is optimized for generating verbose strategic instructions like "go to this location because it will afford you specific items and progression benefits." This is specifically configured for the meta-strategist (instructor) component.

Test LM Studio:
```bash
python examples/test_connection.py
```

## Running the AGI

### Option 1: Safe Test (Recommended First)

This runs in **dry run mode** - it won't control your game, just observes and plans.

```bash
python examples/skyrim_quickstart.py
```

**What it does:**
- Captures screen every 2 seconds
- Analyzes what it sees (CLIP vision)
- Plans actions based on perception
- Learns causal relationships
- Shows what it would do (but doesn't execute)

**Expected output:**
```
SINGULARIS AGI - SKYRIM INTEGRATION
====================================
Initializing components...
  [1/4] Base AGI system...
  [2/4] Perception (CLIP vision)...
  [3/4] Actions (dry run mode)...
  [4/4] World model...

Starting autonomous gameplay for 1 minute...

[Cycle 1] Scene: OUTDOOR_WILDERNESS
  Detected: ['tree', 'mountain', 'sky']
  Goal: EXPLORE (curiosity=0.8)
  Action: MOVE_FORWARD (dry run)
  
[Cycle 2] Scene: COMBAT
  Detected: ['bandit', 'sword', 'shield']
  Goal: SURVIVE (threat detected)
  Action: ATTACK (dry run)
  Learned: bandit_nearby → combat_state
  
...
```

### Option 2: Full Demo with Options

```bash
python examples/skyrim_demo.py
```

Choose from menu:
1. **Basic demo** - 5 min dry run
2. **Custom demo** - Configure all options
3. **Test components** - Test perception/actions individually

### Option 3: Actual Gameplay (Advanced)

⚠️ **WARNING: This will control your keyboard and mouse!**

**Before running:**
1. ✅ Start Skyrim and load your save
2. ✅ Make sure you're in a safe location
3. ✅ Have a recent save backup
4. ✅ Test with dry_run=True first
5. ✅ Know the failsafe: **Move mouse to top-left corner to abort**

Create `run_skyrim_agi.py`:

```python
import asyncio
from singularis.skyrim import SkyrimAGI, SkyrimConfig

async def main():
    config = SkyrimConfig(
        dry_run=False,  # ENABLE ACTUAL CONTROL
        autonomous_duration=3600,  # 1 hour
        cycle_interval=2.0,  # 2 second cycles
        save_interval=300,  # Auto-save every 5 min
    )
    
    agi = SkyrimAGI(config)
    
    # Initialize LLM for smarter play
    await agi.initialize_llm()
    
    print("\n⚠️  AGI will control Skyrim in 5 seconds...")
    print("Move mouse to top-left corner to abort!")
    await asyncio.sleep(5)
    
    # Run!
    await agi.autonomous_play()
    
    # Show stats
    stats = agi.get_stats()
    print(f"\n✓ Learned {stats['world_model']['causal_edges']} causal rules")
    print(f"✓ Met {len(stats['world_model']['npc_relationships'])} NPCs")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python run_skyrim_agi.py
```

## What to Expect

### First 5 Minutes

The AGI will:
1. **Perceive** - Capture screen, identify scene type
2. **Explore** - Move around randomly (high curiosity)
3. **Learn** - Build causal graph from experiences
4. **Avoid danger** - Run from threats initially

### After 30 Minutes

The AGI will:
1. **Navigate** - Know the local area
2. **Combat** - Learned basic attack patterns
3. **NPCs** - Remember who it's met
4. **Causal rules** - "Steal → bounty", "Attack chicken → town hostile"

### After 2 Hours

The AGI will:
1. **Goals** - Form autonomous objectives (explore dungeon, help NPC)
2. **Ethics** - Evaluate choices by Δ𝒞 (coherence change)
3. **Strategies** - Develop play style (stealth, combat, magic)
4. **Relationships** - Track NPC friendships/hostilities

## Monitoring

### Real-Time Stats

The AGI prints stats every 10 cycles:

```
[Cycle 50] Stats:
  Coherence: 0.67 (+0.05 from start)
  Causal rules learned: 23
  NPCs met: 7
  Locations discovered: 4
  Dominant motivation: CURIOSITY (0.72)
```

### Detailed Analysis

After running, check:

```python
stats = agi.get_stats()

# Gameplay
print(stats['gameplay']['cycles_completed'])
print(stats['gameplay']['actions_taken'])
print(stats['gameplay']['coherence_history'])

# Learning
print(stats['world_model']['causal_edges'])
print(stats['world_model']['npc_relationships'])
print(stats['world_model']['locations_discovered'])

# Motivations
print(stats['motivations']['curiosity_over_time'])
print(stats['motivations']['dominant_drive'])
```

## Troubleshooting

### "CLIP not found"

```bash
pip install git+https://github.com/openai/CLIP.git
```

### "Screen capture returns black"

- Ensure Skyrim is running
- Try windowed mode instead of fullscreen
- Check screen_region in config matches your monitor

### "Actions not executing"

- Verify `dry_run=False`
- Make sure Skyrim window is in focus
- Check key bindings match your Skyrim controls

### "High CPU/GPU usage"

Normal! The AGI uses:
- CLIP vision: ~150MB VRAM
- LLM (Huihui 60B): ~31GB VRAM
- Total: ~32GB / 48GB ✓

### "AGI keeps dying in combat"

Early on, it will! It's learning. After ~1 hour it should:
- Recognize threats
- Use block/dodge
- Retreat when low health

### "AGI ignores quests"

By design! It forms **autonomous goals** based on intrinsic motivation:
- Curiosity → Explore new areas
- Competence → Practice combat
- Coherence → Help NPCs (if Δ𝒞 > 0)

It may eventually do quests if they align with its drives.

## Advanced Configuration

### Custom Screen Region

```python
config = SkyrimConfig(
    screen_region={
        'top': 100,
        'left': 200,
        'width': 1600,
        'height': 900
    }
)
```

### Custom Key Bindings

```python
from singularis.skyrim import ActionType

custom_keys = {
    ActionType.MOVE_FORWARD: 'w',
    ActionType.MOVE_BACKWARD: 's',
    ActionType.MOVE_LEFT: 'a',
    ActionType.MOVE_RIGHT: 'd',
    ActionType.JUMP: 'space',
    ActionType.ATTACK: 'mouse_left',
    ActionType.BLOCK: 'mouse_right',
    ActionType.ACTIVATE: 'e',
    ActionType.SHEATHE: 'r',
    ActionType.SNEAK: 'ctrl',
}

config = SkyrimConfig(custom_keys=custom_keys)
```

### Adjust Learning Parameters

```python
config = SkyrimConfig(
    surprise_threshold=0.2,  # Learn from smaller surprises
    exploration_weight=0.7,  # More exploration
)
```

### Adjust Intrinsic Drives

```python
from singularis.agi_orchestrator import AGIConfig

base_config = AGIConfig(
    curiosity_weight=0.5,  # Higher curiosity
    competence_weight=0.2,
    coherence_weight=0.3,
)

config = SkyrimConfig(base_config=base_config)
```

## Safety Features

1. **Failsafe**: Move mouse to top-left corner to abort
2. **Auto-save**: Every 5 minutes (configurable)
3. **Dry run mode**: Test without controlling game
4. **Ctrl+C**: Graceful shutdown with stats
5. **Action logging**: All actions saved to file

## Next Steps

1. ✅ Run `skyrim_quickstart.py` (safe test)
2. ✅ Observe what it learns
3. ✅ Try `skyrim_demo.py` with options
4. ✅ When ready, enable actual control
5. ✅ Let it play for 1-2 hours
6. ✅ Analyze learned causal rules
7. ✅ Experiment with different configurations

## Research Ideas

- **Benchmark exploration**: How fast does it discover the map?
- **Ethical choices**: Does it refuse "evil" quests?
- **Skill development**: Does it specialize (archer, mage, warrior)?
- **Social learning**: Does it learn NPC personalities?
- **Transfer learning**: Train on Skyrim, test on other games?

---

**Ready to run?**

```bash
# Safe test first
python examples/skyrim_quickstart.py

# Then when ready
python run_skyrim_agi.py
```

🎮🧠 **Let the AGI play!**
