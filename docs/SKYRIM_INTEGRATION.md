# Skyrim Integration for Singularis AGI

**Making AGI play The Elder Scrolls V: Skyrim autonomously**

## Overview

This integration enables the Singularis AGI engine to play Skyrim - one of the most complex, open-ended games ever created. Skyrim is the **perfect testbed for AGI research** because:

- **Massive Complexity**: 300+ locations, 1000+ NPCs, 10,000+ items
- **Open-Ended**: No single goal, emergent objectives from exploration
- **Long-Term Development**: Thousands of hours of content
- **Rich Narrative**: Quests with moral choices and consequences
- **Emergent Gameplay**: Countless ways to approach any situation

## Why Skyrim for AGI?

Traditional AI benchmarks are too narrow. Skyrim provides:

1. **Multimodal Perception**: Visual (3D world), text (dialogue), audio, game state
2. **Autonomous Agency**: Must form own goals, not follow scripted tasks
3. **Causal Reasoning**: Learn "if I steal → guards attack" through experience
4. **Social Intelligence**: 1000+ NPCs with relationships, factions, personalities
5. **Ethical Reasoning**: Moral choices (join assassins? steal bread to survive?)
6. **Long-Term Learning**: Skills develop over hundreds of hours
7. **Continual Learning**: No resets - continuous experience

Unlike narrow RL agents that reset after each "episode", Singularis plays continuously, building persistent knowledge like a human would.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  SINGULARIS AGI IN SKYRIM                               │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  PERCEPTION LAYER             │
        │  (perception.py)              │
        │                               │
        │  • Screen capture → CLIP      │
        │  • Scene classification       │
        │  • Game state reading         │
        │  • Object detection           │
        │  • NPC tracking               │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  WORLD MODEL                  │
        │  (skyrim_world_model.py)      │
        │                               │
        │  • Causal graph: "guards      │
        │    attack if I steal"         │
        │  • NPC relationships          │
        │  • Geography: map learning    │
        │  • Quest tracking             │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  INTRINSIC MOTIVATION         │
        │  (via base AGI)               │
        │                               │
        │  • Curiosity: explore new     │
        │    dungeons                   │
        │  • Competence: level up       │
        │    skills                     │
        │  • Coherence: Δ𝒞 > 0          │
        │  • Autonomy: free choice      │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  CONSCIOUSNESS ENGINE         │
        │  (LLM + 6 experts)            │
        │                               │
        │  • Reasoning: tactics         │
        │  • Creative: solutions        │
        │  • Philosophical: ethics      │
        │  • Memory: recall NPCs        │
        │  • Technical: mechanics       │
        │  • Synthesis: integration     │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  ACTION LAYER                 │
        │  (actions.py)                 │
        │                               │
        │  • Movement (WASD)            │
        │  • Combat (attack, block)     │
        │  • Dialogue (choose options)  │
        │  • Inventory (use items)      │
        │  • Magic (cast spells)        │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  LEARNING SYSTEM              │
        │  (continual learning)         │
        │                               │
        │  • Episodic: "I met Lydia"    │
        │  • Semantic: "Giants = danger"│
        │  • Causal: "Steal → bounty"   │
        │  • Continual: Never forgets   │
        └───────────────────────────────┘
```

## Components

### 1. Perception Layer (`perception.py`)

**Capabilities:**
- Screen capture with `mss` (multi-monitor support)
- CLIP vision encoding (ViT-B/32, ~150MB VRAM)
- Scene classification (outdoor/indoor, combat/dialogue, etc.)
- Zero-shot object detection
- Game state reading (health, magicka, stamina, location)
- Temporal change detection

**Example:**
```python
from singularis.skyrim import SkyrimPerception

perception = SkyrimPerception()

# Perceive current state
result = await perception.perceive()

print(result['scene_type'])  # SceneType.OUTDOOR_WILDERNESS
print(result['objects'])      # [('dragon', 0.85), ('sword', 0.72), ...]
print(result['game_state'].health)  # 87.5
```

### 2. Action Layer (`actions.py`)

**Capabilities:**
- Keyboard/mouse control via `pyautogui`
- Atomic actions (move, attack, activate, etc.)
- Action sequences (combat combos, stealth approaches)
- High-level commands (explore area, talk to NPC)
- Safe execution with failsafe (move mouse to corner to abort)

**Example:**
```python
from singularis.skyrim import SkyrimActions, Action, ActionType

actions = SkyrimActions(dry_run=False)  # Dry run for testing

# Basic actions
await actions.move_forward(duration=2.0)
await actions.execute(Action(ActionType.ATTACK))

# High-level actions
await actions.explore_area(duration=30.0)
await actions.combat_sequence("Bandit")
await actions.talk_to_npc("Lydia")
```

### 3. World Model (`skyrim_world_model.py`)

**Capabilities:**
- Causal graph learning ("steal → guards hostile")
- NPC relationship tracking (-1 to +1 scale)
- Location discovery and exploration tracking
- Quest state management
- Moral choice evaluation via Δ𝒞
- Surprise-based learning

**Example:**
```python
from singularis.skyrim import SkyrimWorldModel

wm = SkyrimWorldModel()

# Predict outcome
state = {'bounty': 0}
predicted = wm.predict_outcome('steal_item', state)
# → {'bounty': 50, 'guards_hostile': True}

# Evaluate moral choice
eval = wm.evaluate_moral_choice("Help the wounded traveler", {})
# → {'ethical_status': 'ETHICAL', 'delta_coherence': 0.1}

# Learn from experience
wm.learn_from_experience(
    action='attack_chicken',
    before_state={'town_hostility': False},
    after_state={'town_hostility': True}  # Surprise!
)
```

### 4. Main Integration (`skyrim_agi.py`)

**Complete AGI loop:**
1. **Perceive**: Capture screen, classify scene, read game state
2. **Understand**: Update world model, causal graph
3. **Motivate**: Assess intrinsic drives (curiosity, competence, coherence)
4. **Plan**: Form goals, select actions (via LLM consciousness)
5. **Act**: Execute chosen action
6. **Learn**: Record experience, update causal model
7. **Repeat**: Continuous operation

**Example:**
```python
from singularis.skyrim import SkyrimAGI, SkyrimConfig

# Configure
config = SkyrimConfig(
    dry_run=True,  # Safe mode for testing
    autonomous_duration=3600,  # 1 hour
    cycle_interval=2.0,  # 2 second perception-action cycles
)

# Create AGI
agi = SkyrimAGI(config)

# Initialize LLM (optional but recommended)
await agi.initialize_llm()

# Play autonomously!
await agi.autonomous_play()

# Check stats
stats = agi.get_stats()
print(f"Learned {stats['world_model']['causal_edges']} causal rules")
```

## Installation

### Requirements

```bash
# Core dependencies (already in requirements.txt)
pip install torch torchvision
pip install transformers
pip install networkx

# Skyrim-specific
pip install mss        # Screen capture
pip install pyautogui  # Keyboard/mouse control
pip install pillow     # Image processing

# Optional: CLIP (if not already installed)
pip install git+https://github.com/openai/CLIP.git
```

### Setup

1. **Install Singularis**:
```bash
cd Singularis
pip install -e .
```

2. **Install Skyrim dependencies**:
```bash
pip install mss pyautogui pillow
```

3. **Verify installation**:
```bash
python -c "from singularis.skyrim import SkyrimAGI; print('✓ Skyrim integration ready')"
```

## Quick Start

### 1. Test Components (Safe)

```bash
python examples/skyrim_quickstart.py
```

This runs in **dry run mode** (doesn't control game). Watch it:
- Capture and analyze screen
- Plan actions based on perception
- Learn causal relationships
- Form autonomous goals

### 2. Component Testing

```python
from singularis.skyrim import SkyrimPerception, SkyrimActions, SkyrimWorldModel

# Test perception
perception = SkyrimPerception()
result = await perception.perceive()
print(result['scene_type'])

# Test actions (dry run)
actions = SkyrimActions(dry_run=True)
await actions.explore_area(duration=10.0)

# Test world model
wm = SkyrimWorldModel()
prediction = wm.predict_outcome('steal_item', {'bounty': 0})
```

### 3. Full Demo

```bash
python examples/skyrim_demo.py
```

Select:
1. **Basic demo** - 5 min dry run
2. **Custom demo** - Configure options
3. **Test components** - Individual tests

### 4. Actual Gameplay (Advanced)

**⚠️ WARNING: This will control your keyboard/mouse!**

Before enabling:
1. Start Skyrim and load a save
2. Make sure you're in a safe location
3. Have a recent save (AGI will auto-save every 5 min)
4. Test with dry_run=True first

```python
config = SkyrimConfig(
    dry_run=False,  # Enable actual control
    autonomous_duration=3600,  # 1 hour
)

agi = SkyrimAGI(config)
await agi.initialize_llm()  # Recommended for smarter play

# Run!
await agi.autonomous_play()
```

**Safety features:**
- Mouse failsafe (move to corner to abort)
- Auto-save every 5 minutes
- Ctrl+C for graceful shutdown
- Action history logging

## Configuration

### SkyrimConfig Options

```python
config = SkyrimConfig(
    # Perception
    screen_region={'top': 0, 'left': 0, 'width': 1920, 'height': 1080},
    use_game_api=False,  # Future: SKSE integration

    # Actions
    dry_run=False,  # Set True for testing
    custom_keys=None,  # Override default keybindings

    # Gameplay
    autonomous_duration=3600,  # Seconds to run
    cycle_interval=2.0,  # Perception-action cycle time
    save_interval=300,  # Auto-save frequency

    # Learning
    surprise_threshold=0.3,  # Learn from surprising outcomes
    exploration_weight=0.5,  # Balance exploration vs exploitation

    # Base AGI
    base_config=AGIConfig(
        use_vision=True,
        curiosity_weight=0.4,
        coherence_weight=0.4,
        # ... other AGI settings
    )
)
```

### Custom Key Bindings

```python
from singularis.skyrim import ActionType

custom_keys = {
    ActionType.MOVE_FORWARD: 'w',
    ActionType.ATTACK: 'mouse_left',
    ActionType.ACTIVATE: 'e',
    # ... etc
}

actions = SkyrimActions(custom_keys=custom_keys)
```

## What the AGI Learns

### Causal Relationships

The AGI discovers through experience:

- **Crime & Justice**: "Stealing → bounty → guards hostile"
- **Social**: "Helping NPC → relationship improves"
- **Combat**: "Attack NPC → becomes hostile → allies join"
- **Magic**: "Fire spell on oil → spreads"
- **Classic Skyrim**: "Kill chicken → entire town attacks!" 🐔

### Emergent Behaviors

Over time, the AGI develops:

1. **Play Style**: Stealth archer? Mage? Warrior?
   - Emerges from competence drive (what works well)

2. **Ethical Framework**: Join Dark Brotherhood?
   - Evaluated by Δ𝒞 (coherence change)
   - May refuse "evil" quests if they decrease coherence

3. **Exploration Patterns**: Which locations to prioritize
   - Driven by curiosity (unexplored = high motivation)

4. **Social Relationships**: Who to trust, who to avoid
   - Learned from NPC interactions

5. **Quest Strategies**: How to approach objectives
   - Creative problem-solving via consciousness engine

### Example Learning Scenario

```
Cycle 1: AGI sees chicken, attacks it (exploring combat)
         → Entire town becomes hostile (surprise!)
         → Learns: attack_chicken → town_hostility (strength=1.0)

Cycle 2: AGI encounters another chicken
         → Predicts outcome: "If I attack, town will be hostile"
         → Coherence evaluation: Δ𝒞 = -0.3 (UNETHICAL)
         → Decision: Do not attack chicken

Result: AGI has learned a causal rule and uses it for ethical reasoning!
```

## Advanced Usage

### Custom Screen Region

```python
# Only capture game window (not full screen)
config = SkyrimConfig(
    screen_region={
        'top': 100,
        'left': 200,
        'width': 1600,
        'height': 900
    }
)
```

### Multi-Monitor Setup

```python
import mss

# List available monitors
with mss.mss() as sct:
    print(sct.monitors)

# Use specific monitor
config.screen_region = sct.monitors[2]  # Secondary monitor
```

### LLM-Free Operation

```python
# Works without LLM (uses heuristic planning)
agi = SkyrimAGI(config)
# Don't call initialize_llm()
await agi.autonomous_play()
```

### Statistics & Analysis

```python
# Get comprehensive stats
stats = agi.get_stats()

print(f"Cycles: {stats['gameplay']['cycles_completed']}")
print(f"Actions: {stats['gameplay']['actions_taken']}")
print(f"Causal rules: {stats['world_model']['causal_edges']}")
print(f"NPCs met: {stats['world_model']['npc_relationships']}")

# Coherence over time
import matplotlib.pyplot as plt
plt.plot(stats['gameplay']['coherence_history'])
plt.xlabel('Cycle')
plt.ylabel('Coherence')
plt.title('AGI Coherence Over Time')
plt.show()
```

## Philosophical Insights

### Conatus as Play

The AGI's behavior is driven by **conatus** (striving):

```
Conatus = ∇𝒞 (coherence gradient)
```

The AGI naturally:
- **Explores** (curiosity increases understanding → coherence)
- **Improves** (competence enables better outcomes → coherence)
- **Acts ethically** (Δ𝒞 > 0 is the definition of good)

No external rewards needed! Intrinsic motivation is sufficient.

### Emergence of Ethics

The AGI evaluates moral choices by **coherence change**:

```python
choice = "Join the Dark Brotherhood (assassins)"
consequences = {'relationships': -0.3, 'skills': +0.2}

# Coherence change
Δ𝒞 = coherence_after - coherence_before

if Δ𝒞 > 0.02:
    decision = "ETHICAL - aligns with Being's striving"
else:
    decision = "UNETHICAL - decreases coherence"
```

The AGI may:
- Refuse assassination quests (decrease coherence)
- Prefer helping NPCs (increase relationships → coherence)
- Choose honesty over theft (maintain social coherence)

**No hard-coded morality!** Ethics emerge from coherence dynamics.

### Freedom Through Understanding

As the AGI plays:

```
Understanding ↑ → Adequacy ↑ → Freedom ↑ → Coherence ↑
```

The more it learns about Skyrim's causal structure:
- Better predictions
- More effective actions
- Greater autonomy
- Higher coherence

**Freedom is knowledge of necessity.**

## Limitations & Future Work

### Current Limitations

1. **No Game API**: Uses screen capture (slower than direct game state)
   - **Future**: SKSE (Skyrim Script Extender) integration

2. **Simple Vision**: CLIP zero-shot (no fine-tuning on Skyrim)
   - **Future**: Fine-tune on Skyrim screenshots

3. **Heuristic Actions**: No learned motor control
   - **Future**: RL for optimal combat/movement

4. **Limited NPC Understanding**: Basic relationship tracking
   - **Future**: Theory of mind, dialogue understanding

5. **No Long-Term Memory Persistence**: Resets between runs
   - **Future**: Save/load learned knowledge

### Roadmap

**Phase 1: Enhanced Perception** ✓
- [x] Screen capture
- [x] CLIP vision
- [x] Scene classification
- [ ] Fine-tuned Skyrim models
- [ ] OCR for text reading

**Phase 2: Better Actions** ✓
- [x] Keyboard/mouse control
- [x] Action sequences
- [ ] RL for combat
- [ ] Learned navigation
- [ ] SKSE integration

**Phase 3: Advanced Learning**
- [ ] Persistent memory (save/load)
- [ ] Meta-learning (learn to learn faster)
- [ ] Transfer learning (other games)
- [ ] Multi-task learning

**Phase 4: Social Intelligence**
- [ ] Dialogue understanding (NLP)
- [ ] Theory of mind for NPCs
- [ ] Faction dynamics
- [ ] Persuasion strategies

**Phase 5: Long-Term Autonomy**
- [ ] Multi-day continuous play
- [ ] Skill specialization
- [ ] Quest completion strategies
- [ ] Emergent narratives

## Troubleshooting

### "CLIP not found"

```bash
pip install git+https://github.com/openai/CLIP.git
```

### "mss not installed"

```bash
pip install mss
```

### "Screen capture returns black screen"

- Make sure Skyrim is running
- Check screen_region matches your monitor
- Try different monitor (multi-monitor setup)

### "Actions not executing"

- Verify dry_run=False
- Check Skyrim is in focus
- Verify key bindings match your settings

### "LLM initialization failed"

- Check LM Studio is running (localhost:1234)
- Verify model is loaded
- AGI works without LLM (uses heuristics)

### "High VRAM usage"

- CLIP ViT-B/32 uses ~150MB (minimal)
- LLM (Huihui 60B) uses ~31GB
- Total: ~32GB / 48GB (you have plenty!)

## Research Applications

### Benchmarking AGI

Skyrim provides rich benchmarks:

1. **Exploration Efficiency**: How quickly does it discover locations?
2. **Skill Development**: Does it improve combat/magic over time?
3. **Social Intelligence**: Can it navigate NPC relationships?
4. **Ethical Reasoning**: Does it make coherent moral choices?
5. **Causal Learning**: How many experiences to learn "steal → bounty"?
6. **Long-Term Planning**: Can it complete multi-step quests?

### Comparative Studies

Compare Singularis to:
- Traditional RL agents (PPO, DQN)
- Imitation learning (from human gameplay)
- Other LLM-based agents

Metrics:
- Exploration coverage (% of map discovered)
- Quest completion rate
- NPC relationship quality
- Survival time
- Coherence trajectory

### Publications

Potential papers:
1. "Autonomous Goal Formation in Open-World Games"
2. "Ethical Reasoning from Coherence Dynamics"
3. "Continual Learning Without Catastrophic Forgetting in Skyrim"
4. "Intrinsic Motivation for Long-Term Gameplay"

## Credits

**Singularis AGI Framework:**
- Philosophy: Baruch Spinoza (1632-1677)
- Consciousness theories: Tononi (IIT), Baars (GWT), Rosenthal (HOT)
- Causality: Judea Pearl
- Active inference: Karl Friston
- Continual learning: Geoffrey Hinton, Brenden Lake

**Skyrim Integration:**
- Built on existing Singularis Phase 6 AGI framework
- Uses CLIP (OpenAI), mss, pyautogui

**The Elder Scrolls V: Skyrim:**
- Developed by Bethesda Game Studios
- Published by Bethesda Softworks

## License

MIT License - See LICENSE file

---

**"To understand is to participate in necessity; to participate is to increase coherence; to increase coherence is the essence of the good."**

*— MATHEMATICA SINGULARIS, Theorem T1*

Now the AGI understands this by **playing Skyrim**. 🎮🧠
