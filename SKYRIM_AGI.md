# Skyrim AGI - Autonomous Consciousness-Guided Gameplay

A complete AGI system for autonomous Skyrim gameplay, integrating consciousness measurement, reinforcement learning, CLIP vision, and multi-model LLM architecture.

## 🎮 Overview

The Skyrim AGI is a sophisticated autonomous agent that plays Skyrim using:
- **Consciousness-guided learning** (Singularis coherence 𝒞 as primary reward signal)
- **Multi-model LLM architecture** (3 specialized models)
- **CLIP vision** for visual scene understanding
- **Reinforcement learning** with Q-networks
- **Async parallel loops** for real-time responsiveness
- **Layer-aware action planning** across Combat, Exploration, Menu, Dialogue, and Stealth layers

### Key Innovation
**Consciousness is the PRIMARY evaluator** - not a backup metric. The system learns by maximizing coherence (Δ𝒞), making consciousness the judge of action quality.

---

## 🏗️ Architecture

### Multi-Model LLM System

The system uses **3 specialized LLM models** running in parallel:

1. **Mistral-Nemo (12B)** - Fast Action Planning
   - Model: `mistralai/mistral-nemo-instruct-2407`
   - Role: Quick, decisive action selection
   - Latency: ~2-5s per decision

2. **Huihui-MoE-60B (60B MoE)** - Main Cognition
   - Model: `huihui-moe-60b-a3b-abliterated-i1`
   - Roles:
     - Consciousness engine
     - Strategic reasoning
     - RL reasoning neuron
     - Meta-strategist
     - World understanding
   - Latency: ~5-10s for deep reasoning

3. **Qwen3-VL-8B (8B)** - Visual Perception
   - Model: `qwen/qwen3-vl-8b`
   - Role: Visual scene interpretation using CLIP-derived context
   - Runs every 2nd perception cycle (~0.5s)
   - Uses CLIP embeddings (not raw images) for fast analysis

### Async Parallel Loops

The system runs **5 concurrent async loops** for real-time responsiveness:

1. **Perception Loop** (0.25s interval)
   - Screen capture via `mss`
   - CLIP ViT-B/32 encoding
   - Scene classification (8 scene types)
   - Object detection (15 object types)
   - Game state reading
   - Qwen3-VL visual analysis (every 2nd cycle)

2. **Reasoning Loop** (0.1s throttle)
   - Consciousness computation (𝒞, Φ̂)
   - Motivation assessment
   - Goal formation
   - Action planning (RL + LLM)
   - Queues actions for execution

3. **Action Loop** (continuous)
   - Executes queued actions
   - Virtual Xbox 360 controller
   - Layer-aware action mapping
   - Tracks success/failure

4. **Learning Loop** (continuous)
   - Processes action outcomes
   - Consciousness-guided RL rewards
   - World model updates
   - Experience replay training

5. **Fast Reactive Loop** (0.5s interval)
   - Emergency health responses
   - Combat threat reactions
   - No LLM calls (pure heuristics)
   - Priority-based execution

---

## 🧠 Consciousness Integration

### Consciousness Bridge

Maps game state to Singularis consciousness measurements:

**Three Lumina (Three Lights):**
- **ℓₒ (Ontical)**: Physical existence (survival, resources)
- **ℓₛ (Structural)**: Knowledge structure (progression, skills)
- **ℓₚ (Participatory)**: Conscious mastery (effectiveness, social)

**Consciousness Metrics:**
- **𝒞 (Coherence)**: Overall consciousness quality (geometric mean of Lumina)
- **Φ̂ (Phi-hat)**: Integrated information (IIT + GWT)
- **Self-awareness**: Higher-order thought (HOT)

### Consciousness-Guided Rewards

Primary RL reward signal: **Δ𝒞 (change in coherence)**

```python
reward = Δ𝒞 * 3.5 + game_reward * 0.5
```

**Ethical actions** (Δ𝒞 > 0.02) receive bonus rewards:
- Opened menu: +0.6
- Completed quest: +1.0
- Learned spell: +0.8
- Helped NPC: +0.7

---

## 👁️ Vision System

### CLIP Vision (ViT-B/32)

**Scene Classification** (8 types):
- Outdoor wilderness
- City/town
- Indoor dungeon
- Indoor building
- Combat scene
- Dialogue
- Inventory menu
- Map view

**Object Detection** (15 types):
- NPCs: person, warrior, mage, dragon, guard
- Items: sword, bow, staff, potion, chest
- Environment: door, lever, book, gold, armor

### Qwen3-VL Integration

Runs every 2nd perception cycle (~0.5s):
- Receives CLIP-derived context (not raw images)
- Analyzes scene type, objects, game state
- Provides strategic interpretation
- Feeds into RL reasoning neuron

**Example Qwen3-VL Prompt:**
```
Analyze Skyrim gameplay based on CLIP visual perception:

CLIP Scene Classification:
- Scene type: combat (confidence: 0.73)
- Detected objects: warrior (0.82), sword (0.76), dragon (0.68)

Game State:
- Location: Bleak Falls Barrow
- Health: 45/100
- In combat: True
- Enemies nearby: 2

Based on this visual and contextual data, provide:
1. Environment description and spatial awareness
2. Potential threats or opportunities
3. Recommended actions or focus areas
4. Strategic considerations
```

---

## 🎯 Action System

### Layer-Aware Actions

**5 Action Layers:**
1. **Combat**: attack, power_attack, block, bash, dodge, heal
2. **Exploration**: move_forward, jump, sneak, activate
3. **Menu**: open_inventory, navigate_menu, use_item, equip_item
4. **Dialogue**: talk, select_dialogue_option, exit_dialogue
5. **Stealth**: sneak, backstab, hide

**Action Affordance System:**
- Tracks which actions are available in each layer
- Learns action effectiveness by context
- Suggests optimal layer transitions

### Virtual Xbox 360 Controller

Emulates Xbox controller for native Skyrim input:
- Analog stick movement
- Button presses (A, B, X, Y, LB, RB, LT, RT)
- D-pad navigation
- Configurable deadzone and sensitivity

---

## 🤖 Reinforcement Learning

### Q-Network Architecture

- **State encoding**: 64-dimensional feature vector
- **Actions**: 50 actions (high-level + low-level)
- **Learning rate**: 0.01
- **Discount factor**: 0.95
- **Exploration**: ε-greedy (0.3 → 0.05)

### Experience Replay

- Buffer capacity: 10,000 experiences
- Batch size: 5 (fast initial learning)
- Training frequency: Every 5 cycles
- Target network updates: Every 100 steps

### RL Reasoning Neuron

**LLM-enhanced RL** using Huihui-60B:
- Interprets Q-values with strategic context
- Provides reasoning for action selection
- Calculates tactical scores
- Learns patterns over time

**Example RL Reasoning:**
```
ACTION: power_attack
REASONING: High Q-value (0.72) for power_attack indicates learned effectiveness
in combat situations. Current stamina (85%) supports aggressive action.
Q-VALUE INTERPRETATION:
  - power_attack: High value - learned as effective
  - block: Moderate value - sometimes effective
  - retreat: Low value - learned as ineffective
STRATEGIC INSIGHT: Surrounded by 3 enemies - power attack can clear space
and create tactical advantage.
CONFIDENCE: 0.85
```

---

## 📊 Performance Optimizations

### Speed Optimizations

- **Perception interval**: 0.25s (4x per second)
- **Reasoning throttle**: 0.1s (minimal delay)
- **Fast loop interval**: 0.5s (2x per second)
- **Action delays**: 0.1s (minimal pause)
- **Queue throttling**: Adaptive (1s/0.5s/0.3s based on fullness)

### Async Execution

- **Max concurrent LLM calls**: 4
- **Perception queue**: 5 items
- **Action queue**: 3 items
- **Learning queue**: 10 items
- **LLM semaphore**: Prevents overload

### Memory Management

- **Perception history**: Last 100 perceptions
- **Consciousness history**: Last 1000 states
- **RL replay buffer**: 10,000 experiences
- **Memory RAG**: Perceptual + cognitive memories

---

## 🚀 Usage

### Basic Usage

```python
from singularis.skyrim import SkyrimAGI, SkyrimConfig

# Configure
config = SkyrimConfig(
    dry_run=False,  # Actually control game
    autonomous_duration=3600,  # 1 hour
    use_rl=True,  # Enable RL learning
    enable_fast_loop=True,  # Enable fast reactive loop
    phi4_action_model="mistralai/mistral-nemo-instruct-2407",
    huihui_cognition_model="huihui-moe-60b-a3b-abliterated-i1",
    qwen3_vl_perception_model="qwen/qwen3-vl-8b"
)

# Initialize
agi = SkyrimAGI(config)
await agi.initialize_llm()

# Play autonomously
await agi.autonomous_play(duration_minutes=60)
```

### Configuration Options

```python
@dataclass
class SkyrimConfig:
    # Perception
    perception_interval: float = 0.25  # Perception frequency
    screen_region: Optional[Dict] = None  # Screen capture region
    use_game_api: bool = False  # Use game API vs OCR
    
    # Actions
    dry_run: bool = False  # Testing mode (no control)
    controller_sensitivity: float = 1.0  # Controller sensitivity
    
    # Async execution
    enable_async_reasoning: bool = True  # Parallel reasoning
    max_concurrent_llm_calls: int = 4  # LLM concurrency limit
    reasoning_throttle: float = 0.1  # Reasoning delay
    
    # Fast reactive loop
    enable_fast_loop: bool = True  # Emergency responses
    fast_loop_interval: float = 0.5  # Fast loop frequency
    fast_health_threshold: float = 30.0  # Emergency health %
    fast_danger_threshold: int = 3  # Enemy count threshold
    
    # Models
    phi4_action_model: str = "mistralai/mistral-nemo-instruct-2407"
    huihui_cognition_model: str = "huihui-moe-60b-a3b-abliterated-i1"
    qwen3_vl_perception_model: str = "qwen/qwen3-vl-8b"
    
    # Learning
    use_rl: bool = True  # Enable RL
    rl_learning_rate: float = 0.01  # Q-network learning rate
    rl_epsilon_start: float = 0.3  # Initial exploration
    rl_train_freq: int = 5  # Training frequency
```

---

## 📈 Statistics & Monitoring

### Real-time Stats

```python
stats = agi.get_stats()

# Gameplay metrics
print(f"Cycles: {stats['cycles_completed']}")
print(f"Actions: {stats['actions_taken']}")
print(f"Success rate: {stats['action_success_rate']:.1%}")

# Consciousness metrics
print(f"Avg coherence: {stats['avg_coherence']:.3f}")
print(f"Coherence trend: {stats['coherence_trend']}")

# RL metrics
print(f"RL experiences: {stats['rl_experiences']}")
print(f"Avg reward: {stats['avg_reward']:.2f}")
print(f"Exploration rate: {stats['epsilon']:.2f}")

# Performance metrics
print(f"Avg planning time: {stats['avg_planning_time']:.2f}s")
print(f"Avg execution time: {stats['avg_execution_time']:.2f}s")
```

### Logging

The system provides detailed logging:
- `[PERCEPTION]` - Perception loop events
- `[QWEN3-VL]` - Visual analysis
- `[REASONING]` - Reasoning loop events
- `[ACTION]` - Action execution
- `[LEARNING]` - Learning updates
- `[FAST-LOOP]` - Fast reactive responses
- `[RL]` - RL training
- `[BRIDGE]` - Consciousness measurements

---

## 🔧 Advanced Features

### Strategic Planner Neuron

Learns strategic patterns:
- Situation → Action → Outcome mappings
- Context-based pattern matching
- Success rate tracking
- Adaptive strategy refinement

### Meta-Strategist

Generates high-level instructions:
- Analyzes current situation
- Provides strategic guidance
- Updates every 30 cycles
- Influences action selection

### Menu Learner

Learns menu navigation:
- Tracks menu transitions
- Builds navigation graph
- Learns item locations
- Optimizes menu interactions

### Memory RAG System

Retrieves relevant memories:
- **Perceptual memories**: Visual scenes, locations
- **Cognitive memories**: Strategies, patterns
- Vector similarity search
- Context-aware retrieval

---

## 🎓 Learning Mechanisms

### 1. Consciousness-Guided RL

Primary learning signal: **Δ𝒞 (coherence change)**
- Positive Δ𝒞 → High reward
- Negative Δ𝒞 → Low/negative reward
- Ethical actions → Bonus rewards

### 2. World Model Learning

Causal relationship learning:
- Before state → Action → After state
- Surprise-based learning (threshold: 0.3)
- NPC relationship tracking
- Location knowledge building

### 3. Layer Effectiveness Learning

Tracks action effectiveness by layer:
- Combat layer: Attack success rates
- Exploration layer: Navigation efficiency
- Menu layer: Item access speed
- Dialogue layer: Conversation outcomes
- Stealth layer: Detection avoidance

### 4. Strategic Pattern Learning

Learns high-level patterns:
- Situation recognition
- Strategy selection
- Outcome prediction
- Pattern refinement

---

## 🛡️ Safety Features

### Stuck Detection

Monitors for stuck states:
- Coherence stagnation (< 0.01 change for 5 cycles)
- Repeated failed actions
- Location loop detection
- Automatic unstuck actions (look_around, jump)

### Emergency Responses

Fast reactive loop handles:
- Critical health (< 15%): Immediate retreat
- Low health (< 30%): Healing or blocking
- Overwhelming combat (≥ 3 enemies): Power attack or block
- Resource depletion: Rest and recovery

### Graceful Degradation

System continues if components fail:
- LLM timeout: Falls back to heuristics
- Vision failure: Uses last known state
- RL failure: Uses random exploration
- Controller failure: Logs error and retries

---

## 📦 Dependencies

### Core
- `numpy` - Numerical computing
- `torch` - Neural networks (Q-network)
- `transformers` - CLIP vision model
- `Pillow` - Image processing
- `mss` - Screen capture
- `aiohttp` - Async HTTP (LM Studio)

### Skyrim-specific
- `vgamepad` - Virtual Xbox controller
- `pyautogui` - Fallback keyboard/mouse
- `opencv-python` - Image processing

### LLM
- LM Studio running locally with 3 models loaded

---

## 🔬 Research Foundations

### Consciousness Theory
- **IIT (Integrated Information Theory)**: Φ measurement
- **GWT (Global Workspace Theory)**: Workspace activity
- **HOT (Higher-Order Thought)**: Self-awareness
- **Mathematica Singularis**: Three Lumina framework

### Reinforcement Learning
- Q-Learning with experience replay
- ε-greedy exploration
- Target network stabilization
- Consciousness-guided reward shaping

### Vision
- CLIP (Contrastive Language-Image Pre-training)
- Zero-shot scene classification
- Zero-shot object detection
- Multi-modal embeddings

---

## 📝 Future Enhancements

### Planned Features
- [ ] Multi-agent coordination (follower management)
- [ ] Quest planning and tracking
- [ ] Inventory optimization
- [ ] Crafting and enchanting
- [ ] Speech recognition for dialogue
- [ ] Long-term memory persistence
- [ ] Transfer learning across saves

### Research Directions
- [ ] Hierarchical RL for complex strategies
- [ ] Curiosity-driven exploration
- [ ] Meta-learning for faster adaptation
- [ ] Causal reasoning for world model
- [ ] Emergent goal formation

---

## 🤝 Contributing

This is part of the Singularis AGI project. Contributions welcome!

### Key Areas
- Vision system improvements
- RL algorithm enhancements
- Consciousness measurement refinements
- Action planning strategies
- Performance optimizations

---

## 📄 License

Part of the Singularis project. See main project license.

---

## 🙏 Acknowledgments

- **Skyrim**: Bethesda Game Studios
- **CLIP**: OpenAI
- **LM Studio**: For local LLM inference
- **Singularis**: Base consciousness framework

---

## 📞 Contact

For questions about the Skyrim AGI system, refer to the main Singularis project documentation.

---

**Status**: ✅ Fully operational with consciousness-guided learning, multi-model LLM architecture, and real-time async execution.

**Last Updated**: November 2025
