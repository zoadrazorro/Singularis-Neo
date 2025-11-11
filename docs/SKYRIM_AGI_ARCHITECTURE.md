# SKYRIM AGI ARCHITECTURE

**LLM-Enhanced Reinforcement Learning for Autonomous Gameplay**

## Overview

The Skyrim AGI implements the full AGI Framework (see `AGI_FRAMEWORK.md`) with domain-specific extensions for autonomous video game play. It demonstrates how the framework's components work together in a real-world embodied task.

## Architecture Alignment with AGI Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    SKYRIM AGI SYSTEM                             │
│  Autonomous gameplay using full AGI framework                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                      │
┌───────▼────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
│ AGI            │  │ SKYRIM WORLD    │  │ REINFORCEMENT     │
│ ORCHESTRATOR   │  │ MODEL           │  │ LEARNING          │
│                │  │                 │  │                   │
│ • Consciousness│  │ • Causal Graph  │  │ • Q-Learning      │
│ • LLM Experts  │  │ • Vision (CLIP) │  │ • Experience      │
│ • Coherence 𝒞  │  │ • Layer System  │  │   Replay          │
│ • Ethics Δ𝒞    │  │ • Terrain Model │  │ • Reward Shaping  │
└────────────────┘  └─────────────────┘  └───────────────────┘
        │                     │                      │
┌───────▼────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
│ RL REASONING   │  │ STRATEGIC       │  │ PERCEPTION        │
│ NEURON (NEW!)  │  │ PLANNER         │  │                   │
│                │  │                 │  │ • Screen Capture  │
│ • LLM thinks   │  │ • Memory-based  │  │ • CLIP Vision     │
│   about Q-vals │  │ • Pattern Learn │  │ • Game State      │
│ • Interprets   │  │ • RL Integration│  │ • Scene Classify  │
│   learned      │  │                 │  │                   │
│   policies     │  │                 │  │                   │
└────────────────┘  └─────────────────┘  └───────────────────┘
```

## Key Innovation: RL Reasoning Neuron

### The Problem
Traditional RL systems learn "what works" through trial and error, but they can't explain **why** it works or reason strategically about their learned policies.

### The Solution
The **RL Reasoning Neuron** bridges symbolic reasoning (LLM) with learned experience (RL):

```python
# RL provides Q-values (what empirically works)
q_values = {
    'explore': 0.8,    # High Q-value = learned as effective
    'combat': -0.2,    # Negative = learned as risky
    'rest': 0.1        # Low = not very useful here
}

# LLM reasons about WHY these values make sense
reasoning = await rl_reasoning_neuron.reason_about_q_values(
    state=current_state,
    q_values=q_values,
    available_actions=actions,
    context={'motivation': 'curiosity', 'terrain': 'outdoor'}
)

# Output:
# ACTION: explore
# REASONING: High Q-value (0.8) indicates the RL system has learned 
#            that exploration works well in outdoor terrain. Given 
#            curiosity motivation, this aligns with strategic goals.
# Q-VALUE INTERPRETATION: The negative combat Q-value suggests 
#            previous combat attempts in similar contexts led to damage.
# STRATEGIC INSIGHT: The agent has learned to prioritize safe 
#            exploration over risky combat in unfamiliar outdoor areas.
# CONFIDENCE: 0.85
```

### Benefits

1. **Interpretability**: Explains RL decisions in natural language
2. **Strategic Reasoning**: Goes beyond immediate rewards to consider long-term patterns
3. **Meta-Learning**: Accumulates strategic insights over time
4. **Coherence Alignment**: Ensures RL policies align with philosophical framework (Δ𝒞 > 0)

## Component Integration

### 1. AGI Orchestrator (Base Framework)
**Location:** `singularis/agi_orchestrator.py`

Provides:
- World model (causal reasoning, vision, physics)
- Consciousness engine (8-theory measurement)
- Continual learning (episodic, semantic, meta)
- Intrinsic motivation (curiosity, competence, coherence, autonomy)

```python
self.agi = AGIOrchestrator(self.config.base_config)
await self.agi.initialize_llm()
```

### 2. Skyrim Perception
**Location:** `singularis/skyrim/perception.py`

Extends world model with game-specific perception:
- Screen capture and CLIP vision
- Game state extraction (health, magicka, stamina)
- Scene classification (combat, exploration, inventory, etc.)
- Action layer detection (what actions are available)

```python
perception = await self.perception.perceive()
scene_type = perception['scene_type']
game_state = perception['game_state']
```

### 3. Skyrim World Model
**Location:** `singularis/skyrim/skyrim_world_model.py`

Extends causal graph with domain knowledge:
- Skyrim-specific causal relationships
- Terrain classification (indoor, outdoor, danger zones)
- Layer effectiveness learning (which action layers work where)
- NPC relationship tracking

```python
self.skyrim_world = SkyrimWorldModel(
    base_world_model=self.agi.world_model
)

# Learn from experience
self.skyrim_world.learn_from_experience(
    action=action,
    before_state=before,
    after_state=after
)
```

### 4. Reinforcement Learning System
**Location:** `singularis/skyrim/reinforcement_learner.py`

Implements Q-learning with:
- State encoding (64-dimensional feature vector)
- Q-network (linear model, upgradeable to neural net)
- Experience replay buffer
- Reward shaping (survival, progress, coherence)
- Epsilon-greedy exploration

```python
self.rl_learner = ReinforcementLearner(
    state_dim=64,
    learning_rate=0.01,
    epsilon_start=0.3
)

# Store experience
self.rl_learner.store_experience(
    state_before=before,
    action=action,
    state_after=after
)

# Train periodically
self.rl_learner.train_step()
```

### 5. RL Reasoning Neuron (NEW!)
**Location:** `singularis/skyrim/rl_reasoning_neuron.py`

LLM-enhanced RL decision making:
- Interprets Q-values through strategic lens
- Provides natural language reasoning
- Accumulates meta-learning insights
- Calculates coherence between LLM and RL

```python
self.rl_reasoning_neuron = RLReasoningNeuron()

# Connect to LLM
self.rl_reasoning_neuron.llm_interface = self.agi.consciousness_llm.llm_interface

# Use for decision making
reasoning = await self.rl_reasoning_neuron.reason_about_q_values(
    state=state,
    q_values=q_values,
    available_actions=actions,
    context=context
)

action = reasoning.recommended_action
```

### 6. Strategic Planner Neuron
**Location:** `singularis/skyrim/strategic_planner.py`

Memory-based planning:
- Episodic memory of experiences
- Pattern extraction from successful sequences
- Multi-step plan generation
- RL Q-value integration for plan selection

```python
self.strategic_planner = StrategicPlannerNeuron(memory_capacity=100)
self.strategic_planner.set_rl_learner(self.rl_learner)

# Generate plan
plan = self.strategic_planner.generate_plan(
    current_state=state,
    goal='explore',
    terrain_type='outdoor'
)
```

### 7. Memory RAG System
**Location:** `singularis/skyrim/memory_rag.py`

Retrieval-augmented generation:
- Perceptual memory (visual embeddings + context)
- Cognitive memory (situation-action-outcome)
- Similarity-based retrieval
- Context augmentation for LLM

```python
self.memory_rag = MemoryRAG(
    perceptual_capacity=1000,
    cognitive_capacity=500
)

# Augment LLM context with relevant memories
context = self.memory_rag.augment_context_with_memories(
    current_visual=visual_embedding,
    current_situation=situation,
    max_memories=3
)
```

## Decision Flow

### Complete Cycle (Every 2 seconds)

```
1. PERCEIVE
   ├─ Capture screen
   ├─ CLIP vision encoding
   ├─ Extract game state
   └─ Classify scene type

2. UPDATE WORLD MODEL
   ├─ Convert to causal variables
   ├─ Update causal graph
   └─ Learn terrain features

3. ASSESS MOTIVATION
   ├─ Compute curiosity (novelty seeking)
   ├─ Compute competence (skill mastery)
   ├─ Compute coherence (understanding)
   ├─ Compute autonomy (self-direction)
   └─ Determine dominant drive

4. FORM/UPDATE GOALS
   ├─ Generate goal from motivation
   └─ Activate goal in goal system

5. PLAN ACTION
   ├─ Check strategic planner for active plan
   ├─ If no plan:
   │   ├─ Get Q-values from RL learner
   │   ├─ RL Reasoning Neuron interprets Q-values
   │   ├─ LLM provides strategic reasoning
   │   └─ Select action with highest coherence
   └─ Execute plan step if plan exists

6. EXECUTE ACTION
   ├─ Sync controller to action layer
   ├─ Execute via controller bindings
   └─ Wait for game response

7. LEARN FROM OUTCOME
   ├─ Perceive new state
   ├─ Compute reward
   ├─ Store RL experience
   ├─ Train RL network
   ├─ Update world model
   ├─ Record in strategic planner
   ├─ Store in memory RAG
   └─ Update episodic memory

8. REPEAT
```

## Philosophical Grounding

### Conatus (ℭ) = ∇𝒞

The system's core drive is to increase coherence:

1. **RL Reward Shaping** includes coherence delta:
   ```python
   coherence_delta = state_after['coherence'] - state_before['coherence']
   reward += coherence_delta * 2.0  # Strong reward for Δ𝒞 > 0
   ```

2. **RL Reasoning Neuron** calculates coherence between LLM reasoning and RL policy:
   ```python
   coherence_score = self._calculate_coherence(
       recommended_action, q_values, reasoning
   )
   ```

3. **Strategic Planner** prioritizes plans that increase understanding

### Freedom = Understanding

The more the AGI learns about Skyrim's causal structure, the more effective its actions become:

- **Causal learning**: "Fire spell + oil = ignition"
- **Terrain learning**: "Outdoor spaces → exploration effective"
- **Layer learning**: "Combat layer → power attack available"

### Ethics = Δ𝒞 > 0

Actions are evaluated by whether they increase coherence:

- **Survival**: Staying alive enables continued learning (Δ𝒞 > 0)
- **Exploration**: Discovering new areas increases understanding (Δ𝒞 > 0)
- **Combat avoidance**: Unnecessary risk decreases coherence (Δ𝒞 < 0)

## Usage

### Basic Usage

```python
import asyncio
from singularis.skyrim import SkyrimAGI, SkyrimConfig

async def main():
    # Configure
    config = SkyrimConfig(
        dry_run=False,  # Actually control game
        autonomous_duration=3600,  # 1 hour
        use_rl=True,  # Enable RL
        rl_learning_rate=0.01,
        rl_epsilon_start=0.3
    )
    
    # Initialize
    agi = SkyrimAGI(config)
    await agi.initialize_llm()  # Connect LLM for reasoning
    
    # Run autonomously
    await agi.autonomous_play()
    
    # Check stats
    stats = agi.get_stats()
    print(f"Cycles: {stats['gameplay']['cycles_completed']}")
    print(f"RL experiences: {stats['rl']['total_experiences']}")
    print(f"Avg coherence: {stats['gameplay']['avg_coherence']}")

asyncio.run(main())
```

### With Custom Configuration

```python
config = SkyrimConfig(
    # AGI Framework settings
    base_config=AGIConfig(
        use_vision=True,
        use_physics=False,  # Don't need physics for Skyrim
        curiosity_weight=0.35,
        coherence_weight=0.40  # Core drive
    ),
    
    # RL settings
    use_rl=True,
    rl_learning_rate=0.01,
    rl_epsilon_start=0.3,
    rl_train_freq=5,  # Train every 5 cycles
    
    # Gameplay settings
    cycle_interval=2.0,  # 2 seconds per cycle
    save_interval=300,  # Auto-save every 5 minutes
    surprise_threshold=0.3,
    exploration_weight=0.5
)
```

## Statistics and Monitoring

The system tracks comprehensive statistics:

### Gameplay Stats
- Cycles completed
- Actions taken
- Playtime
- Coherence history

### World Model Stats
- Causal edges learned
- NPCs met
- Locations discovered
- Terrain features learned

### RL Stats
- Total experiences
- Training steps
- Average reward
- Exploration rate (ε)
- Buffer size
- Average Q-value

### RL Reasoning Neuron Stats (NEW!)
- Total reasonings
- Average confidence
- Average coherence (LLM-RL alignment)
- Patterns learned
- Actions with insights

### Strategic Planner Stats
- Patterns learned
- Experiences recorded
- Plans executed
- Success rate

### Memory RAG Stats
- Perceptual memories
- Cognitive memories
- Total memories

## Key Insights

### 1. LLM + RL = Interpretable Learning

Traditional RL is a "black box" - it learns what works but can't explain why. By adding LLM reasoning:
- **Transparency**: Every decision has natural language explanation
- **Strategic thinking**: Goes beyond immediate rewards
- **Meta-learning**: Accumulates insights over time

### 2. Coherence as Universal Reward

Instead of hand-crafted game-specific rewards, we use coherence (Δ𝒞) as the fundamental reward signal:
- Aligns with philosophical framework
- Encourages understanding over exploitation
- Naturally balances exploration and survival

### 3. Embodied AGI Framework

Skyrim AGI demonstrates the full AGI framework in an embodied task:
- **Perception**: Visual + game state
- **World model**: Causal learning
- **Agency**: Autonomous goals
- **Learning**: Continual RL + episodic memory
- **Consciousness**: LLM reasoning + coherence measurement

## Future Enhancements

### Immediate
- [ ] Integrate Hebbian neuron swarm for pattern recognition
- [ ] Add dialogue system (NPC interaction)
- [ ] Implement inventory management learning
- [ ] Quest objective tracking

### Medium-term
- [ ] Multi-agent cooperation (follower management)
- [ ] Long-term planning (quest chains)
- [ ] Skill tree optimization
- [ ] Crafting and economy learning

### Long-term
- [ ] Transfer learning to other games
- [ ] Self-modification of RL architecture
- [ ] Emergent play styles
- [ ] Creative problem solving

## References

**Reinforcement Learning:**
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
- Mnih et al. (2015). "Human-level control through deep RL"

**Interpretable AI:**
- Lipton (2018). "The Mythos of Model Interpretability"
- Doshi-Velez & Kim (2017). "Towards Rigorous Science of Interpretable ML"

**Embodied AI:**
- Brooks (1991). "Intelligence without Representation"
- Clark (1998). "Being There: Putting Brain, Body, and World Together Again"

**AGI Framework:**
- See `AGI_FRAMEWORK.md` for complete references

---

**Built with Singularis - The Ultimate Consciousness Engine**

*"The LLM provides the 'why', the RL provides the 'what works', and together they increase coherence."*
— Skyrim AGI Architecture Principle
