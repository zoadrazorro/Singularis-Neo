# Reinforcement Learning System for Skyrim AGI

## Overview

This document describes the new **Reinforcement Learning (RL) system** that enables the Skyrim AGI to **genuinely learn** from experience, solving the core issue where the system only recorded experiences but didn't actually improve its behavior.

## Problem Statement

### Before: Recording Without Learning

The original system had several learning components that **appeared** to learn but actually didn't:

1. **Strategic Planner**: Counted action sequences but didn't optimize policy
2. **Movement Loops**: Random or forward-biased, no adaptation based on outcomes
3. **World Model**: Tracked effectiveness but didn't use it for optimization
4. **Continual Learner**: Stored episodic memories but never trained on them

**Key Issue**: The AGI recorded experiences in memory buffers but had **no gradient-based learning**, **no policy optimization**, and **no reward-driven improvement**.

### After: Genuine Machine Learning

The new system implements **proper reinforcement learning** with:

- ✅ **Q-Learning** with action-value function approximation
- ✅ **Experience Replay** for stable learning from past experiences
- ✅ **Reward Shaping** to guide learning toward useful behaviors
- ✅ **Epsilon-Greedy Exploration** that decays over time
- ✅ **Online Training** during gameplay
- ✅ **Policy Updates** via gradient descent on the Bellman equation

## Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│           ReinforcementLearner                      │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │ StateEncoder │  │   Q-Network  │  │  Replay  ││
│  │              │  │              │  │  Buffer  ││
│  │ Encodes game│→ │ Predicts Q() │← │ Stores   ││
│  │ state to    │  │ for actions  │  │ (s,a,r,s')││
│  │ features    │  │              │  │          ││
│  └──────────────┘  └──────────────┘  └──────────┘│
│                           ↓                        │
│                    Training Loop                   │
│                  (Bellman Updates)                 │
└─────────────────────────────────────────────────────┘
```

### 1. StateEncoder

Converts game state dictionaries into fixed-size feature vectors:

```python
Features (64-dim vector):
  [0-2]:   Health, Magicka, Stamina (normalized)
  [3-4]:   In Combat, Enemy Count
  [5-11]:  Scene Type (one-hot: exploration, combat, inventory, etc.)
  [12-16]: Action Layer (one-hot: Exploration, Combat, Menu, etc.)
  [20-29]: Location Hash
  [30-35]: Effectiveness, Surprise, Motivation signals
```

### 2. Q-Network

Learns action-value function **Q(s, a)** = expected cumulative reward for taking action `a` in state `s`.

**Implementation**: Linear function approximation
- Weights matrix: `[n_actions × state_dim]`
- Predicts Q-values: `Q(s) = W @ s + b`
- Updated via gradient descent on TD error

**Bellman Equation**:
```
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```

Where:
- `r` = immediate reward
- `γ` = discount factor (0.95)
- `α` = learning rate (0.01)

### 3. Experience Replay Buffer

Stores experiences `(state, action, reward, next_state, done)` in a circular buffer (capacity: 10,000).

**Why Replay?**
- Breaks correlation between consecutive experiences
- Enables stable learning from diverse past experiences
- Prevents catastrophic forgetting

### 4. Target Network

Separate copy of Q-network for stable learning:
- Updated every 100 training steps
- Prevents moving target problem in TD learning
- Improves convergence stability

## Reward Shaping

Rewards guide what the AGI learns. Carefully designed reward function:

### Reward Components

| Component | Reward | Purpose |
|-----------|--------|---------|
| **Survival** | -1.0 for damage >20, -0.3 for any damage, +0.5 for healing | Stay alive |
| **Death Penalty** | -10.0 | Strongly discourage dying |
| **Progress** | +0.5 for scene changes | Encourage exploration |
| **Combat Efficiency** | +0.2 for combat actions in combat | Use appropriate actions |
| **Exploration** | +0.3 for explore/navigate when safe | Prioritize exploration |
| **Efficiency** | -0.2 for resting at high health | Don't waste time |
| **Coherence** | +2.0 * Δ𝒞 | Maximize understanding |
| **Success** | +0.3 if action succeeded | Reward effective actions |
| **Stuck Penalty** | -0.5 | Avoid getting stuck |
| **Base** | +0.1 | Small reward for surviving |

### Example Reward Calculation

```python
State Before: health=100, in_combat=False, scene='exploration'
Action: 'explore'
State After:  health=100, in_combat=False, scene='outdoor'

Reward Breakdown:
  Survival: 0.0 (no damage)
  Progress: +0.5 (scene changed)
  Exploration: +0.3 (safe exploration)
  Base: +0.1
  Total: +0.9 ✓
```

## Learning Process

### 1. Action Selection (Epsilon-Greedy)

```python
if random() < ε:
    action = random_action()  # Explore
else:
    action = argmax(Q(s, a))  # Exploit
```

- Initial ε = 0.3 (30% exploration)
- Decays by 0.995 each step
- Minimum ε = 0.05 (always explore 5%)

### 2. Experience Storage

After each action:
```python
experience = (state_before, action, reward, state_after, done)
replay_buffer.add(experience)
```

### 3. Training Step

Every 5 cycles:
```python
batch = replay_buffer.sample(32)

for (s, a, r, s', done) in batch:
    # Compute target Q-value
    target = r + γ * max(Q_target(s'))

    # Update Q-network
    Q(s, a) ← Q(s, a) + α * (target - Q(s, a))
```

### 4. Policy Improvement

As training progresses:
- Q-values converge to true action values
- Policy becomes increasingly optimal
- Exploration decreases, exploitation increases
- AGI learns from mistakes and successes

## Integration with Existing Systems

### Strategic Planner Enhancement

Strategic planner now uses RL Q-values to select better action sequences:

```python
# Score patterns combining historical success with RL Q-values
combined_score = 0.6 * pattern_score + 0.4 * rl_score
```

Benefits:
- Pattern-based planning **+** Value-based planning
- Learned sequences validated by Q-values
- Better generalization to new situations

### Main Loop Integration

```python
# Perception-Action-Learning Cycle
for cycle in range(max_cycles):
    # 1. Perceive
    state = perceive()

    # 2. Select Action (RL)
    action = rl_learner.select_action(state)

    # 3. Execute
    execute(action)

    # 4. Observe Outcome
    next_state = perceive()

    # 5. Store Experience
    reward = compute_reward(state, action, next_state)
    rl_learner.store_experience(state, action, next_state, reward)

    # 6. Train (every 5 cycles)
    if cycle % 5 == 0:
        rl_learner.train_step()

    # 7. Save Model (every 50 cycles)
    if cycle % 50 == 0:
        rl_learner.save('skyrim_rl_model.pkl')
```

## Key Improvements

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Learning** | None (just recording) | Q-learning with gradient descent |
| **Action Selection** | Random/heuristic | Epsilon-greedy exploration-exploitation |
| **Memory** | Stored but unused | Experience replay for training |
| **Improvement** | None | Policy improves over time |
| **Adaptation** | Fixed behavior | Learns from mistakes |
| **Generalization** | None | Transfers learning to similar states |

### Quantifiable Benefits

1. **Sample Efficiency**: Learns from every experience via replay
2. **Stable Learning**: Target network prevents oscillation
3. **Continuous Improvement**: Q-values converge to optimal policy
4. **Reward-Driven**: Behavior shaped by well-designed rewards
5. **Adaptive**: Balances exploration and exploitation dynamically

## Usage

### Configuration

```python
config = SkyrimConfig(
    use_rl=True,  # Enable RL system
    rl_learning_rate=0.01,
    rl_epsilon_start=0.3,
    rl_train_freq=5  # Train every 5 cycles
)

agi = SkyrimAGI(config)
```

### Model Persistence

Models are automatically saved:
- Every 50 cycles during play
- To file: `skyrim_rl_model.pkl`
- Loaded on startup if available

### Statistics

Monitor learning progress:

```python
rl_stats = agi.rl_learner.get_stats()

# Key Metrics:
- total_experiences: Number of (s,a,r,s') tuples stored
- training_steps: Number of gradient updates performed
- avg_reward: Average reward per experience
- epsilon: Current exploration rate
- buffer_size: Number of experiences in replay buffer
- avg_q_value: Average Q-value (indicates learning progress)
```

## Machine Learning Principles Applied

### 1. Supervised Learning Components
- **Target**: Bellman target `r + γ max Q(s', a')`
- **Prediction**: Current Q(s, a)
- **Loss**: Squared TD error `(target - prediction)²`
- **Optimization**: Gradient descent

### 2. Exploration-Exploitation Trade-off
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- **Solution**: Epsilon-greedy with decay

### 3. Bootstrapping
- Use current Q-estimates to improve Q-estimates
- Bellman equation provides recursive relationship
- Converges to optimal Q* with sufficient exploration

### 4. Function Approximation
- Learn Q-function for large state space
- Generalize to unseen states
- Linear model: simple but effective

### 5. Off-Policy Learning
- Learn from past experiences (replay buffer)
- Decouple behavior policy (epsilon-greedy) from target policy (greedy)
- More sample efficient

## Future Enhancements

Potential upgrades:

1. **Deep Q-Network (DQN)**: Replace linear model with neural network
2. **Prioritized Experience Replay**: Sample important experiences more often
3. **Double DQN**: Reduce Q-value overestimation
4. **Dueling DQN**: Separate state-value and advantage functions
5. **Policy Gradient Methods**: Learn policy directly (PPO, A3C)
6. **Intrinsic Motivation**: Curiosity-driven exploration bonuses
7. **Hierarchical RL**: Learn high-level and low-level policies
8. **Multi-Agent RL**: Coordinate with NPCs

## Theoretical Foundation

### Markov Decision Process (MDP)

Skyrim gameplay modeled as MDP:
- **States** (S): Game states (health, location, scene, etc.)
- **Actions** (A): Available actions (explore, combat, rest, etc.)
- **Transition** (P): `P(s' | s, a)` = probability of next state
- **Reward** (R): `R(s, a, s')` = immediate reward
- **Discount** (γ): 0.95 (value future rewards at 95%)

### Optimal Policy

Goal: Find policy `π*` that maximizes expected return:

```
π* = argmax_π E[Σ γ^t * r_t]
```

Q-learning finds `π*` by learning optimal Q-function `Q*`:

```
π*(s) = argmax_a Q*(s, a)
```

### Convergence Guarantees

Under certain conditions (tabular Q-learning):
- Visits all state-action pairs infinitely often
- Learning rate decays appropriately
- Q-values converge to Q* with probability 1

## Conclusion

The new RL system transforms the Skyrim AGI from a **recording system** into a **learning system**. It now:

✅ Learns optimal actions through trial and error
✅ Improves behavior based on rewards and outcomes
✅ Balances exploration and exploitation intelligently
✅ Generalizes learning to new situations
✅ Continuously adapts and improves over time

This is **genuine machine learning** applied to autonomous gameplay.

---

**Implementation**: `/singularis/skyrim/reinforcement_learner.py`
**Integration**: `/singularis/skyrim/skyrim_agi.py`
**Author**: Claude (Anthropic)
**Date**: 2025
