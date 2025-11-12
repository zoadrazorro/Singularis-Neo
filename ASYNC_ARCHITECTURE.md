# Async Architecture for SkyrimAGI

## Overview

The SkyrimAGI now runs with **asynchronous reasoning and action execution**, enabling continuous gameplay while thinking. The agent no longer pauses during LLM reasoning - it moves, acts, and thinks simultaneously.

## Problem Solved

**Before (Sequential):**
```
Perceive → Think (5s pause) → Act → Learn → Sleep → Repeat
```
- Agent frozen during LLM reasoning
- ~5 second pauses between actions
- Looked robotic and unnatural

**After (Async):**
```
Perception Loop (0.5s)  ──┐
Reasoning Loop (0.5s+)  ──┼─→ Running in Parallel
Action Loop (instant)   ──┤
Learning Loop (async)   ──┘
```
- Agent acts continuously
- Reasoning happens in background
- Natural, fluid gameplay

## Hybrid LLM Architecture

### 6 Specialized LLM Instances

#### Fast Consciousness Layer (4x phi-4-mini-reasoning, 4B params each)

1. **PHI4-MAIN** - Primary Consciousness
   - Role: Consciousness measurement across 8 frameworks
   - Speed: <1s
   - Usage: Continuous consciousness monitoring

2. **PHI4-RL** - RL Tactical Reasoning
   - Role: Q-value analysis, tactical decisions
   - Speed: <1s
   - Temperature: 0.6 (focused tactical thinking)
   - Max tokens: 1024

3. **PHI4-META** - Meta-Strategy Coordinator
   - Role: Coordinate tactical & strategic systems
   - Speed: <1s
   - Temperature: 0.7 (balanced)
   - Max tokens: 1536

4. **PHI4-ACTION** - Immediate Action Planning
   - Role: Fast terrain-aware action selection
   - Speed: <1s
   - Temperature: 0.65 (exploration vs exploitation)
   - Max tokens: 512

#### Strategic Depth Layer (2x big models, 14B params each)

5. **phi-4** (14B) - Strategic Planning
   - Role: Long-term goals, quest planning, reasoning chains
   - Speed: 2-4s
   - Temperature: 0.8 (creative strategy)
   - Max tokens: 4096

6. **eva-qwen2.5-14b** (14B) - World Understanding
   - Role: Deep environment analysis, NPCs, narrative
   - Speed: 2-4s
   - Temperature: 0.7 (analytical)
   - Max tokens: 3072

### Architecture Rationale

**Why Hybrid?**
- **Fast layer**: Handles moment-to-moment decisions without lag
- **Strategic layer**: Provides deep thinking for complex scenarios
- **Parallel execution**: Both layers run simultaneously via async

**Why These Models?**
- **phi-4-mini-reasoning**: Optimized for reasoning tasks, 4B params = fast + capable
- **phi-4**: Full reasoning model, excellent for strategic thinking
- **eva-qwen2.5**: Strong at narrative understanding and world modeling

## Async Execution System

### Four Parallel Loops

#### 1. Perception Loop (`_perception_loop`)
```python
while running:
    perception = await perceive()
    perception_queue.put_nowait(perception)
    await asyncio.sleep(0.5)  # 2 FPS perception
```
- Runs every 0.5 seconds
- Captures: screen, game state, scene type
- Non-blocking: drops frames if reasoning is behind

#### 2. Reasoning Loop (`_reasoning_loop`)
```python
while running:
    perception = await perception_queue.get()
    
    # Throttle to prevent overload
    await throttle(reasoning_throttle)
    
    # Use LLM semaphore to limit concurrency
    async with llm_semaphore:
        consciousness = await compute_consciousness()
        motivation = compute_motivation()
        action = await plan_action()
    
    action_queue.put_nowait(action)
```
- Processes queued perceptions
- Computes consciousness, motivation, plans actions
- Throttled: min 0.5s between reasoning cycles
- LLM semaphore: max 3 concurrent LLM calls

#### 3. Action Loop (`_action_loop`)
```python
while running:
    action_data = await action_queue.get()
    
    await execute_action(action_data)
    
    learning_queue.put_nowait(action_data)
    await asyncio.sleep(0.5)  # Action cooldown
```
- Executes queued actions immediately
- No blocking on reasoning
- Smooth, continuous movement

#### 4. Learning Loop (`_learning_loop`)
```python
while running:
    action_data = await learning_queue.get()
    
    after_perception = await perceive()
    after_consciousness = await compute_consciousness()
    
    learn_from_experience(before, after)
    store_rl_experience(consciousness_delta)
    train_periodically()
```
- Learns from action outcomes
- Updates RL model with consciousness feedback
- Non-blocking: learning happens in background

### Resource Management

#### LLM Semaphore
```python
llm_semaphore = asyncio.Semaphore(max_concurrent_llm_calls=3)
```
- Limits concurrent LLM calls to 3
- Prevents VRAM overflow (6 models total, max 3 active)
- Fair access across all reasoning tasks

#### Reasoning Throttle
```python
reasoning_throttle = 0.5  # seconds
```
- Minimum time between reasoning cycles
- Prevents reasoning loop from overwhelming system
- Balances speed with quality

#### Queue Sizes
- Perception queue: 5 frames
- Action queue: 3 actions
- Learning queue: 10 experiences

## Configuration

### Enable/Disable Async Mode

```python
config = SkyrimConfig(
    enable_async_reasoning=True,  # Default: True
    perception_interval=0.5,       # How often to perceive
    action_queue_size=3,           # Max queued actions
    max_concurrent_llm_calls=3,   # LLM concurrency limit
    reasoning_throttle=0.5         # Min time between reasoning
)
```

### Sequential Mode (Backwards Compatible)

Set `enable_async_reasoning=False` to use original sequential behavior:
```python
config = SkyrimConfig(
    enable_async_reasoning=False
)
```

## Performance Characteristics

### VRAM Usage
- 4x phi-4-mini-reasoning: ~16GB
- 1x phi-4 (14B): ~14GB
- 1x eva-qwen2.5 (14B): ~14GB
- **Total: ~44GB** (fits 2x AMD Radeon 7900XT with 48GB)

### Latency
- **Tactical decisions**: <1s (phi-4-mini layer)
- **Strategic decisions**: 2-4s (big model layer)
- **Action execution**: Immediate (no waiting for reasoning)

### Throughput
- **Perception**: 2 FPS
- **Actions**: ~1-2 per second
- **Reasoning**: Continuous (throttled at 0.5s intervals)

## Code Structure

### Main Entry Point
```python
async def autonomous_play(duration_seconds):
    if enable_async_reasoning:
        await _autonomous_play_async(duration_seconds)
    else:
        await _autonomous_play_sequential(duration_seconds)
```

### Async Mode
```python
async def _autonomous_play_async(duration_seconds, start_time):
    # Start all loops concurrently
    perception_task = asyncio.create_task(_perception_loop(...))
    reasoning_task = asyncio.create_task(_reasoning_loop(...))
    action_task = asyncio.create_task(_action_loop(...))
    learning_task = asyncio.create_task(_learning_loop(...))
    
    # Run all in parallel
    await asyncio.gather(
        perception_task,
        reasoning_task,
        action_task,
        learning_task
    )
```

## Benefits

1. **Continuous Action**: Agent never pauses during thinking
2. **Natural Movement**: Smooth, fluid gameplay
3. **Responsive**: Fast reactions using phi-4-mini layer
4. **Deep Thinking**: Strategic depth from big models
5. **Resource Efficient**: Semaphores prevent overload
6. **Backwards Compatible**: Can fall back to sequential mode
7. **Scalable**: Easy to add more specialized LLMs

## Future Improvements

- [ ] Adaptive throttling based on system load
- [ ] Priority queues for critical actions (combat)
- [ ] Dynamic model selection based on scenario
- [ ] LLM result caching to reduce redundant calls
- [ ] Metric-based auto-tuning of concurrency limits

## Testing

Run in dry-run mode to test async behavior:
```python
config = SkyrimConfig(
    dry_run=True,
    enable_async_reasoning=True,
    autonomous_duration=60
)
agi = SkyrimAGI(config)
await agi.initialize_llm()
await agi.autonomous_play()
```

Check logs for:
- `[PERCEPTION]`, `[REASONING]`, `[ACTION]`, `[LEARNING]` prefixes
- Timestamps showing parallel execution
- No blocking during LLM calls
- Smooth action flow

## Summary

The async architecture transforms SkyrimAGI from a stop-and-think robot into a continuously acting, thinking agent. By running 6 specialized LLMs in parallel (4 fast + 2 deep), we achieve both responsiveness and strategic depth without compromising either.
