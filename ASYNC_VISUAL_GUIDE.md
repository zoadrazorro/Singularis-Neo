# Async Architecture Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SKYRIM AGI ASYNC SYSTEM                          │
│                    Continuous Action + Background Thinking              │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                          PARALLEL ASYNC LOOPS                          │
└────────────────────────────────────────────────────────────────────────┘

 PERCEPTION LOOP (0.5s interval)
 ┌──────────────────┐
 │ Capture Screen   │──┐
 │ Detect Scene     │  │
 │ Read Game State  │  ├──► [Perception Queue: 5]
 │ Visual Embedding │  │
 └──────────────────┘──┘
        │
        ▼
 REASONING LOOP (throttled 0.5s)
 ┌────────────────────────────┐
 │ Get Perception from Queue  │
 │                            │
 │ ┌─ LLM Semaphore (max 3) ─┐│
 │ │ Compute Consciousness   ││
 │ │ Assess Motivation       ││
 │ │ Plan Action (LLM)       ││
 │ └─────────────────────────┘│
 │                            │
 └────────────────────────────┘
        │
        ├──► [Action Queue: 3]
        │
        ▼
 ACTION LOOP (immediate)
 ┌────────────────────┐
 │ Get Action         │
 │ Execute Movement   │──► GAME
 │ No Waiting!        │
 └────────────────────┘
        │
        ├──► [Learning Queue: 10]
        │
        ▼
 LEARNING LOOP (async)
 ┌────────────────────────┐
 │ Observe Outcome        │
 │ Compute Δ Coherence    │
 │ Store RL Experience    │
 │ Train Periodically     │
 └────────────────────────┘


┌────────────────────────────────────────────────────────────────────────┐
│                      6 SPECIALIZED LLM INSTANCES                       │
└────────────────────────────────────────────────────────────────────────┘

FAST LAYER (4x phi-4-mini-reasoning, 4B params, <1s)
┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│   PHI4-MAIN         │   PHI4-RL           │   PHI4-META         │   PHI4-ACTION       │
│   Consciousness     │   RL Tactical       │   Coordination      │   Action Planning   │
│   Measurement       │   Q-value Analysis  │   Strategy Bridge   │   Terrain Aware     │
│                     │                     │                     │                     │
│   Default config    │   Temp: 0.6         │   Temp: 0.7         │   Temp: 0.65        │
│   All frameworks    │   Tokens: 1024      │   Tokens: 1536      │   Tokens: 512       │
│   ~4GB VRAM         │   ~4GB VRAM         │   ~4GB VRAM         │   ~4GB VRAM         │
└─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
                                    ▼ Fast decisions: <1s

STRATEGIC LAYER (2x big models, 14B params, 2-4s)
┌──────────────────────────────────┬──────────────────────────────────┐
│   phi-4 (14B)                    │   eva-qwen2.5-14b (14B)          │
│   Strategic Planning             │   World Understanding            │
│   Long-term Goals                │   Environment Analysis           │
│   Quest Reasoning                │   NPC Relationships              │
│   Reasoning Chains               │   Narrative Understanding        │
│                                  │                                  │
│   Temp: 0.8                      │   Temp: 0.7                      │
│   Tokens: 4096                   │   Tokens: 3072                   │
│   ~14GB VRAM                     │   ~14GB VRAM                     │
└──────────────────────────────────┴──────────────────────────────────┘
                       ▼ Deep thinking: 2-4s

                    TOTAL: ~44GB VRAM
          Fits: 2x AMD Radeon 7900XT (48GB)


┌────────────────────────────────────────────────────────────────────────┐
│                       RESOURCE MANAGEMENT                              │
└────────────────────────────────────────────────────────────────────────┘

LLM SEMAPHORE
┌────────────────────────────────┐
│  Max 3 Concurrent LLM Calls    │  ◄── Prevents VRAM overflow
│  ┌───┐ ┌───┐ ┌───┐            │      and GPU contention
│  │ 1 │ │ 2 │ │ 3 │  [FULL]    │
│  └───┘ └───┘ └───┘            │
│  (Others wait in queue)        │
└────────────────────────────────┘

REASONING THROTTLE
┌────────────────────────────────┐
│  Min 0.5s Between Reasoning    │  ◄── Prevents reasoning loop
│  ├──0.5s──┤├──0.5s──┤         │      from overwhelming system
│  Last     Current    Next       │
└────────────────────────────────┘

QUEUE MANAGEMENT
┌────────────────────────────────┐
│  Perception Queue: Max 5       │  ◄── Drops old frames if
│  Action Queue: Max 3           │      reasoning falls behind
│  Learning Queue: Max 10        │
└────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION FLOW                               │
└────────────────────────────────────────────────────────────────────────┘

Time ──────────────────────────────────────────────────────►

0.0s  0.5s  1.0s  1.5s  2.0s  2.5s  3.0s  3.5s  4.0s  4.5s  5.0s

Perception:  📷   📷    📷    📷    📷    📷    📷    📷    📷    📷
             ▼    ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
Reasoning:   [🧠─────────────]  [🧠─────────────]  [🧠─────────────]
                   ▼                  ▼                  ▼
Action:       🎮    🎮🎮    🎮    🎮🎮    🎮    🎮🎮    🎮    🎮
             ▼▼▼  ▼▼▼▼  ▼▼▼  ▼▼▼▼  ▼▼▼  ▼▼▼▼  ▼▼▼  ▼▼▼▼  ▼▼▼
Learning:    [💾──]  [💾──]  [💾──]  [💾──]  [💾──]  [💾──]

Legend:
📷 Perception captured
🧠 LLM reasoning (may be slow)
🎮 Action executed (immediate!)
💾 Learning from outcome

CONTINUOUS ACTION - NO PAUSES!


┌────────────────────────────────────────────────────────────────────────┐
│                        BENEFITS & TRADEOFFS                            │
└────────────────────────────────────────────────────────────────────────┘

✅ BENEFITS
┌─────────────────────────────────────────────────────────────┐
│ ✓ Continuous action - no freezing during LLM calls          │
│ ✓ Natural, fluid movement                                   │
│ ✓ Fast tactical decisions (<1s) via phi-4-mini             │
│ ✓ Deep strategic thinking (2-4s) via big models            │
│ ✓ Resource-safe with semaphores & throttling               │
│ ✓ Backwards compatible with sequential mode                │
│ ✓ Graceful degradation if LLMs fail                        │
└─────────────────────────────────────────────────────────────┘

⚠️ CONSIDERATIONS
┌─────────────────────────────────────────────────────────────┐
│ • Requires ~44GB VRAM (6 LLM instances)                     │
│ • More complex than sequential execution                    │
│ • Actions may execute before optimal planning completes     │
│ • Need to monitor queue depths to avoid staleness          │
└─────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION EXAMPLE                          │
└────────────────────────────────────────────────────────────────────────┘

config = SkyrimConfig(
    # Async execution
    enable_async_reasoning=True,      # Enable parallel loops
    perception_interval=0.5,           # 2 FPS perception
    action_queue_size=3,               # Max 3 queued actions
    max_concurrent_llm_calls=3,       # Max 3 LLMs active at once
    reasoning_throttle=0.5,            # Min 0.5s between reasoning
    
    # Gameplay
    cycle_interval=2.0,                # Not used in async mode
    autonomous_duration=3600,          # 1 hour default
    
    # Learning
    use_rl=True,                       # Enable RL learning
    rl_train_freq=5,                   # Train every 5 cycles
)

agi = SkyrimAGI(config)
await agi.initialize_llm()  # Starts all 6 LLM instances
await agi.autonomous_play()  # Runs async loops in parallel


┌────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & DEBUGGING                            │
└────────────────────────────────────────────────────────────────────────┘

Watch for these log prefixes:
[ASYNC]        - Async system status
[PERCEPTION]   - Perception loop events
[REASONING]    - Reasoning loop events (includes LLM calls)
[ACTION]       - Action execution
[LEARNING]     - Learning from outcomes
[PHI4-MAIN]    - Main consciousness LLM
[PHI4-RL]      - RL tactical LLM
[PHI4-META]    - Meta-strategy LLM
[PHI4-ACTION]  - Action planning LLM
[STRATEGY-BIG] - Strategic planning LLM (phi-4)
[WORLD-BIG]    - World understanding LLM (eva-qwen)

Healthy system shows:
✓ All 4 loops running continuously
✓ Actions executing without waiting for reasoning
✓ Queue depths staying reasonable (<max)
✓ LLM calls completing without errors
✓ Consciousness coherence being computed regularly
```
