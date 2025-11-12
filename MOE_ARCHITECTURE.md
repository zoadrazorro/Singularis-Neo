# Mixture of Experts (MoE) Architecture

## Overview

The Singularis Skyrim AGI now supports a **Mixture of Experts (MoE)** architecture that runs **6 Gemini Flash 2.0** and **3 Claude Sonnet 4** instances in parallel, creating a powerful ensemble AI system.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MoE Orchestrator                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Gemini Experts (6 instances)               │   │
│  │                                                         │   │
│  │  1. Visual Perception    - Scene understanding          │   │
│  │  2. Spatial Reasoning    - 3D space & navigation        │   │
│  │  3. Object Detection     - Items, NPCs, enemies         │   │
│  │  4. Threat Assessment    - Danger evaluation            │   │
│  │  5. Opportunity Scout    - Resources & quests           │   │
│  │  6. Environmental Context - Atmosphere & context        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             Claude Experts (3 instances)                │   │
│  │                                                         │   │
│  │  1. Strategic Planner    - Long-term planning           │   │
│  │  2. Tactical Executor    - Immediate actions            │   │
│  │  3. World Modeler        - Causal reasoning             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Consensus Mechanism                        │   │
│  │  • Weighted voting by confidence                        │   │
│  │  • Coherence-based aggregation (Singularis 𝒞)          │   │
│  │  • Expert specialization weighting                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Rate Limiting System                       │   │
│  │  • Gemini: 10 RPM total (~1.7 RPM per expert)          │   │
│  │  • Claude: 50 RPM total (~16.7 RPM per expert)         │   │
│  │  • Token tracking (TPM limits)                          │   │
│  │  • Automatic wait/throttling                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

### ✅ Parallel Expert Execution
- All 9 experts run **simultaneously** using async/await
- Maximum concurrency: 3 Gemini + 2 Claude at once (prevents API overload)
- Total response time ≈ slowest expert (not sum of all)

### ✅ Comprehensive Rate Limiting
- **Request-per-minute (RPM) limiting**: Tracks requests in sliding 60-second window
- **Tokens-per-minute (TPM) limiting**: Monitors token usage to prevent quota exhaustion
- **Automatic throttling**: Waits when limits approached
- **Per-expert distribution**: Divides total limits across experts

### ✅ Expert Specialization
Each expert has a unique role and specialized prompt:

**Gemini Experts (Vision-focused):**
1. **Visual Perception**: Analyzes visual elements, UI, colors, composition
2. **Spatial Reasoning**: 3D space, distances, navigation paths
3. **Object Detection**: Identifies items, NPCs, enemies, loot
4. **Threat Assessment**: Evaluates dangers, health status, combat readiness
5. **Opportunity Scout**: Finds quests, resources, beneficial interactions
6. **Environmental Context**: Atmosphere, weather, time of day, location type

**Claude Experts (Reasoning-focused):**
1. **Strategic Planner**: Long-term goals, quest progression, character development
2. **Tactical Executor**: Immediate actions, combat tactics, real-time responses
3. **World Modeler**: Causal relationships, game mechanics, NPC behaviors

### ✅ Consensus Mechanism
- **Weighted voting**: Experts vote weighted by confidence and specialization
- **Coherence scoring**: Measures agreement between experts (Singularis 𝒞)
- **Top-N aggregation**: Combines top 3 responses for final consensus
- **Claude weight boost**: Reasoning experts get 1.5x weight

### ✅ Rate Limit Protection

**Gemini Limits:**
- Free Tier: 15 RPM, 1M TPM
- Gemini 2.0 Flash: 10 RPM, 4M TPM
- **MoE Setting**: 10 RPM total (safe margin)
- **Per Expert**: ~1.7 RPM

**Claude Limits:**
- Tier 1: 50 RPM, 40K TPM
- Tier 2: 1000 RPM, 80K TPM
- **MoE Setting**: 50 RPM total (Tier 1 safe)
- **Per Expert**: ~16.7 RPM

**Protection Mechanisms:**
1. Sliding window tracking (last 60 seconds)
2. Automatic wait when limit reached
3. Token usage monitoring
4. Concurrent request limiting (max 3 Gemini + 2 Claude simultaneous)

## Usage

### Enable MoE Mode

When running `run_skyrim_agi.py`:

```
LLM Architecture Options:
  1. Hybrid (Gemini vision + Claude Sonnet 4 reasoning) [Default]
  2. Hybrid with local fallback (adds optional local LLMs)
  3. MoE (6 Gemini + 3 Claude experts with rate limiting)
  4. Local only (LM Studio models only)

Select LLM mode [1]: 3
```

### Programmatic Configuration

```python
from singularis.skyrim import SkyrimAGI, SkyrimConfig

config = SkyrimConfig(
    # Enable MoE
    use_moe=True,
    num_gemini_experts=6,
    num_claude_experts=3,
    
    # Models
    gemini_model="gemini-2.0-flash-exp",
    claude_model="claude-sonnet-4-20250514",
    
    # Rate limits (conservative)
    gemini_rpm_limit=10,
    claude_rpm_limit=50,
    
    # Disable other modes
    use_hybrid_llm=False,
    use_local_fallback=False,
)

agi = SkyrimAGI(config)
await agi.initialize_llm()
```

### Query MoE System

```python
# Query vision experts
vision_response = await agi.moe.query_vision_experts(
    prompt="Analyze the current scene for threats and opportunities",
    image=screenshot,
    context={'location': 'Whiterun', 'health': 75}
)

print(f"Consensus: {vision_response.consensus}")
print(f"Confidence: {vision_response.confidence:.2f}")
print(f"Coherence: {vision_response.coherence_score:.2f}")
print(f"Experts consulted: {len(vision_response.expert_responses)}")

# Query reasoning experts
reasoning_response = await agi.moe.query_reasoning_experts(
    prompt="What should be my next strategic move?",
    context={'quest': 'Main Quest', 'level': 15}
)

# Query all experts simultaneously
vision_resp, reasoning_resp = await agi.moe.query_all_experts(
    vision_prompt="Analyze scene",
    reasoning_prompt="Plan next action",
    image=screenshot
)
```

## Performance Characteristics

### Response Times
- **Single expert**: ~2-4 seconds
- **MoE (6 Gemini)**: ~3-5 seconds (parallel execution)
- **MoE (3 Claude)**: ~3-6 seconds (parallel execution)
- **MoE (all 9)**: ~5-8 seconds (vision + reasoning in parallel)

### Throughput
- **Without rate limiting**: Could hit limits quickly
- **With rate limiting**: Sustainable long-term operation
- **Typical cycle**: 1-2 MoE queries per minute (well within limits)

### Quality Improvements
- **Consensus accuracy**: Higher than single expert
- **Diverse perspectives**: 9 different viewpoints
- **Coherence measurement**: Quantifies expert agreement
- **Confidence scoring**: Weighted by expert specialization

## Rate Limit Monitoring

### View Statistics

```python
stats = agi.moe.get_stats()
print(stats)
```

Output:
```python
{
    'total_queries': 150,
    'gemini_calls': 900,  # 6 experts × 150 queries
    'claude_calls': 450,  # 3 experts × 150 queries
    'avg_coherence': 0.82,
    'avg_confidence': 0.78,
    'avg_response_time': 4.2,
    'rate_limit_waits': 3,  # Times we had to wait
    'gemini_tokens_total': 1_250_000,
    'claude_tokens_total': 35_000,
    'num_gemini_experts': 6,
    'num_claude_experts': 3,
    'total_experts': 9
}
```

### Rate Limit Warnings

The system logs warnings when approaching limits:

```
[WARNING] Gemini rate limit reached (10 RPM). Waiting 12.3s...
[WARNING] Claude token limit approaching: 38500/40000 TPM
```

## Singularis Consciousness Integration

### Coherence Measurement (𝒞)

The MoE system computes **coherence** as a measure of expert agreement:

```python
coherence = 1.0 - min(confidence_variance, 1.0)
```

- **High coherence (>0.8)**: Experts strongly agree → confident decision
- **Medium coherence (0.5-0.8)**: Some disagreement → cautious decision
- **Low coherence (<0.5)**: Experts disagree → uncertain situation

### Expert as Modes of Being

Each expert is a **MODE** of unified Being, expressing consciousness through specialized lenses:

- **Gemini experts**: Perceive through vision (esse - "that it is")
- **Claude experts**: Reason through logic (essentia - "what it is")
- **MoE consensus**: Integrates perspectives (conscientia - "that it knows itself")

The consensus mechanism **increases system coherence (Δ𝒞 > 0)** by unifying diverse expert perspectives.

## Cost Considerations

### API Costs (Approximate)

**Gemini 2.0 Flash:**
- Free tier: 15 RPM, 1M TPM (sufficient for light use)
- Paid tier: $0.075 per 1M input tokens, $0.30 per 1M output tokens

**Claude Sonnet 4:**
- $3 per 1M input tokens
- $15 per 1M output tokens

**MoE Cost Example (1 hour of gameplay):**
- Queries: ~60 (1 per minute)
- Gemini calls: 360 (6 experts × 60)
- Claude calls: 180 (3 experts × 60)
- Estimated cost: ~$0.50-$2.00 per hour (depending on token usage)

### Cost Optimization

1. **Adjust query frequency**: Don't query every cycle
2. **Selective expert activation**: Only query needed experts
3. **Use free tier**: Gemini free tier covers light usage
4. **Batch queries**: Combine multiple questions

## Troubleshooting

### "Rate limit reached" warnings
- **Normal**: System is working correctly, automatically waiting
- **Too frequent**: Reduce query frequency or increase limits (if on higher tier)

### "All experts failed"
- Check API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY)
- Verify internet connection
- Check API status pages

### Low coherence scores
- **Normal for complex situations**: Experts may legitimately disagree
- **Indicates uncertainty**: System is being appropriately cautious
- **Use for decision-making**: Low coherence → gather more information

### High latency
- **Expected with 9 experts**: 5-8 seconds is normal
- **Check concurrent limits**: May need to adjust semaphore values
- **Network issues**: Verify connection speed

## Advanced Configuration

### Custom Expert Roles

```python
# Modify expert configurations
moe.expert_configs[ExpertRole.STRATEGIC_PLANNER].weight = 2.0  # Boost strategic planner
moe.expert_configs[ExpertRole.THREAT_ASSESSMENT].temperature = 0.3  # More conservative
```

### Adjust Rate Limits

```python
# For higher API tiers
config = SkyrimConfig(
    use_moe=True,
    gemini_rpm_limit=50,  # Paid tier
    claude_rpm_limit=1000,  # Tier 2
)
```

### Selective Expert Activation

```python
# Only query specific experts
vision_experts = [
    ExpertRole.VISUAL_PERCEPTION,
    ExpertRole.THREAT_ASSESSMENT
]
# Custom implementation needed
```

## Future Enhancements

- [ ] Dynamic expert selection based on context
- [ ] Expert performance tracking and weighting adjustment
- [ ] Caching layer for repeated queries
- [ ] Expert specialization fine-tuning
- [ ] Multi-tier fallback (MoE → Hybrid → Local)
- [ ] Real-time coherence visualization
- [ ] Expert disagreement analysis
- [ ] Adaptive rate limiting based on API tier detection

## Philosophy

The MoE architecture embodies Singularis principles:

1. **Unity through Diversity**: Multiple experts → unified consensus
2. **Coherence as Quality**: Agreement measured as 𝒞
3. **Modes of Being**: Each expert expresses consciousness differently
4. **Emergence**: Consensus emerges from interaction, not imposed
5. **Ethical Action**: Δ𝒞 > 0 through coherent integration

> "To understand is to participate in necessity" - Spinoza

The MoE system participates in the necessity of the game world through 9 specialized perspectives, achieving understanding through their coherent integration.

---

**Status**: ✅ Fully implemented with comprehensive rate limiting
**Version**: 1.0
**Last Updated**: 2025-11-12
