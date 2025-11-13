# Hybrid LLM Architecture

## Overview

The Singularis Skyrim AGI now uses a **hybrid LLM architecture** that combines the best of cloud AI and local models:

- **Primary**: Gemini 2.0 Flash (vision) + Claude Sonnet 4 (reasoning)
- **Optional Fallback**: Local LLMs via LM Studio

This architecture provides:
- ✅ **Best-in-class vision** with Gemini 2.0 Flash
- ✅ **Advanced reasoning** with Claude Sonnet 4
- ✅ **Async execution** for maximum performance
- ✅ **Optional local fallback** for reliability
- ✅ **Automatic failover** when cloud APIs are unavailable

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid LLM Client                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Gemini     │  │   Claude     │  │  Local LLMs     │  │
│  │  2.0 Flash   │  │  Sonnet 4    │  │  (Optional)     │  │
│  │              │  │              │  │                 │  │
│  │  - Vision    │  │  - Reasoning │  │  - Vision       │  │
│  │  - Scene     │  │  - Strategy  │  │  - Reasoning    │  │
│  │  - Spatial   │  │  - Planning  │  │  - Action       │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│         ▲                 ▲                    ▲           │
│         │                 │                    │           │
│         └─────────────────┴────────────────────┘           │
│                    Automatic Failover                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      Skyrim AGI Components            │
        │                                       │
        │  • Perception System                  │
        │  • Strategic Planner                  │
        │  • Meta-Strategist                    │
        │  • RL Reasoning Neuron                │
        │  • World Model                        │
        │  • Consciousness Bridge               │
        │  • Quest Tracker                      │
        │  • Dialogue Intelligence              │
        └───────────────────────────────────────┘
```

## Usage Modes

### 1. Hybrid Mode (Default)
Uses Gemini for vision and Claude Sonnet 4 for reasoning. No local models required.

```python
config = SkyrimConfig(
    use_hybrid_llm=True,
    use_gemini_vision=True,
    use_claude_reasoning=True,
    use_local_fallback=False  # No local fallback
)
```

**Requirements:**
- `GEMINI_API_KEY` environment variable
- `ANTHROPIC_API_KEY` environment variable

### 2. Hybrid with Local Fallback
Uses cloud AI primarily, but falls back to local LLMs if cloud APIs fail.

```python
config = SkyrimConfig(
    use_hybrid_llm=True,
    use_gemini_vision=True,
    use_claude_reasoning=True,
    use_local_fallback=True  # Enable local fallback
)
```

**Requirements:**
- `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` (primary)
- LM Studio running on `localhost:1234` (fallback)
- Local models loaded in LM Studio

### 3. Local Only Mode
Uses only local LLMs via LM Studio. No cloud APIs required.

```python
config = SkyrimConfig(
    use_hybrid_llm=False,  # Disable hybrid
    use_gemini_vision=False,
    use_claude_reasoning=False,
)
```

**Requirements:**
- LM Studio running on `localhost:1234`
- Local models loaded:
  - `qwen/qwen3-vl-8b` (vision)
  - `mistralai/mistral-7b-instruct-v0.3` (reasoning)
  - `mistralai/mistral-nemo-instruct-2407` (action)

## Interactive Setup

When running `run_skyrim_agi.py`, you'll be prompted to select the LLM mode:

```
LLM Architecture Options:
  1. Hybrid (Gemini vision + Claude Sonnet 4 reasoning) [Default]
  2. Hybrid with local fallback (adds optional local LLMs)
  3. Local only (LM Studio models only)

Select LLM mode [1]:
```

## API Keys Setup

### Gemini API Key
1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY = "your-api-key-here"
   
   # Or add to .env file
   GEMINI_API_KEY=your-api-key-here
   ```

### Claude API Key
1. Get an API key from [Anthropic Console](https://console.anthropic.com/)
2. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:ANTHROPIC_API_KEY = "your-api-key-here"
   
   # Or add to .env file
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Performance Characteristics

### Gemini 2.0 Flash (Vision)
- **Speed**: ~1-2 seconds per image analysis
- **Quality**: Excellent scene understanding and spatial awareness
- **Cost**: Free tier available (60 requests/minute)
- **Best for**: Real-time visual perception, scene analysis

### Claude Sonnet 4 (Reasoning)
- **Speed**: ~2-4 seconds per reasoning task
- **Quality**: State-of-the-art strategic thinking and planning
- **Cost**: Pay-per-use (check Anthropic pricing)
- **Best for**: Complex decision making, strategic planning, causal reasoning

### Local LLMs (Fallback)
- **Speed**: Varies by hardware (typically 5-15 seconds)
- **Quality**: Good for most tasks, excellent for specialized domains
- **Cost**: Free (runs locally)
- **Best for**: Privacy, offline operation, cost control

## Async Execution

The hybrid system uses async execution for maximum performance:

```python
# Multiple LLM calls can run in parallel
async with hybrid_llm.semaphore:
    vision_task = hybrid_llm.analyze_image(...)
    reasoning_task = hybrid_llm.generate_reasoning(...)
    
    # Both run concurrently
    vision_result, reasoning_result = await asyncio.gather(
        vision_task,
        reasoning_task
    )
```

**Benefits:**
- ⚡ Faster decision cycles
- 🔄 Parallel processing of vision and reasoning
- 📊 Better resource utilization
- 🎯 Lower latency

## Automatic Failover

The system automatically falls back to local LLMs when:
- Cloud API is unavailable
- API rate limits are hit
- Network connectivity issues
- API key is missing or invalid

**Failover Flow:**
```
1. Try Gemini/Claude (primary)
   ↓ (if fails)
2. Try local LLM (fallback)
   ↓ (if fails)
3. Raise error
```

## Statistics and Monitoring

The hybrid client tracks usage statistics:

```python
stats = hybrid_llm.get_stats()
print(stats)
# {
#     'gemini_calls': 150,
#     'claude_calls': 200,
#     'local_calls': 10,
#     'fallback_activations': 5,
#     'errors': 2,
#     'total_calls': 360,
#     'avg_time': 2.3,
#     'primary_success_rate': 0.97,
#     'fallback_rate': 0.014
# }
```

## Configuration Reference

### HybridConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gemini_vision` | bool | True | Enable Gemini for vision |
| `gemini_model` | str | "gemini-2.0-flash-exp" | Gemini model name |
| `use_claude_reasoning` | bool | True | Enable Claude for reasoning |
| `claude_model` | str | "claude-sonnet-4-20250514" | Claude model name |
| `use_local_fallback` | bool | False | Enable local LLM fallback |
| `local_base_url` | str | "http://localhost:1234/v1" | LM Studio URL |
| `timeout` | int | 30 | Request timeout (seconds) |
| `max_concurrent_requests` | int | 4 | Max parallel requests |
| `min_request_interval` | float | 0.1 | Min time between requests |

## Troubleshooting

### "Gemini API key not found"
- Set `GEMINI_API_KEY` environment variable
- Or add to `.env` file in project root

### "Claude API key not found"
- Set `ANTHROPIC_API_KEY` environment variable
- Or add to `.env` file in project root

### "Local fallback failed"
- Ensure LM Studio is running on `localhost:1234`
- Check that models are loaded in LM Studio
- Verify model names match configuration

### "No vision model available"
- At least one vision model must be available (Gemini or local)
- Check API keys and LM Studio connection

### "No reasoning model available"
- At least one reasoning model must be available (Claude or local)
- Check API keys and LM Studio connection

## Best Practices

1. **Start with Hybrid Mode**: Use cloud AI for best performance
2. **Enable Fallback for Production**: Add local fallback for reliability
3. **Monitor Statistics**: Track usage and failover rates
4. **Set Appropriate Timeouts**: Balance speed vs reliability
5. **Use Rate Limiting**: Respect API rate limits
6. **Handle Errors Gracefully**: Implement proper error handling

## Future Enhancements

- [ ] Support for additional cloud providers (OpenAI, Cohere, etc.)
- [ ] Dynamic model selection based on task complexity
- [ ] Caching layer for repeated queries
- [ ] Load balancing across multiple API keys
- [ ] Cost tracking and optimization
- [ ] A/B testing framework for model comparison

## License

Part of the Singularis AGI project.
