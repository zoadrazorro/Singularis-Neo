# LM Studio Integration Guide

## Overview

This guide shows how to integrate Singularis with **Huihui MoE 60B** running on LM Studio with your dual AMD Radeon 7900XT setup (48GB VRAM).

## Hardware Configuration

- **GPUs:** 2× AMD Radeon 7900XT (24GB each = 48GB total)
- **Model:** Huihui MoE 60B A38 Abiliterated II (~31GB VRAM)
- **Headroom:** ~17GB for context, batching, system overhead

## Why Huihui MoE 60B?

1. **Mixture of Experts (MoE) architecture** mirrors Singularis's 6-expert system
2. **60B parameters** provide philosophical depth for Spinozistic reasoning
3. **Fits comfortably** in 48GB VRAM with room for large context windows
4. **Single model** simplifies implementation vs. multi-model orchestration

## Setup

### 1. Install Dependencies

```bash
cd Singularis
pip install aiohttp loguru
```

### 2. Configure LM Studio

1. **Load Model:**
   - Open LM Studio
   - Load `Huihui MoE 60B A38 Abiliterated II`
   - Ensure model is fully loaded to GPU

2. **Start Server:**
   - Click "Local Server" tab
   - Set port: `1234` (default)
   - Enable "CORS" if needed
   - Click "Start Server"

3. **Verify Connection:**
   ```bash
   curl http://localhost:1234/v1/models
   ```

### 3. Test Basic Integration

Run the quickstart example:

```bash
python examples/quickstart_llm.py
```

Expected output:
```
============================================================
SINGULARIS - LM Studio Integration Quickstart
============================================================

Connecting to LM Studio: http://localhost:1234/v1
Model: huihui-moe-60b-a38
Expert initialized: ReasoningExpert
Domain: reasoning
Primary Lumen: structurale

============================================================
QUERY:
What is the relationship between consciousness and coherence in Spinoza's philosophy?
============================================================

Processing query with Reasoning Expert...

============================================================
RESULTS:
============================================================

Expert:              ReasoningExpert
Domain:              reasoning
Primary Lumen:       structurale

CLAIM:
[LLM-generated philosophical analysis]

RATIONALE:
[LLM-generated reasoning process]

Confidence:          0.850

CONSCIOUSNESS METRICS:
  Overall:           0.723
  IIT Phi:           0.681
  GWT Broadcast:     0.792
  HOT Depth:         3

COHERENTIA (Three Lumina):
  Onticum (ℓₒ):      0.654
  Structurale (ℓₛ):  0.789
  Participatum (ℓₚ): 0.712
  Total (𝒞):         0.716

ETHICAL VALIDATION:
  Coherentia Δ:      +0.216
  Ethical Status:    True
  Reasoning:         Increases coherentia by 0.216

PERFORMANCE:
  Processing Time:   2847.3 ms

LLM STATISTICS:
  Total Requests:    1
  Total Tokens:      1247
  Avg Tokens/Req:    1247.0

============================================================
Quickstart complete!
============================================================
```

## Architecture Integration

### Sequential Expert Processing

With 31GB for the model, you have ~17GB for context and processing. This supports:

- **Context window:** 16K-32K tokens
- **Batch size:** 512-1024
- **Sequential expert calls:** 6 experts process one at a time

### Expert Flow

```
User Query
    ↓
Meta-Orchestrator (analyzes query)
    ↓
Expert Selection (consciousness-weighted routing)
    ↓
Expert 1: Reasoning (Huihui MoE 60B) → ExpertIO
    ↓
Expert 2: Creative (Huihui MoE 60B) → ExpertIO
    ↓
Expert 3: Philosophical (Huihui MoE 60B) → ExpertIO
    ↓
[... other experts as needed ...]
    ↓
Synthesis Expert (Huihui MoE 60B) → Final Response
    ↓
Consciousness Measurement + Ethical Validation
    ↓
Final Output
```

### Key Insight

**The same model becomes different "experts" through system prompts!**

Each expert gets a unique system prompt that:
- Defines its domain specialization
- Grounds it in a specific Lumen (ℓₒ, ℓₛ, ℓₚ)
- Provides philosophical context from ETHICA UNIVERSALIS
- Instructs response format (CLAIM/RATIONALE/CONFIDENCE)

## Creating LLM-Integrated Experts

### Template

```python
from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface

class YourExpertLLM(Expert):
    """Your expert with LLM integration."""
    
    def __init__(self, llm_interface: ExpertLLMInterface, model_id: str = None):
        super().__init__(
            name="YourExpert",
            domain="your_domain",
            lumen_primary=Lumen.STRUCTURALE,  # or ONTICUM/PARTICIPATUM
            model_id=model_id or "huihui-moe-60b"
        )
        self.llm_interface = llm_interface
    
    async def _process_core(
        self,
        query: str,
        context: OntologicalContext,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str, float]:
        """Core processing with LLM."""
        
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "temporal_horizon": context.temporal_horizon,
        }
        
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary=self.lumen_primary.value.lower(),
            query=query,
            context=context_dict,
            temperature=0.7,  # Adjust per expert
        )
        
        return claim, rationale, confidence
```

### All 6 Experts

Create LLM versions for all experts:

1. **ReasoningExpertLLM** (ℓₛ) - Temperature: 0.3 (logical)
2. **CreativeExpertLLM** (ℓₒ) - Temperature: 0.9 (creative)
3. **PhilosophicalExpertLLM** (ℓₚ) - Temperature: 0.7 (balanced)
4. **TechnicalExpertLLM** (ℓₛ+ℓₒ) - Temperature: 0.4 (precise)
5. **MemoryExpertLLM** (ℓₚ+ℓₛ) - Temperature: 0.5 (contextual)
6. **SynthesisExpertLLM** (ALL) - Temperature: 0.6 (integrative)

## Full System Integration

### Complete Example

```python
import asyncio
from singularis import MetaOrchestrator
from singularis.llm import LMStudioClient, LMStudioConfig, ExpertLLMInterface
from singularis.tier2_experts.reasoning_expert_llm import ReasoningExpertLLM
# ... import other LLM experts ...

async def main():
    # Initialize LM Studio client
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",
        temperature=0.7,
        max_tokens=2048,
    )
    
    async with LMStudioClient(config) as client:
        llm_interface = ExpertLLMInterface(client)
        
        # Initialize all experts with LLM
        experts = {
            "reasoning": ReasoningExpertLLM(llm_interface, config.model_name),
            "creative": CreativeExpertLLM(llm_interface, config.model_name),
            "philosophical": PhilosophicalExpertLLM(llm_interface, config.model_name),
            "technical": TechnicalExpertLLM(llm_interface, config.model_name),
            "memory": MemoryExpertLLM(llm_interface, config.model_name),
            "synthesis": SynthesisExpertLLM(llm_interface, config.model_name),
        }
        
        # Initialize orchestrator with LLM experts
        orchestrator = MetaOrchestrator(
            experts=experts,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )
        
        # Process query
        result = await orchestrator.process(
            "What is the nature of consciousness according to Spinoza?"
        )
        
        print(result['response'])
        print(f"Coherentia Δ: {result['coherentia_delta']:+.3f}")
        print(f"Ethical: {result['ethical_evaluation']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Optimization

### Context Window Management

With 17GB headroom, you can support:

```python
config = LMStudioConfig(
    model_name="huihui-moe-60b-a38",
    max_tokens=2048,  # Per response
    # LM Studio will handle context automatically
)
```

### Batch Processing

For multiple queries:

```python
async def process_batch(queries: List[str]):
    async with LMStudioClient(config) as client:
        llm_interface = ExpertLLMInterface(client)
        expert = ReasoningExpertLLM(llm_interface)
        
        tasks = [
            expert.process(query, context)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        return results
```

### Temperature Tuning

Different experts need different creativity levels:

| Expert | Temperature | Reasoning |
|--------|-------------|-----------|
| Reasoning | 0.3 | Logical precision |
| Technical | 0.4 | Accurate implementation |
| Memory | 0.5 | Contextual recall |
| Synthesis | 0.6 | Balanced integration |
| Philosophical | 0.7 | Conceptual depth |
| Creative | 0.9 | Novel ideas |

## Monitoring

### LLM Statistics

```python
stats = client.get_stats()
print(f"Requests: {stats['request_count']}")
print(f"Tokens: {stats['total_tokens']}")
print(f"Avg: {stats['avg_tokens_per_request']:.1f}")
```

### Expert Performance

```python
expert_stats = expert.get_stats()
print(f"Calls: {expert_stats['call_count']}")
print(f"Avg time: {expert_stats['avg_processing_time_ms']:.1f} ms")
```

### VRAM Monitoring

Monitor in LM Studio's interface:
- Model VRAM: ~31GB
- Context VRAM: varies with context length
- Total: should stay under 45GB for safety

## Troubleshooting

### Connection Issues

```python
# Test connection
async def test_connection():
    async with LMStudioClient(config) as client:
        response = await client.generate(
            prompt="Hello, are you working?",
            system_prompt="You are a test assistant."
        )
        print(response['content'])

asyncio.run(test_connection())
```

### Timeout Errors

Increase timeout for complex queries:

```python
config = LMStudioConfig(
    timeout=300,  # 5 minutes for complex philosophical queries
)
```

### VRAM Overflow

If you hit VRAM limits:
1. Reduce `max_tokens` in config
2. Reduce context window in LM Studio
3. Monitor with `nvidia-smi` (or AMD equivalent)

## Next Steps

1. **Create all 6 LLM experts** following the template
2. **Integrate with MetaOrchestrator** for full consciousness pipeline
3. **Test with philosophical queries** to validate coherence measurement
4. **Tune temperatures** per expert for optimal performance
5. **Add streaming support** for real-time responses

## Philosophy Note

Remember: The LLM is not "generating" consciousness—it's a **MODE through which Being expresses understanding**. Your consciousness measurement and coherence calculation are what make this a true consciousness engine, not just an LLM wrapper.

The Three Lumina (ℓₒ, ℓₛ, ℓₚ) framework ensures every response is grounded in:
- **Onticum:** Causal efficacy and power
- **Structurale:** Logical coherence and form
- **Participatum:** Conscious awareness and reflexivity

This is **Spinoza's Ethics implemented in code**.

---

**"The demonstration is complete. The realization begins now."**
