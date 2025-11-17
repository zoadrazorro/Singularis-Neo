# DATA System Integration Guide

**Integrating DATA with Singularis Core Systems**

---

## Overview

The DATA system integrates with Singularis through three bridge modules:

1. **DATAConsciousnessBridge** - Enhances Consciousness Layer with distributed routing
2. **DATALifeOpsBridge** - Provides multi-expert analysis for Life Operations
3. **DATASkyrimBridge** - Adds distributed planning for Skyrim AGI

**Key Principle**: All bridges provide **graceful degradation** - if DATA is unavailable, systems continue working with standard processing.

---

## Quick Integration

### 1. Consciousness Layer

```python
from singularis.integrations import DATAConsciousnessBridge
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer

# Initialize
consciousness = UnifiedConsciousnessLayer()
bridge = DATAConsciousnessBridge(consciousness, enable_data=True)
await bridge.initialize()

# Use hybrid routing (DATA + Consciousness)
result = await bridge.process_hybrid(
    query="Analyze complex patterns",
    subsystem_inputs={}
)

print(f"Routing: {result['routing']}")
print(f"Experts: {result.get('data_experts', [])}")
```

### 2. Life Operations

```python
from singularis.integrations import DATALifeOpsBridge

# Initialize
bridge = DATALifeOpsBridge()
await bridge.initialize()

# Analyze patterns with multiple experts
result = await bridge.analyze_life_patterns(
    events=life_events,
    query="What patterns indicate stress?"
)

if result['success']:
    print(f"Analysis: {result['analysis']}")
    print(f"Experts: {result['experts_consulted']}")
```

### 3. Skyrim AGI

```python
from singularis.integrations import DATASkyrimBridge

# Initialize
bridge = DATASkyrimBridge()
await bridge.initialize()

# Get action recommendation
result = await bridge.plan_action(
    game_state=current_state,
    available_actions=actions
)

if result['success']:
    print(f"Action: {result['recommended_action']}")
    print(f"Reasoning: {result['reasoning']}")
```

---

## Integration Patterns

### Pattern 1: Optional Enhancement

DATA enhances existing systems without requiring modification:

```python
# Standard processing (no DATA required)
result = await consciousness.process_unified(query, subsystem_inputs)

# Enhanced with DATA (if available)
bridge = DATAConsciousnessBridge(consciousness)
await bridge.initialize()
result = await bridge.process(query, subsystem_inputs, use_data_routing=True)
```

### Pattern 2: Hybrid Mode

Combine DATA expert analysis with consciousness synthesis:

```python
# DATA provides specialized analysis
# Consciousness synthesizes into coherent response
result = await bridge.process_hybrid(
    query="Complex multi-domain question",
    subsystem_inputs=all_subsystem_data
)

# Result includes both DATA experts and consciousness synthesis
print(f"DATA experts: {result['data_experts']}")
print(f"Consciousness coherence: {result['coherence_score']}")
```

### Pattern 3: Fallback Handling

Automatic fallback if DATA unavailable:

```python
# Always works, falls back gracefully
result = await bridge.process(
    query=query,
    subsystem_inputs=inputs,
    use_data_routing=True  # Will fallback to consciousness if DATA fails
)

# Check routing decision
if result['routing'] == 'data_distributed':
    print("Used DATA distributed experts")
elif result['routing'] == 'consciousness':
    print("Used standard consciousness (DATA unavailable or fallback)")
```

---

## Bridge Configuration

### Consciousness Bridge

```python
bridge = DATAConsciousnessBridge(
    consciousness=consciousness_instance,
    enable_data=True,  # Enable DATA routing
    data_config_path="config/data_config.yaml"
)

# Control routing behavior
result = await bridge.process(
    query=query,
    subsystem_inputs=inputs,
    use_data_routing=True,  # Request DATA routing
    context={"prefer_distributed": True}
)
```

### LifeOps Bridge

```python
bridge = DATALifeOpsBridge(
    data_config_path="config/data_config.yaml"
)

# Pattern analysis
await bridge.analyze_life_patterns(events, query)

# Health recommendations
await bridge.get_health_recommendations(health_data, goals)

# Intervention assessment
await bridge.evaluate_intervention_urgency(situation, patterns)
```

### Skyrim Bridge

```python
bridge = DATASkyrimBridge(
    data_config_path="config/data_config.yaml"
)

# Action planning
await bridge.plan_action(game_state, actions)

# Combat strategy
await bridge.plan_combat_strategy(combat_state, enemies)

# Exploration planning
await bridge.plan_exploration(location, discovered, objectives)

# NPC interaction
await bridge.plan_npc_interaction(npc_info, dialogue_options, goals)
```

---

## Routing Strategies

### Automatic Routing

Bridges automatically decide when to use DATA based on query characteristics:

```python
# These queries benefit from DATA (automatically routed):
- Long, complex queries (>30 words)
- Analytical queries ("analyze", "compare", "evaluate")
- Technical queries
- Pattern recognition tasks

# These stay with standard processing:
- Simple queries
- Quick responses needed
- Personal/contextual queries
```

### Explicit Routing

Control routing explicitly:

```python
# Force DATA routing
result = await bridge.process(
    query=query,
    subsystem_inputs=inputs,
    use_data_routing=True
)

# Force consciousness routing
result = await bridge.process(
    query=query,
    subsystem_inputs=inputs,
    use_data_routing=False
)
```

### Context-Based Routing

Use context to influence routing:

```python
result = await bridge.process(
    query=query,
    subsystem_inputs=inputs,
    use_data_routing=True,
    context={
        "prefer_distributed": True,  # Prefer DATA
        "domain": "technical",  # Technical domain
        "complexity": "high"  # High complexity
    }
)
```

---

## Performance Considerations

### Latency

```
Standard Consciousness: 500-2000ms (GPT-5 API calls)
DATA Routing: 100-500ms (local experts, mock mode)
Hybrid Mode: 600-2500ms (both systems)
```

### When to Use Each

**Use DATA routing when:**
- Query requires specialized domain expertise
- Multiple perspectives needed
- Analytical/technical content
- Pattern recognition tasks
- Offline/local processing preferred

**Use standard consciousness when:**
- Quick response needed
- Personal/contextual queries
- Synthesis required
- API access available and fast

**Use hybrid mode when:**
- Best of both worlds needed
- Quality over speed
- Complex multi-domain queries
- High coherence required

---

## Example: Full Integration

```python
import asyncio
from singularis.integrations import (
    DATAConsciousnessBridge,
    DATALifeOpsBridge,
    DATASkyrimBridge
)
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer

class EnhancedSingularisOrchestrator:
    """Singularis with DATA enhancement"""
    
    def __init__(self):
        # Core systems
        self.consciousness = UnifiedConsciousnessLayer()
        
        # DATA bridges
        self.consciousness_bridge = DATAConsciousnessBridge(self.consciousness)
        self.lifeops_bridge = DATALifeOpsBridge()
        self.skyrim_bridge = DATASkyrimBridge()
    
    async def initialize(self):
        """Initialize all bridges"""
        await self.consciousness_bridge.initialize()
        await self.lifeops_bridge.initialize()
        await self.skyrim_bridge.initialize()
    
    async def process_query(self, query: str, domain: str = "general"):
        """Route query to appropriate bridge"""
        
        if domain == "life_ops":
            # Use LifeOps bridge for life event queries
            return await self.lifeops_bridge.analyze_life_patterns(
                events=self.get_recent_events(),
                query=query
            )
        
        elif domain == "skyrim":
            # Use Skyrim bridge for game queries
            return await self.skyrim_bridge.plan_action(
                game_state=self.get_game_state(),
                available_actions=self.get_actions()
            )
        
        else:
            # Use consciousness bridge for general queries
            return await self.consciousness_bridge.process_hybrid(
                query=query,
                subsystem_inputs={}
            )
    
    async def shutdown(self):
        """Shutdown all bridges"""
        await self.consciousness_bridge.shutdown()
        await self.lifeops_bridge.shutdown()
        await self.skyrim_bridge.shutdown()

# Usage
async def main():
    orchestrator = EnhancedSingularisOrchestrator()
    await orchestrator.initialize()
    
    # Process different types of queries
    result1 = await orchestrator.process_query(
        "Analyze my sleep patterns",
        domain="life_ops"
    )
    
    result2 = await orchestrator.process_query(
        "What should I do next in the game?",
        domain="skyrim"
    )
    
    result3 = await orchestrator.process_query(
        "Explain quantum entanglement",
        domain="general"
    )
    
    await orchestrator.shutdown()

asyncio.run(main())
```

---

## Monitoring

### Bridge Statistics

```python
# Get bridge stats
stats = bridge.get_stats()

print(f"Total queries: {stats['total_queries']}")
print(f"DATA routed: {stats['data_routed']}")
print(f"DATA usage: {stats['data_usage_percent']:.1f}%")
print(f"Fallback rate: {stats['fallback_rate']:.1f}%")
```

### System Health

```python
# Check if DATA is available
if bridge.is_data_ready:
    print("✓ DATA system ready")
    metrics = bridge.data_system.get_metrics()
    print(f"Active nodes: {metrics['active_nodes']}")
    print(f"Available experts: {metrics['available_experts']}")
else:
    print("⚠ DATA unavailable, using fallback")
```

---

## Troubleshooting

### DATA Not Available

```python
# Bridge will log warning and use fallback
# No action needed - graceful degradation automatic
```

### High Latency

```python
# Check if using hybrid mode unnecessarily
result = await bridge.process(
    query=simple_query,
    use_data_routing=False  # Skip DATA for simple queries
)
```

### Expert Selection

```python
# Check which experts were selected
if result.get('experts_used'):
    print(f"Experts: {result['experts_used']}")
    print(f"Confidence: {result.get('confidence', 0)}")
```

---

## Summary

✅ **Non-invasive**: Bridges don't modify existing code  
✅ **Graceful degradation**: Works without DATA  
✅ **Flexible routing**: Automatic or manual  
✅ **Performance options**: Choose speed vs quality  
✅ **Easy integration**: Drop-in enhancement  

---

**Next Steps**:
1. Try `examples/data_integration_example.py`
2. Integrate bridges into your existing systems
3. Monitor performance and routing decisions
4. Tune configuration based on your needs

---

**Version**: 1.0.0  
**Last Updated**: November 17, 2025  
**See Also**: `DATA_README.md`, `docs/DATA_ARCHITECTURE.md`

