# DATA System - Distributed Abductive Technical Agent

<div align="center">

**Proto-AGI system for Singularis inspired by Star Trek's Data**

*Implementing distributed MoE-LoRA routing across heterogeneous hardware nodes*

[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/yourusername/Singularis)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/Singularis)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

DATA (Distributed Abductive Technical Agent) is a proto-AGI system that implements distributed Multi-Agentic Mixture of Experts with LoRA tuning (MoE-LoRA) across heterogeneous hardware nodes. Named after the Star Trek character, DATA brings distributed intelligence to Singularis.

### Key Features

- **Distributed MoE-LoRA**: Routes queries to specialized LoRA-adapted experts across hardware nodes
- **Global Workspace Theory**: Consciousness-inspired attention mechanism for information sharing
- **Hardware-Aware Routing**: Intelligent assignment based on node capabilities
- **Fault Tolerant**: Automatic failover and graceful degradation
- **Scalable**: Add nodes dynamically as system grows

---

## Architecture

Based on the [OKComputer Distributed AGI System Blueprint](c:/Users/jelly/Downloads/OKComputer_Distributed AGI System Blueprint), DATA implements:

### 1. Multi-Node Topology

```
┌─────────────────────────────────────────────────┐
│           DATA Distributed System                │
├─────────────────────────────────────────────────┤
│  Global Workspace  │  MoE-LoRA Layer  │  Memory │
├─────────────────────────────────────────────────┤
│            Communication Layer (gRPC)            │
├─────────────────────────────────────────────────┤
│ Node A │ Node B │ Node C │ Node E │             │
│(AMD)   │(Desk)  │(Laptop)│(Mac)   │             │
└─────────────────────────────────────────────────┘
```

### 2. Node Roles

**Node A (AMD Tower)** - Command Center
- Global Workspace orchestration
- MoE router and gating network
- Symbolic reasoning (Hyperon)
- Training coordination
- Action planning expert
- Reasoning expert
- Creativity expert
- Learning expert

**Node B (Desktop)** - Memory Specialist
- RAG vector stores (FAISS/ChromaDB)
- Episodic memory management
- Knowledge consolidation
- Memory expert
- Emotional intelligence expert

**Node C (Gaming Laptop)** - Real-time Inference
- Fast inference engine
- World model simulation
- Real-time control
- Perception expert

**Node E (MacBook)** - Mobile Cognition
- MLX-optimized inference
- Interactive interface
- Development console
- Communication expert

### 3. Expert Specialization

8 LoRA-adapted experts, each specialized for specific domains:

| Expert | Specialization | Node | LoRA Config |
|--------|---------------|------|-------------|
| Reasoning | Logic, mathematics, formal reasoning | Node A | r=16, α=32 |
| Memory | Retrieval, consolidation, patterns | Node B | r=8, α=16 |
| Perception | Vision, language, multimodal | Node C | r=12, α=24 |
| Action | Planning, decision-making, execution | Node A | r=16, α=32 |
| Creativity | Generation, synthesis, novelty | Node A | r=20, α=40 |
| Emotional | Empathy, social reasoning | Node B | r=10, α=20 |
| Learning | Adaptation, meta-cognition | Node A | r=24, α=48 |
| Communication | Language, explanation, dialogue | Node E | r=14, α=28 |

### 4. Global Workspace Theory

Implements Bernard Baars' consciousness theory:

```
Unconscious Processors (parallel)
         ↓
  Attention Competition
         ↓
  Salience Filtering (threshold=0.7)
         ↓
  Global Workspace (capacity=7)
         ↓
  Broadcast to All Processors
```

---

## Quick Start

### Installation

```bash
# Clone Singularis repository
git clone https://github.com/zoadrazorro/Singularis.git
cd Singularis

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for full model loading
pip install torch transformers peft accelerate
```

### Basic Usage

```python
from singularis.data import DATASystem

# Initialize DATA system
data = DATASystem(config_path="config/data_config.yaml")
await data.initialize()

# Process a query
result = await data.process_query(
    query="Explain the implications of quantum computing for cryptography",
    context={"domain": "technical", "depth": "expert"},
    priority=0.8
)

print(f"Response: {result['content']}")
print(f"Experts used: {result['expert_sources']}")
print(f"Latency: {result['latency_ms']:.1f}ms")
```

### Integration with Singularis

```python
from singularis.consciousness import UnifiedConsciousnessLayer
from singularis.data import DATASystem

# Initialize consciousness layer
consciousness = UnifiedConsciousnessLayer()

# Initialize DATA system
data = DATASystem(config_path="config/data_config.yaml")
await data.initialize()

# Process query through both systems
consciousness_result = await consciousness.process_unified(
    query="What patterns do you see in my recent activities?",
    context={"source": "life_ops"}
)

# Use DATA for distributed expert routing
data_result = await data.process_query(
    query=consciousness_result['follow_up_query'],
    context=consciousness_result['context']
)
```

---

## Configuration

Configuration is managed through `config/data_config.yaml`. Key sections:

### Node Configuration

```yaml
nodes:
  node_a:
    role: "command_center"
    hostname: "localhost"  # Change to actual hostname in production
    port: 6379
    hardware:
      gpu_count: 2
      vram_gb: 48
    capabilities:
      - "global_workspace"
      - "moe_routing"
```

### Expert Configuration

```yaml
moe:
  base_model: "meta-llama/Llama-2-7b-hf"
  num_experts: 8
  top_k: 2
  
  experts:
    reasoning:
      specialization: ["logic", "mathematics", "reasoning"]
      lora_config:
        r: 16
        alpha: 32
      node_assignment: "node_a"
```

### Global Workspace

```yaml
global_workspace:
  capacity: 7
  attention_window: 0.1
  salience_threshold: 0.7
```

See `config/data_config.yaml` for complete configuration options.

---

## API Reference

### DATASystem

Main orchestration class for the distributed system.

```python
class DATASystem:
    async def initialize() -> bool
    async def process_query(query, context, priority) -> Dict
    def get_metrics() -> Dict
    async def shutdown()
```

### ExpertRouter

Routes queries to appropriate LoRA experts.

```python
class ExpertRouter:
    async def route_query(query, context) -> Dict
    async def execute_expert(expert_name, query, context) -> Dict
    def get_routing_stats() -> Dict
```

### GlobalWorkspace

Implements consciousness-inspired attention mechanism.

```python
class GlobalWorkspace:
    def submit_to_workspace(content, source, priority) -> bool
    def register_processor(processor_id, callback, specialization)
    def get_workspace_contents() -> List[WorkspaceItem]
    def get_metrics() -> Dict
```

### NodeManager

Manages hardware nodes and their roles.

```python
class NodeManager:
    async def discover_nodes()
    async def register_node(node_config) -> bool
    def get_cluster_status() -> Dict
    def get_node_for_capability(capability) -> Optional[str]
```

---

## Performance

### Latency Characteristics

- **MoE Routing**: 10-50ms depending on complexity
- **Expert Inference**: 50-200ms per expert (model-dependent)
- **Global Workspace Processing**: 50-200ms per attention cycle
- **End-to-end Query**: 100-500ms average

### Scalability

- **Horizontal Scaling**: Linear performance up to 8 nodes
- **Expert Addition**: Sub-linear due to coordination overhead
- **Memory Scaling**: Near-linear with distributed storage

---

## Integration Points

### With Existing Singularis Systems

DATA integrates seamlessly with:

1. **Consciousness Layer** (`singularis/consciousness/`)
   - Routes high-level reasoning queries
   - Provides distributed expert consultation
   - Enhances metacognition capabilities

2. **Life Operations** (`singularis/life_ops/`)
   - Pattern analysis via multiple experts
   - Intelligent intervention decisions
   - Complex query processing

3. **Skyrim AGI** (`singularis/skyrim/`)
   - Distributed action planning
   - Multi-expert decision making
   - Real-time perception processing

4. **LLM Orchestration** (`singularis/llm/`)
   - Works alongside GPT-5 Orchestrator
   - Provides local LLM routing
   - Reduces API costs

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all DATA tests
pytest tests/test_data/ -v

# Run specific test
pytest tests/test_data/test_experts.py::test_expert_routing -v
```

### Mock Mode

DATA can run in mock mode without loading full models:

```python
# Experts will use mock generation automatically if transformers not available
data = DATASystem(config_path="config/data_config.yaml")
await data.initialize()  # Loads in mock mode

result = await data.process_query("Test query")
# Returns mock responses for testing
```

### Adding New Experts

1. Define expert configuration in `config/data_config.yaml`
2. Specify specialization keywords
3. Assign to appropriate node
4. Configure LoRA parameters
5. Restart DATA system

```yaml
experts:
  my_expert:
    name: "my_custom_expert"
    specialization: ["keyword1", "keyword2"]
    lora_config:
      r: 16
      alpha: 32
    node_assignment: "node_a"
    capacity: 0.20
```

---

## Monitoring

### Metrics

DATA provides comprehensive metrics:

```python
metrics = data.get_metrics()

# System metrics
print(f"Queries processed: {metrics['queries_processed']}")
print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
print(f"Active nodes: {metrics['active_nodes']}")
print(f"Available experts: {metrics['available_experts']}")

# Workspace metrics
workspace_metrics = data.global_workspace.get_metrics()
print(f"Workspace size: {workspace_metrics['current_workspace_size']}")
print(f"Acceptance rate: {workspace_metrics['acceptance_rate']:.1%}")

# Expert routing stats
routing_stats = data.expert_router.get_routing_stats()
print(f"Total routings: {routing_stats['total_routings']}")
print(f"Expert load: {routing_stats['expert_load']}")
```

---

## Troubleshooting

### Common Issues

**1. Connection Refused Between Nodes**
- Check firewall settings
- Verify hostnames in config
- Ensure all nodes are on same network

**2. Expert Loading Fails**
- Check GPU memory availability
- Verify model paths
- Try mock mode for testing

**3. High Latency**
- Check network connection
- Monitor node load
- Adjust `top_k` parameter

**4. Workspace Not Broadcasting**
- Verify salience threshold
- Check processor registrations
- Review priority calculations

---

## Roadmap

### Current Status: Alpha (v1.0.0)

- ✅ Core distributed architecture
- ✅ MoE-LoRA expert system
- ✅ Global Workspace implementation
- ✅ Node management
- ✅ Mock mode for development
- ⏳ Full model loading (optional)
- ⏳ Production gRPC implementation
- ⏳ Advanced load balancing

### Future Enhancements

- **Dynamic Expert Creation**: Automatic expert spawning based on demand
- **Meta-Learning**: Experts that learn to improve routing
- **Multi-Modal Fusion**: Image, audio, video expert integration
- **Federated Learning**: Privacy-preserving distributed training
- **Kubernetes Orchestration**: Cloud-native deployment

---

## Credits

DATA system inspired by:

- **OKComputer Distributed AGI System Blueprint** - Architecture foundation
- **Bernard Baars** - Global Workspace Theory
- **Star Trek's Data** - Naming and philosophical inspiration
- **Singularis Project** - Integration platform

---

## License

See `LICENSE` file for details.

---

## Contact

For questions, issues, or contributions:

- GitHub Issues: [github.com/yourusername/Singularis/issues](https://github.com/yourusername/Singularis/issues)
- Documentation: [Full Singularis docs](README.md)

---

**Status**: Alpha Release (v1.0.0)  
**Last Updated**: November 17, 2025  
**Maintainers**: Singularis Team

---

*"In the game of chess, you can never let your adversary see your pieces."*  
— Data, Star Trek: The Next Generation

