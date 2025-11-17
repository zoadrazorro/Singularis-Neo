# DATA System Architecture

**Distributed Abductive Technical Agent**  
*Proto-AGI System for Singularis*

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [System Components](#system-components)
4. [Node Topology](#node-topology)
5. [Expert Routing](#expert-routing)
6. [Global Workspace](#global-workspace)
7. [Communication Layer](#communication-layer)
8. [Integration Points](#integration-points)
9. [Performance Characteristics](#performance-characteristics)
10. [Future Enhancements](#future-enhancements)

---

## Overview

DATA (Distributed Abductive Technical Agent) is a proto-AGI system that implements distributed Multi-Agentic Mixture of Experts with LoRA tuning (MoE-LoRA) across heterogeneous hardware nodes. The system is inspired by:

- **OKComputer Distributed AGI Blueprint**: Architecture foundation
- **Star Trek's Data**: Philosophical inspiration and naming
- **Global Workspace Theory** (Bernard Baars): Consciousness model
- **MoE-LoRA**: Parameter-efficient expert specialization

### Design Goals

1. **Distributed Intelligence**: Leverage multiple hardware nodes for parallel processing
2. **Expert Specialization**: Route queries to domain-specific LoRA experts
3. **Consciousness-Inspired**: Implement attention mechanisms from cognitive science
4. **Hardware Awareness**: Optimize routing based on node capabilities
5. **Fault Tolerance**: Graceful degradation and automatic failover

---

## Architecture Principles

### 1. Distributed Intelligence

Intelligence emerges from the interaction of specialized components rather than centralized processing:

```
Query → Router → [Expert 1, Expert 2] → Aggregator → Response
         ↓
    Node Selection (hardware-aware)
```

### 2. Specialization and Coordination

Different experts handle different aspects of intelligence:

- **Reasoning Expert**: Logic, mathematics, formal reasoning
- **Memory Expert**: Retrieval, consolidation, pattern matching
- **Perception Expert**: Vision, language, multimodal processing
- **Action Expert**: Planning, decision-making, execution
- **Creativity Expert**: Generation, synthesis, novelty
- **Emotional Expert**: Empathy, social reasoning
- **Learning Expert**: Adaptation, meta-cognition
- **Communication Expert**: Language, explanation, dialogue

### 3. Global Workspace Theory

Implements consciousness-inspired attention mechanism:

```
┌──────────────────────────────────────┐
│   Unconscious Processors (Parallel)  │
│   ┌──────┐ ┌──────┐ ┌──────┐        │
│   │ P1   │ │ P2   │ │ P3   │        │
│   └──┬───┘ └──┬───┘ └──┬───┘        │
│      └────────┼────────┘             │
│               ↓                      │
│    ┌──────────────────┐             │
│    │ Attention Gate   │ (salience)  │
│    └────────┬─────────┘             │
│             ↓                        │
│    ┌──────────────────┐             │
│    │ Workspace (7)    │             │
│    └────────┬─────────┘             │
│             ↓                        │
│    ┌──────────────────┐             │
│    │ Global Broadcast │             │
│    └──────────────────┘             │
└──────────────────────────────────────┘
```

### 4. Parameter-Efficient Fine-Tuning

Uses LoRA (Low-Rank Adaptation) for expert specialization:

- **Small adapter matrices**: Only 0.1-1% of base model parameters
- **Fast training**: Minutes instead of hours
- **Multiple experts**: Same base model, different adapters
- **Efficient memory**: Multiple experts in same VRAM

---

## System Components

### 1. DATASystem (Core Orchestrator)

**File**: `singularis/data/core.py`

Main coordination system that manages:
- Node initialization
- Expert routing
- Global workspace
- Communication layer
- Metrics and monitoring

**Key Methods**:
```python
async def initialize() -> bool
async def process_query(query, context, priority) -> Dict
def get_metrics() -> Dict
async def shutdown()
```

### 2. ExpertRouter (MoE-LoRA)

**File**: `singularis/data/experts.py`

Routes queries to appropriate experts:
- Gating network for expert selection
- Top-k routing (default k=2)
- Load balancing across nodes
- Specialization-aware scoring

**Components**:
- `LoRAExpert`: Individual expert with LoRA adapter
- `GatingNetwork`: Neural network for routing
- `ExpertRouter`: Orchestrates expert selection

### 3. GlobalWorkspace

**File**: `singularis/data/workspace.py`

Implements consciousness-inspired attention:
- Limited capacity (7 items)
- Salience-based filtering
- Attention competition
- Global broadcast system

**Key Features**:
- Processor registration
- Priority queuing
- Attention weights (learned)
- Broadcast history

### 4. DistributedCommunicator

**File**: `singularis/data/communication.py`

Handles inter-node communication:
- Message passing
- Broadcast support
- Heartbeat monitoring
- Health checks

### 5. NodeManager

**File**: `singularis/data/node_manager.py`

Manages hardware nodes:
- Node discovery
- Role assignment
- Health monitoring
- Load balancing
- Capability matching

---

## Node Topology

### Hardware Configuration (Sephirot Cluster)

```
┌────────────────────────────────────────────────────┐
│              Distributed Node Network               │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐     ┌─────────────┐              │
│  │   Node A    │────▶│   Node B    │              │
│  │ (AMD Tower) │     │  (Desktop)  │              │
│  │  Command    │     │   Memory    │              │
│  │   Center    │     │ Specialist  │              │
│  └──────┬──────┘     └─────────────┘              │
│         │                                           │
│         ▼                                           │
│  ┌─────────────┐     ┌─────────────┐              │
│  │   Node C    │────▶│   Node E    │              │
│  │  (Laptop)   │     │  (MacBook)  │              │
│  │  Real-time  │     │   Mobile    │              │
│  │  Inference  │     │  Cognition  │              │
│  └─────────────┘     └─────────────┘              │
└────────────────────────────────────────────────────┘
```

### Node Specifications

#### Node A: AMD Tower (Command Center)
- **Role**: Global orchestration, symbolic reasoning
- **Hardware**: 2x AMD 7900 XT (48GB VRAM), 128GB RAM
- **Experts**: Reasoning, Action, Creativity, Learning
- **Capabilities**: 
  - Global Workspace coordination
  - MoE routing
  - Symbolic reasoning (OpenCog Hyperon)
  - Training coordination

#### Node B: Desktop (Memory Specialist)
- **Role**: Memory management, RAG operations
- **Hardware**: 1x AMD 6900 XT (16GB VRAM), 16GB RAM
- **Experts**: Memory, Emotional Intelligence
- **Capabilities**:
  - RAG vector stores (FAISS)
  - Episodic memory
  - Knowledge consolidation

#### Node C: Gaming Laptop (Real-time Inference)
- **Role**: Fast inference, world simulation
- **Hardware**: 1x NVIDIA RTX (8GB VRAM), 16GB RAM
- **Experts**: Perception
- **Capabilities**:
  - Fast inference
  - World model simulation
  - Real-time control

#### Node E: MacBook (Mobile Cognition)
- **Role**: Mobile inference, interface
- **Hardware**: Apple M3 Pro (18GB unified), 12 cores
- **Experts**: Communication
- **Capabilities**:
  - MLX-optimized inference
  - Interactive interface
  - Development console

---

## Expert Routing

### Routing Algorithm

1. **Specialization Scoring**
   - Calculate keyword match for each expert
   - Weighted by expert capacity

2. **Load Balancing**
   - Adjust scores based on current node load
   - Prevent overloading single node

3. **Top-K Selection**
   - Select top-k experts (default k=2)
   - Apply softmax for routing weights

4. **Execution**
   - Execute query on selected experts (parallel)
   - Aggregate responses with weights

### Routing Example

```python
Query: "Analyze sleep patterns and suggest improvements"

Step 1: Specialization Scores
  - memory_expert: 0.8 (pattern matching, recall)
  - reasoning_expert: 0.6 (analysis, logic)
  - emotional_expert: 0.5 (well-being)
  - action_expert: 0.7 (recommendations)

Step 2: Load Balancing
  - memory_expert: 0.8 * 0.9 = 0.72 (node_b at 10% load)
  - reasoning_expert: 0.6 * 0.8 = 0.48 (node_a at 20% load)
  - emotional_expert: 0.5 * 0.95 = 0.475 (node_b at 5% load)
  - action_expert: 0.7 * 0.85 = 0.595 (node_a at 15% load)

Step 3: Top-2 Selection
  - Selected: memory_expert (0.72), action_expert (0.595)
  - Weights: [0.58, 0.42] (softmax)

Step 4: Aggregation
  - Response = 0.58 * memory_response + 0.42 * action_response
```

---

## Global Workspace

### Attention Mechanism

Based on Bernard Baars' Global Workspace Theory:

1. **Unlimited Unconscious Processing**
   - Multiple processors operate in parallel
   - Each specialized for different functions
   - No capacity limits at this stage

2. **Attention Competition**
   - Items compete for workspace access
   - Based on salience (priority)
   - Winner-take-all dynamics

3. **Limited Workspace**
   - Capacity: 7 items (Miller's magic number)
   - High-priority items replace low-priority
   - FIFO for equal priority

4. **Global Broadcast**
   - Contents broadcast to all processors
   - Creates coherent global state
   - Enables coordination

### Implementation Details

```python
# Workspace configuration
capacity = 7  # Miller's magic number
salience_threshold = 0.7  # Minimum priority for admission
attention_window = 0.1  # seconds
broadcast_interval = 0.05  # seconds

# Attention weights (learned)
attention_weights = {
    'processor_id': weight  # 0.0-1.0
}

# Decay factor
attention_decay = 0.95  # Applied each cycle
```

---

## Communication Layer

### Message Passing

```python
@dataclass
class Message:
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    message_id: str
```

### Communication Patterns

1. **Point-to-Point**
   - Direct node-to-node messages
   - Used for expert execution
   - Latency: ~5-15ms (local network)

2. **Broadcast**
   - One-to-many distribution
   - Used for workspace broadcasts
   - Parallel message delivery

3. **Heartbeat**
   - Periodic health checks
   - Interval: 1 second
   - Timeout: 30 seconds

### Health Monitoring

```python
# Node health criteria
healthy = (
    status in ["active", "degraded"] and
    time_since_heartbeat < 30 and
    current_load < 0.95
)
```

---

## Integration Points

### 1. Consciousness Layer

```python
from singularis.data.singularis_integration import DATAConsciousnessIntegration

integration = DATAConsciousnessIntegration(
    data_system=data,
    consciousness=consciousness,
    enable_fallback=True
)

result = await integration.process_query_hybrid(
    query="Complex reasoning query",
    use_distributed=True
)
```

### 2. Life Operations

```python
from singularis.data.singularis_integration import LifeOpsIntegration

life_ops = LifeOpsIntegration(data_system=data)

result = await life_ops.analyze_patterns_distributed(
    events=life_events,
    query="What patterns indicate stress?"
)
```

### 3. Skyrim AGI

```python
from singularis.data.singularis_integration import SkyrimAGIIntegration

skyrim_agi = SkyrimAGIIntegration(data_system=data)

result = await skyrim_agi.distributed_action_planning(
    game_state=current_state,
    available_actions=actions
)
```

---

## Performance Characteristics

### Latency

| Component | Latency | Notes |
|-----------|---------|-------|
| MoE Routing | 10-50ms | Depends on query complexity |
| Expert Inference | 50-200ms | Model-dependent (mock: 100ms) |
| Workspace Processing | 50-200ms | Per attention cycle |
| Communication | 5-15ms | Local network |
| End-to-End Query | 100-500ms | Average case |

### Throughput

- **Queries per second**: 10-50 (depends on expert load)
- **Workspace broadcasts**: 20/second
- **Node communication**: 100+ messages/second

### Scalability

- **Horizontal**: Linear up to 8 nodes
- **Expert addition**: Sub-linear (coordination overhead)
- **Memory**: Near-linear with distributed storage

---

## Future Enhancements

### Near-term (v1.1)

1. **Production gRPC Implementation**
   - Replace mock communication
   - Add TLS encryption
   - Implement proper service discovery

2. **Full Model Loading**
   - Load actual LoRA models
   - GPU memory optimization
   - Model caching strategies

3. **Advanced Load Balancing**
   - Predictive load balancing
   - Dynamic expert migration
   - Resource-aware routing

### Mid-term (v1.5)

1. **Dynamic Expert Creation**
   - Automatic expert spawning
   - Domain detection
   - On-demand training

2. **Meta-Learning**
   - Learn optimal routing
   - Adapt to query patterns
   - Self-improving system

3. **Multi-Modal Integration**
   - Image expert
   - Audio expert
   - Video expert

### Long-term (v2.0)

1. **Federated Learning**
   - Privacy-preserving training
   - Distributed gradient aggregation
   - Cross-node learning

2. **Kubernetes Orchestration**
   - Cloud-native deployment
   - Auto-scaling
   - Container orchestration

3. **Advanced Consciousness**
   - Deeper GWT implementation
   - Attention learning
   - Meta-cognitive capabilities

---

## References

1. **OKComputer Distributed AGI Blueprint**  
   Foundation for distributed architecture

2. **Baars, B. J. (1988)**  
   "A Cognitive Theory of Consciousness"  
   Cambridge University Press

3. **Hu, E. J., et al. (2021)**  
   "LoRA: Low-Rank Adaptation of Large Language Models"  
   arXiv:2106.09685

4. **Dehaene, S., et al. (2017)**  
   "What is consciousness, and could machines have it?"  
   Science, 358(6362), 486-492

---

**Version**: 1.0.0  
**Last Updated**: November 17, 2025  
**Maintainers**: Singularis Team

