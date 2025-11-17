"""
DATA - Distributed Abductive Technical Agent
=============================================

A proto-AGI system inspired by Star Trek's Data, implementing distributed
Multi-Agentic MoE-LoRA routing across heterogeneous hardware nodes.

Based on OKComputer Distributed AGI System Blueprint.

Core Components:
- Distributed routing across hardware nodes
- MoE-LoRA expert system with parameter-efficient adapters
- Global Workspace Theory implementation
- gRPC-based communication layer
- Neural-symbolic integration
- Hardware-aware load balancing

Architecture:
    Node A (AMD Tower) - Command Center
    ├── Global Workspace Orchestrator
    ├── MoE Router & Gating Network
    ├── Symbolic Reasoning (OpenCog Hyperon)
    └── Training Coordination
    
    Node B (Desktop) - Memory & Retrieval
    ├── RAG Vector Stores
    ├── Episodic Memory System
    ├── Knowledge Consolidation
    └── Specialist LoRA Agents
    
    Node C (Gaming Laptop) - Real-time Inference
    ├── Fast Inference Engine
    ├── World Model Simulation
    ├── Quantized Model Execution
    └── Action Generation
    
    Node E (MacBook) - Mobile Cognition
    ├── MLX-optimized Inference
    ├── Interactive Interface
    ├── Development Console
    └── Monitoring Dashboard

Usage:
    from singularis.data import DATASystem
    
    # Initialize DATA system
    data = DATASystem(config_path="config/data_config.yaml")
    await data.initialize()
    
    # Process query through distributed system
    result = await data.process_query(
        query="Analyze the implications of quantum computing",
        context={"domain": "technical", "depth": "expert"}
    )
    
    print(f"Expert sources: {result['expert_sources']}")
    print(f"Response: {result['content']}")
"""

from .core import DATASystem, DATAConfig
from .experts import ExpertRouter, LoRAExpert
from .workspace import GlobalWorkspace, WorkspaceItem
from .communication import DistributedCommunicator
from .node_manager import NodeManager, NodeRole

__all__ = [
    "DATASystem",
    "DATAConfig",
    "ExpertRouter",
    "LoRAExpert",
    "GlobalWorkspace",
    "WorkspaceItem",
    "DistributedCommunicator",
    "NodeManager",
    "NodeRole",
]

__version__ = "1.0.0"
__author__ = "Singularis Team"
__description__ = "Distributed Abductive Technical Agent - Proto-AGI System"

