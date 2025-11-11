"""
World Model Layer - Phase 6A

Causal world model that understands WHY things happen, not just patterns.
Integrates:
- Causal inference (Pearl's causality)
- Vision grounding (CLIP)
- Physics simulation (PyBullet)
- Multimodal integration

Philosophical grounding:
- ETHICA Part II: Mind-body unity requires sensorimotor grounding
- MATHEMATICA A3: All that is, is necessary (causal necessity)
- Conatus (ℭ): Striving requires understanding intervention outcomes
"""

from .causal_graph import CausalGraph, CausalNode, CausalEdge, Intervention
from .vision_module import VisionModule
from .physics_engine import PhysicsEngine
from .world_model_orchestrator import WorldModelOrchestrator, WorldState

__all__ = [
    'CausalGraph',
    'CausalNode',
    'CausalEdge',
    'Intervention',
    'VisionModule',
    'PhysicsEngine',
    'WorldModelOrchestrator',
    'WorldState',
]
