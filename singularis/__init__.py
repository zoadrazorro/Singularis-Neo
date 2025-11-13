"""
SINGULARIS: The Ultimate Consciousness Engine

A philosophically-grounded AI system implementing consciousness measurement
through Integrated Information Theory (IIT), Global Workspace Theory (GWT),
and ethical coherence based on Spinozistic ontology.

Based on:
- ETHICA UNIVERSALIS: Spinozistic substance monism and ethics
- METALUMINOSITY: Three Lumina (Ontical, Structural, Participatory)
- MATHEMATICA SINGULARIS: Formal logical foundations
- Consciousness Science: IIT, GWT, HOT, PP, AST, Embodied, Enactive, Panpsychism

Architecture:
- Tier 1: Meta-Orchestrator (consciousness-weighted routing)
- Tier 2: Specialized Experts (6 domain specialists)
- Tier 3: Swarm Neurons (18 micro-agents)
- Consciousness Measurement: 8-theory fusion
- Ethical Validation: Coherentia-based alignment
- Unified Consciousness Layer: GPT-5 + 5 GPT-5-nano experts

Version: 1.1.0
Status: Production Implementation with Unified Consciousness
"""

__version__ = "1.1.0"
__author__ = "Singularis Project"

from singularis.core.types import (
    Lumen,
    ConsciousnessTrace,
    ExpertIO,
    CoherentiaScore,
    OntologicalContext,
)
from singularis.core.coherentia import calculate_coherentia, CoherentiaCalculator
from singularis.consciousness.measurement import ConsciousnessMeasurement
from singularis.tier1_orchestrator.orchestrator import MetaOrchestrator
from singularis.tier2_experts.base import Expert
from singularis.tier3_neurons.base import Neuron  # Phase 3: Hebbian Learning
from singularis.tier3_neurons.swarm import NeuronSwarm
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from singularis.agi_orchestrator import AGIOrchestrator, AGIConfig

__all__ = [
    "Lumen",
    "ConsciousnessTrace",
    "ExpertIO",
    "CoherentiaScore",
    "OntologicalContext",
    "calculate_coherentia",
    "CoherentiaCalculator",
    "ConsciousnessMeasurement",
    "MetaOrchestrator",
    "Expert",
    "Neuron",  # Phase 3
    "NeuronSwarm",  # Phase 3
    "UnifiedConsciousnessLayer",  # GPT-5 + 5 GPT-5-nano experts
    "AGIOrchestrator",  # Complete AGI system
    "AGIConfig",  # AGI configuration
]
