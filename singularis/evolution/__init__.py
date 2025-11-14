"""
Evolution Module

Implements evolutionary learning systems:
- Darwinian Modal Logic (Gemini Flash 2.0)
- Analytic Evolution Heuristic (Claude Haiku)
- Double-Helix Architecture with Self-Improvement Gating
"""

from .darwinian_modal_logic import (
    DarwinianModalLogic,
    PossibleWorld,
    ModalProposition,
    ModalOperator
)
from .analytic_evolution import (
    AnalyticEvolution,
    AnalyticNode,
    EvolutionaryTrajectory
)
from .double_helix_architecture import (
    DoubleHelixArchitecture,
    SystemNode,
    SystemStrand
)
from .omega_hyperhelix import OmegaHyperhelix

__all__ = [
    'DarwinianModalLogic',
    'PossibleWorld',
    'ModalProposition',
    'ModalOperator',
    'AnalyticEvolution',
    'AnalyticNode',
    'EvolutionaryTrajectory',
    'DoubleHelixArchitecture',
    'SystemNode',
    'SystemStrand',
    'OmegaHyperhelix'
]
