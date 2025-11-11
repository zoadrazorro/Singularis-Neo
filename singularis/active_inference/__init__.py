"""
Active Inference - Phase 6E

Implements Karl Friston's Free Energy Principle.
Agents minimize surprise by building better world models and acting to resolve uncertainty.

Key concepts:
- Free Energy = Surprise + Model Complexity
- Agents act to minimize free energy
- Epistemic value: reduce uncertainty
- Pragmatic value: achieve goals

Philosophical grounding:
- Closely related to Δ𝒞 (coherence increase)
- Both minimize surprise/increase understanding
"""

from .free_energy import FreeEnergyAgent
from .predictive_processing import PredictiveProcessor

__all__ = [
    'FreeEnergyAgent',
    'PredictiveProcessor',
]
