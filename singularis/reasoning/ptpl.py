"""
Phase-Temporal Participatory Logic (PTPL)

A 4D, state-aware logic scaffold that combines:
- Phase logic: gates modulated by OMEGA phase (integration, temporal, causal, predictive)
- Temporal logic: soft scoring over short horizon
- Participatory logic: subject-in-system weighting (embodiment tension, relevance)

This is a lightweight scaffold providing scores and priorities. Concrete rule
encodings can be built on top of this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import time
import math


@dataclass
class PTPLResult:
    score: float
    priority: float
    rationale: str


class PTPL:
    def __init__(self, omega=None):
        self.omega = omega

    def _get_phase(self) -> Dict[str, float]:
        if self.omega and hasattr(self.omega, "get_phase_state"):
            return self.omega.get_phase_state()
        # Neutral defaults
        return dict(integration=0.6, temporal=0.6, causal=0.6, predictive=0.6)

    def evaluate(self, belief_delta: float, participatory_tension: float, temporal_window: float = 10.0) -> PTPLResult:
        """
        Evaluate a generic PTPL rule, returning a scored priority.
        
        Args:
            belief_delta: change in confidence for a proposition (0..1)
            participatory_tension: embodiment/self-relevance (0..1)
            temporal_window: seconds until expected outcome
        """
        phase = self._get_phase()
        # Phase gates: emphasize causal when high; temporal for short window
        causal_gate = phase["causal"]
        temporal_gate = phase["temporal"] * (1.0 - 1.0 / (1.0 + temporal_window))
        integration_gate = phase["integration"]
        predictive_gate = phase["predictive"]

        # Score composition
        score = (
            0.35 * belief_delta * causal_gate +
            0.25 * participatory_tension * integration_gate +
            0.20 * temporal_gate +
            0.20 * predictive_gate
        )
        score = max(0.0, min(1.0, score))

        # Priority scaling (softplus-like)
        priority = math.log1p(math.e ** (3.0 * score)) / 4.0
        priority = max(0.0, min(1.0, priority))

        rationale = (
            f"PTPL: beliefΔ={belief_delta:.2f}, part={participatory_tension:.2f}, "
            f"phase(c={causal_gate:.2f},t={temporal_gate:.2f},i={integration_gate:.2f},p={predictive_gate:.2f})"
        )
        return PTPLResult(score=score, priority=priority, rationale=rationale)
