"""BDH PolicyHead Nanon implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .nanon_base import BDHNanon


@dataclass
class PolicyProposal:
    """Container for BDH candidate proposals."""

    candidates: List[Dict[str, Any]]
    certainty: float
    expected_utilities: Dict[str, float]
    sigma_snapshot: Dict[str, Any]


class BDHPolicyHead(BDHNanon):
    """Generate enriched action proposals based on BDH state."""

    def __init__(self, max_candidates: int = 4):
        super().__init__(name="BDH-PolicyHead", nanon_type="policy")
        self.max_candidates = max_candidates

    def propose_candidates(
        self,
        situation_vector: np.ndarray,
        affordance_scores: Dict[str, float],
        goals: Optional[Sequence[str]] = None,
        recent_actions: Optional[Iterable[str]] = None,
        being_state: Optional[Any] = None,
    ) -> PolicyProposal:
        """Return structured candidate proposals."""

        scores = dict(affordance_scores)
        if not scores:
            scores = {"explore": 1.0}

        penalties = self._loop_penalties(recent_actions)
        enriched: List[Dict[str, Any]] = []
        expected: Dict[str, float] = {}

        for action, base_score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.max_candidates]:
            risk = penalties.get(action, 0.1)
            goal_bonus = self._goal_bonus(action, goals)
            utility = float(np.clip(base_score + goal_bonus - (risk * 0.3), -1.0, 1.0))
            enriched.append(
                {
                    "action": action,
                    "score": float(base_score),
                    "risk": float(np.clip(risk, 0.0, 1.0)),
                    "goal_bonus": goal_bonus,
                    "expected_utility": utility,
                }
            )
            expected[action] = utility

        certainty = float(np.clip(max(expected.values()) if expected else 0.0, 0.0, 1.0))
        sigma_payload = {
            "recent_actions": list(recent_actions or []),
            "expected": expected,
            "certainty": certainty,
        }
        report = self.build_metric_report(
            "bdh_policy",
            metrics={
                "candidates": enriched,
                "certainty": certainty,
                "top_action": enriched[0]["action"] if enriched else "none",
            },
            extra_sigma=sigma_payload,
        )

        if being_state is not None:
            self.register_with_being_state(being_state, report)

        return PolicyProposal(
            candidates=enriched,
            certainty=certainty,
            expected_utilities=expected,
            sigma_snapshot=report.sigma_snapshot,
        )

    # ------------------------------------------------------------------
    def _loop_penalties(self, recent_actions: Optional[Iterable[str]]) -> Dict[str, float]:
        """Penalise actions that have recently looped."""

        penalties: Dict[str, float] = {}
        if not recent_actions:
            return penalties

        history = list(recent_actions)[-5:]
        for action in set(history):
            count = history.count(action)
            penalties[action] = min(1.0, 0.1 + (count - 1) * 0.2)
        return penalties

    def _goal_bonus(self, action: str, goals: Optional[Sequence[str]]) -> float:
        """Boost actions aligned with current goals."""

        if not goals:
            return 0.0
        action_lower = action.lower()
        for goal in goals:
            if goal and goal.lower() in action_lower:
                return 0.2
        return 0.0
