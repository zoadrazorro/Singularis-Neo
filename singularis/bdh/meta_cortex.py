"""BDH MetaCortex decision tier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .nanon_base import BDHNanon


class MetaDecisionStrategy(str, Enum):
    """High level control strategies emitted by the MetaCortex."""

    EXECUTE = "execute"
    REPLAN = "replan"
    DEFER = "defer"
    ESCALATE = "escalate"


@dataclass
class MetaDecision:
    """Decision object returned by the MetaCortex."""

    strategy: MetaDecisionStrategy
    confidence: float
    justification: str
    selected_action: Optional[str] = None
    escalate_reason: Optional[str] = None
    stress: float = 0.0
    sigma_snapshot: Dict[str, Any] = None


class BDHMetaCortex(BDHNanon):
    """Tier-1 reasoning layer before external LLM escalation."""

    def __init__(self):
        super().__init__(name="BDH-MetaCortex", nanon_type="meta")

    def evaluate(
        self,
        being_state: Any,
        candidate_actions: Iterable[Dict[str, Any]],
        escalation_threshold: float = 0.35,
    ) -> MetaDecision:
        """Evaluate whether to execute locally or escalate."""

        candidates = list(candidate_actions)
        temporal_coherence = getattr(being_state, "temporal_coherence", 1.0)
        stuck_loops = getattr(being_state, "stuck_loop_count", 0)
        conflict_flags = getattr(being_state, "action_plan_conflicts", [])

        if not candidates:
            decision = self._emit_decision(
                strategy=MetaDecisionStrategy.REPLAN,
                confidence=0.5,
                justification="No candidate actions available",
                stress=self._compute_stress(temporal_coherence, stuck_loops),
            )
            self._report(being_state, decision, candidates)
            return decision

        top_action, top_utility, spread = self._select_best(candidates)
        stress = self._compute_stress(temporal_coherence, stuck_loops)

        if top_utility < 0.1:
            decision = self._emit_decision(
                strategy=MetaDecisionStrategy.REPLAN,
                confidence=1.0 - top_utility,
                justification="Utilities too low",
                stress=stress,
            )
        elif stress > 0.75 or spread < escalation_threshold or conflict_flags:
            decision = self._emit_decision(
                strategy=MetaDecisionStrategy.ESCALATE,
                confidence=max(0.2, 1.0 - spread),
                justification="High stress or conflicts detected",
                selected_action=top_action,
                escalate_reason="temporal_incoherence" if stress > 0.75 else "conflict",
                stress=stress,
            )
        else:
            decision = self._emit_decision(
                strategy=MetaDecisionStrategy.EXECUTE,
                confidence=max(top_utility, 0.5),
                justification="Policy head confident",
                selected_action=top_action,
                stress=stress,
            )

        self._report(being_state, decision, candidates)
        return decision

    # ------------------------------------------------------------------
    def _select_best(self, candidates: List[Dict[str, Any]]):
        utilities = []
        for candidate in candidates:
            if "expected_utility" in candidate:
                utilities.append(candidate["expected_utility"])
            elif "confidence" in candidate:
                utilities.append(candidate["confidence"])
            else:
                utilities.append(candidate.get("score", 0.0))
        top_idx = int(max(range(len(utilities)), key=utilities.__getitem__))
        top_utility = float(utilities[top_idx])
        if len(utilities) == 1:
            spread = 1.0
        else:
            sorted_utils = sorted(utilities, reverse=True)
            spread = float(sorted_utils[0] - sorted_utils[1])
        return candidates[top_idx].get("action"), top_utility, spread

    def _compute_stress(self, temporal_coherence: float, stuck_loops: int) -> float:
        stress = 0.0
        if temporal_coherence < 0.7:
            stress += (0.7 - temporal_coherence)
        stress += min(1.0, stuck_loops * 0.1)
        return float(max(0.0, min(1.0, stress)))

    def _emit_decision(
        self,
        strategy: MetaDecisionStrategy,
        confidence: float,
        justification: str,
        selected_action: Optional[str] = None,
        escalate_reason: Optional[str] = None,
        stress: float = 0.0,
    ) -> MetaDecision:
        return MetaDecision(
            strategy=strategy,
            confidence=float(max(0.0, min(1.0, confidence))),
            justification=justification,
            selected_action=selected_action,
            escalate_reason=escalate_reason,
            stress=stress,
        )

    def _report(self, being_state: Any, decision: MetaDecision, candidates: List[Dict[str, Any]]):
        sigma_payload = {
            "decision": decision.strategy.value,
            "confidence": decision.confidence,
            "stress": decision.stress,
            "candidate_count": len(candidates),
        }
        report = self.build_metric_report(
            "bdh_meta",
            metrics={
                "strategy": decision.strategy.value,
                "confidence": decision.confidence,
                "stress": decision.stress,
                "selected_action": decision.selected_action,
                "escalate_reason": decision.escalate_reason or "",
            },
            extra_sigma=sigma_payload,
        )
        decision.sigma_snapshot = report.sigma_snapshot
        if being_state is not None:
            self.register_with_being_state(being_state, report)
