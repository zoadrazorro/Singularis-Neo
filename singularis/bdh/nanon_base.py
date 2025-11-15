"""Base utilities for BDH "Nanon" modules."""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


@dataclass
class SigmaState:
    """Fast-weight memory container for a Nanon."""

    weights: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None

    def age(self) -> float:
        """Return seconds since the sigma state was last updated."""

        return time.time() - self.timestamp


@dataclass
class BDHMetricReport:
    """Bundle emitted from a Nanon to the BeingState."""

    subsystem: str
    metrics: Dict[str, Any]
    sigma_snapshot: Dict[str, Any]


class BDHNanon:
    """Common behaviour for BDH Nanons."""

    def __init__(self, name: str, nanon_type: str, history_size: int = 32):
        self.name = name
        self.nanon_type = nanon_type
        self.sigma_state = SigmaState()
        self._sigma_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Sigma memory helpers
    # ------------------------------------------------------------------
    def snapshot_sigma(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Capture a read-only snapshot of the current sigma state."""

        payload = {
            "trace_id": self.sigma_state.trace_id or f"{self.name}-{time.time():.6f}",
            "timestamp": time.time(),
            "weights": copy.deepcopy(self.sigma_state.weights),
        }
        if extra:
            payload["weights"].update(copy.deepcopy(extra))
        self._sigma_history.append(payload)
        self.sigma_state.timestamp = payload["timestamp"]
        self.sigma_state.trace_id = payload["trace_id"]
        return copy.deepcopy(payload)

    def restore_sigma(self, snapshot: Dict[str, Any]):
        """Restore sigma weights from a snapshot."""

        self.sigma_state.weights = copy.deepcopy(snapshot.get("weights", {}))
        self.sigma_state.timestamp = snapshot.get("timestamp", time.time())
        self.sigma_state.trace_id = snapshot.get("trace_id")

    def reset_sigma(self):
        """Reset sigma weights and history."""

        self.sigma_state = SigmaState()
        self._sigma_history.clear()

    def get_sigma_history(self) -> Deque[Dict[str, Any]]:
        """Expose recent sigma snapshots (read-only)."""

        return copy.deepcopy(self._sigma_history)

    # ------------------------------------------------------------------
    # Being state helpers
    # ------------------------------------------------------------------
    def build_metric_report(
        self,
        subsystem: str,
        metrics: Dict[str, Any],
        extra_sigma: Optional[Dict[str, Any]] = None,
    ) -> BDHMetricReport:
        """Return a metric report capturing current sigma information."""

        sigma_snapshot = self.snapshot_sigma(extra=extra_sigma)
        return BDHMetricReport(subsystem=subsystem, metrics=metrics, sigma_snapshot=sigma_snapshot)

    def register_with_being_state(self, being_state: Any, report: BDHMetricReport):
        """Push metrics into the unified BeingState."""

        if hasattr(being_state, "update_subsystem"):
            payload = dict(report.metrics)
            payload["sigma_age"] = self.sigma_state.age()
            payload["sigma_trace_id"] = report.sigma_snapshot.get("trace_id")
            being_state.update_subsystem(report.subsystem, payload)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def configure(self, **kwargs: Any):
        """Optional configuration hook for subclasses."""

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name!r}, type={self.nanon_type!r})"
