"""BDH PerceptionSynth Nanon implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .nanon_base import BDHNanon


@dataclass
class PerceptionSynthesisResult:
    """Structured output from the perception Nanon."""

    situation_vector: np.ndarray
    affordance_scores: Dict[str, float]
    loop_likelihood: float
    confidence: float
    sigma_snapshot: Dict[str, Any]


class BDHPerceptionSynthNanon(BDHNanon):
    """Compress multi-modal perception into actionable state."""

    def __init__(self, situation_dim: int = 32):
        super().__init__(name="BDH-PerceptionSynth", nanon_type="perception")
        self.situation_dim = situation_dim

    # ------------------------------------------------------------------
    async def process(
        self,
        visual_embedding: Optional[np.ndarray],
        audio_embedding: Optional[np.ndarray],
        text_embedding: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        being_state: Optional[Any] = None,
    ) -> PerceptionSynthesisResult:
        """Fuse embeddings into a compact situation vector."""

        fused_vector = self._fuse_embeddings(visual_embedding, audio_embedding, text_embedding)
        affordances = self._derive_affordances(fused_vector, metadata or {})
        loop_likelihood = float(np.clip(np.std(fused_vector), 0.0, 1.0))
        confidence = float(np.clip(1.0 - (loop_likelihood * 0.5), 0.0, 1.0))

        sigma_payload = {
            "situation_vector": fused_vector.tolist(),
            "affordance_hash": self._hash_affordances(affordances),
            "loop_likelihood": loop_likelihood,
        }
        metrics = {
            "vector": fused_vector.tolist(),
            "dominant_affordance": max(affordances, key=affordances.get, default="none"),
            "loop_likelihood": loop_likelihood,
            "confidence": confidence,
        }
        report = self.build_metric_report("bdh_perception", metrics, extra_sigma=sigma_payload)

        if being_state is not None:
            self.register_with_being_state(being_state, report)

        return PerceptionSynthesisResult(
            situation_vector=fused_vector,
            affordance_scores=affordances,
            loop_likelihood=loop_likelihood,
            confidence=confidence,
            sigma_snapshot=report.sigma_snapshot,
        )

    # ------------------------------------------------------------------
    def _fuse_embeddings(
        self,
        visual_embedding: Optional[np.ndarray],
        audio_embedding: Optional[np.ndarray],
        text_embedding: Optional[np.ndarray],
    ) -> np.ndarray:
        """Create a fixed length vector from available modalities."""

        candidates = [
            emb.astype(np.float32)[: self.situation_dim]
            for emb in (visual_embedding, audio_embedding, text_embedding)
            if emb is not None
        ]
        if not candidates:
            return np.zeros(self.situation_dim, dtype=np.float32)

        stacked = np.vstack([
            np.pad(emb, (0, max(0, self.situation_dim - emb.shape[0])), mode="constant")
            for emb in candidates
        ])
        fused = np.mean(stacked, axis=0)
        norm = np.linalg.norm(fused)
        if norm > 1e-6:
            fused = fused / norm
        return fused.astype(np.float32)

    def _derive_affordances(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Generate action affordance scores."""

        affordances: Dict[str, float] = {}
        candidate_source = metadata.get("affordances")
        if isinstance(candidate_source, dict):
            affordances.update({str(k): float(v) for k, v in candidate_source.items()})
        elif isinstance(candidate_source, (list, tuple)):
            for idx, name in enumerate(candidate_source):
                affordances[str(name)] = float(1.0 - (idx / max(1, len(candidate_source))))

        if not affordances:
            # Deterministic pseudo-affordances using vector bins
            bins = min(4, len(vector))
            for i in range(bins):
                affordances[f"latent_{i}"] = float(np.clip(abs(vector[i]), 0.0, 1.0))

        total = sum(affordances.values())
        if total > 1e-6:
            affordances = {k: v / total for k, v in affordances.items()}
        return affordances

    def _hash_affordances(self, affordances: Dict[str, float]) -> str:
        """Create a stable hash for the affordance distribution."""

        items = ",".join(f"{k}:{v:.3f}" for k, v in sorted(affordances.items()))
        return hashlib.md5(items.encode()).hexdigest()
