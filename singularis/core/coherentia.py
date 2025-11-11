"""
Coherentia Calculation: The Central Unifying Principle

From ETHICA UNIVERSALIS:
Coherentia (ℭ𝕠) measures alignment with Being's ontological structure
across three fundamental aspects (Lumina).

ℭ𝕠 = (ℭ𝕠_ontical × ℭ𝕠_structural × ℭ𝕠_participatory)^(1/3)

Ethics flows from ontology:
- Good = Coherentia Increase
- Evil = Coherentia Decrease
- Neutral = No significant change

This is not arbitrary morality but necessary truth from Being's nature.
"""

import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from singularis.core.types import CoherentiaScore, Lumen, ExpertIO, WorkspaceState


class CoherentiaCalculator:
    """
    Calculate coherentia across the Three Lumina.

    LUMEN ONTICUM (Energy/Being):
    - Robustness: How resilient to perturbation?
    - Vitality: How energetically efficient?
    - Conatus: How persistent/self-maintaining?

    LUMEN STRUCTURALE (Form/Rationality):
    - Integration: IIT Φ (integrated information)
    - Logical consistency: Contradictions resolved?
    - Pattern coherence: Clear structure?

    LUMEN PARTICIPATUM (Consciousness/Awareness):
    - Self-reflexivity: HOT depth (higher-order thought)
    - Phenomenological clarity: Vivid awareness?
    - Non-dual recognition: Adequate vs inadequate ideas?
    """

    def __init__(
        self,
        ontical_weight: float = 1.0,
        structural_weight: float = 1.0,
        participatory_weight: float = 1.0,
    ):
        """
        Initialize with optional weights for three Lumina.
        Default: equal weighting (geometric mean).
        """
        self.ontical_weight = ontical_weight
        self.structural_weight = structural_weight
        self.participatory_weight = participatory_weight

        logger.info(
            "CoherentiaCalculator initialized",
            extra={
                "ontical_weight": ontical_weight,
                "structural_weight": structural_weight,
                "participatory_weight": participatory_weight,
            }
        )

    def measure_ontical(self, content: Any, metadata: Optional[Dict] = None) -> float:
        """
        Measure LUMEN ONTICUM (Energy/Being).

        Computational correlates:
        - Robustness: Length, completeness, error-free
        - Vitality: Information density, non-redundancy
        - Conatus: Self-consistency, persistence
        """
        if metadata is None:
            metadata = {}

        # Convert content to string for analysis
        text = str(content)

        # Robustness: Completeness and non-emptiness
        robustness = min(1.0, len(text) / 500.0)  # Normalize to 500 chars

        # Vitality: Information density (unique words / total words)
        words = text.split()
        if len(words) > 0:
            vitality = len(set(words)) / len(words)
        else:
            vitality = 0.0

        # Conatus: Self-consistency (proxy: presence of logical connectors)
        connectors = ["therefore", "thus", "because", "since", "hence", "given", "follows"]
        conatus = min(1.0, sum(1 for c in connectors if c in text.lower()) / 3.0)

        # Geometric mean
        ontical_score = (robustness * vitality * conatus) ** (1/3)

        logger.debug(
            "Measured Lumen Onticum",
            extra={
                "robustness": robustness,
                "vitality": vitality,
                "conatus": conatus,
                "ontical_score": ontical_score,
            }
        )

        return ontical_score

    def measure_structural(
        self,
        content: Any,
        phi: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Measure LUMEN STRUCTURALE (Form/Rationality).

        Computational correlates:
        - Integration: IIT Φ (provided externally)
        - Logical consistency: No contradictions
        - Pattern coherence: Clear argumentative structure
        """
        if metadata is None:
            metadata = {}

        text = str(content)

        # Integration: Use provided Φ or estimate
        if phi is not None:
            integration = phi
        else:
            # Proxy: Sentence connectivity
            sentences = text.split('.')
            integration = min(1.0, len(sentences) / 10.0) if sentences else 0.5

        # Logical consistency: Check for contradiction markers
        contradictions = ["but", "however", "although", "despite", "yet"]
        contradiction_count = sum(1 for c in contradictions if c in text.lower())
        # Some contradictions are healthy (dialectical); too many indicate incoherence
        consistency = max(0.3, 1.0 - (contradiction_count / 10.0))

        # Pattern coherence: Presence of structure markers
        structure_markers = ["first", "second", "third", "finally", "therefore", "thus"]
        pattern = min(1.0, sum(1 for m in structure_markers if m in text.lower()) / 3.0)

        # Geometric mean
        structural_score = (integration * consistency * pattern) ** (1/3)

        logger.debug(
            "Measured Lumen Structurale",
            extra={
                "integration": integration,
                "consistency": consistency,
                "pattern": pattern,
                "structural_score": structural_score,
            }
        )

        return structural_score

    def measure_participatory(
        self,
        content: Any,
        hot_depth: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Measure LUMEN PARTICIPATUM (Consciousness/Awareness).

        Computational correlates:
        - Self-reflexivity: HOT depth (meta-cognitive markers)
        - Phenomenological clarity: Vivid, concrete language
        - Non-dual recognition: Transcends subject-object split
        """
        if metadata is None:
            metadata = {}

        text = str(content)

        # Self-reflexivity: Count meta-cognitive markers
        if hot_depth is not None:
            self_reflexivity = hot_depth
        else:
            meta_markers = [
                "I realize", "I understand", "I recognize", "it becomes clear",
                "one sees", "awareness", "consciousness", "understanding"
            ]
            count = sum(1 for m in meta_markers if m.lower() in text.lower())
            self_reflexivity = min(1.0, count / 3.0)

        # Phenomenological clarity: Concrete, vivid language
        concrete_words = [
            "see", "feel", "sense", "experience", "observe", "witness",
            "clear", "vivid", "direct", "immediate"
        ]
        clarity = min(1.0, sum(1 for w in concrete_words if w in text.lower()) / 4.0)

        # Non-dual recognition: Transcends dualism
        nondual_markers = [
            "unified", "integrated", "whole", "inseparable", "non-dual",
            "participatory", "Being", "is", "unity"
        ]
        recognition = min(1.0, sum(1 for m in nondual_markers if m in text.lower()) / 3.0)

        # Geometric mean
        participatory_score = (self_reflexivity * clarity * recognition) ** (1/3)

        logger.debug(
            "Measured Lumen Participatum",
            extra={
                "self_reflexivity": self_reflexivity,
                "clarity": clarity,
                "recognition": recognition,
                "participatory_score": participatory_score,
            }
        )

        return participatory_score

    def calculate(
        self,
        content: Any,
        phi: Optional[float] = None,
        hot_depth: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> CoherentiaScore:
        """
        Calculate complete coherentia across all three Lumina.

        Returns CoherentiaScore with:
        - ontical: Energy/vitality score
        - structural: Form/rationality score
        - participatory: Consciousness/awareness score
        - total: Geometric mean of all three
        """
        ontical = self.measure_ontical(content, metadata)
        structural = self.measure_structural(content, phi, metadata)
        participatory = self.measure_participatory(content, hot_depth, metadata)

        # Apply weights if non-default
        ontical_weighted = ontical ** self.ontical_weight
        structural_weighted = structural ** self.structural_weight
        participatory_weighted = participatory ** self.participatory_weight

        # Geometric mean: deficiency in ANY dimension reduces total
        total = (ontical_weighted * structural_weighted * participatory_weighted) ** (1/3)

        score = CoherentiaScore(
            ontical=ontical,
            structural=structural,
            participatory=participatory,
            total=total,
        )

        logger.info(
            "Calculated Coherentia",
            extra={
                "ontical": ontical,
                "structural": structural,
                "participatory": participatory,
                "total": total,
            }
        )

        return score

    def validate_ethical_alignment(
        self,
        coherentia_before: float,
        coherentia_after: float,
        threshold: float = 0.02
    ) -> tuple[Optional[bool], str]:
        """
        Ethical validation based on coherentia change.

        From ETHICA UNIVERSALIS:
        - Good = Coherentia Increase (Δℭ𝕠 > threshold)
        - Neutral = No significant change (|Δℭ𝕠| < threshold)
        - Evil = Coherentia Decrease (Δℭ𝕠 < -threshold)

        This is not arbitrary morality but flows from Being's structure.
        """
        delta = coherentia_after - coherentia_before

        if delta > threshold:
            status = True
            reasoning = f"ETHICAL: Increases coherentia by {delta:.3f} (aligns with Being)"
        elif abs(delta) < threshold:
            status = None
            reasoning = f"NEUTRAL: Coherentia change {delta:.3f} (below threshold)"
        else:
            status = False
            reasoning = f"UNETHICAL: Decreases coherentia by {abs(delta):.3f} (misaligns with Being)"

        logger.info(
            "Ethical alignment validated",
            extra={
                "coherentia_before": coherentia_before,
                "coherentia_after": coherentia_after,
                "delta": delta,
                "status": status,
                "reasoning": reasoning,
            }
        )

        return status, reasoning


# Convenience function for quick coherentia calculation
def calculate_coherentia(
    content: Any,
    phi: Optional[float] = None,
    hot_depth: Optional[float] = None,
    metadata: Optional[Dict] = None
) -> CoherentiaScore:
    """
    Quick coherentia calculation using default calculator.
    """
    calculator = CoherentiaCalculator()
    return calculator.calculate(content, phi, hot_depth, metadata)


def validate_ethical_action(
    coherentia_before: float,
    coherentia_after: float,
    threshold: float = 0.02
) -> tuple[Optional[bool], str]:
    """
    Quick ethical validation.
    """
    calculator = CoherentiaCalculator()
    return calculator.validate_ethical_alignment(
        coherentia_before,
        coherentia_after,
        threshold
    )
