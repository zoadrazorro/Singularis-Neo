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
    Calculates the Coherentia score, a measure of alignment with Being's
    ontological structure across three fundamental aspects (Lumina).

    The three Lumina are:
    - LUMEN ONTICUM (Energy/Being): Measures robustness, vitality, and persistence.
    - LUMEN STRUCTURALE (Form/Rationality): Measures integration, consistency, and structure.
    - LUMEN PARTICIPATUM (Consciousness/Awareness): Measures self-reflexivity, clarity,
      and non-dual recognition.

    The total Coherentia is the geometric mean of the scores from the three Lumina.
    """

    def __init__(
        self,
        ontical_weight: float = 1.0,
        structural_weight: float = 1.0,
        participatory_weight: float = 1.0,
    ):
        """
        Initializes the CoherentiaCalculator with optional weights for the three Lumina.

        The weights act as exponents in the geometric mean calculation, allowing for
        different aspects of Coherentia to be emphasized.

        Args:
            ontical_weight (float, optional): The weight for the ontical Lumen. Defaults to 1.0.
            structural_weight (float, optional): The weight for the structural Lumen. Defaults to 1.0.
            participatory_weight (float, optional): The weight for the participatory Lumen. Defaults to 1.0.
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
        Measures the ontical Lumen (Energy/Being).

        This score is calculated based on computational correlates of:
        - Robustness: The length and completeness of the content.
        - Vitality: The information density, measured as the ratio of unique words to total words.
        - Conatus: Self-consistency, proxied by the presence of logical connectors.

        Args:
            content (Any): The content to be analyzed, which will be converted to a string.
            metadata (Optional[Dict], optional): Additional metadata (not currently used). Defaults to None.

        Returns:
            float: The ontical score, in the range [0, 1].
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
            vitality = 0.3  # Minimum vitality for empty/minimal content

        # Conatus: Self-consistency (proxy: presence of logical connectors)
        connectors = ["therefore", "thus", "because", "since", "hence", "given", "follows"]
        conatus = max(0.2, min(1.0, sum(1 for c in connectors if c in text.lower()) / 3.0))

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
        Measures the structural Lumen (Form/Rationality).

        This score is based on:
        - Integration: The integrated information (Φ), if provided, otherwise estimated from sentence connectivity.
        - Logical consistency: The absence of contradiction markers.
        - Pattern coherence: The presence of structural markers.

        Args:
            content (Any): The content to be analyzed, converted to a string.
            phi (Optional[float], optional): The integrated information score (Φ). If not provided, it is estimated. Defaults to None.
            metadata (Optional[Dict], optional): Additional metadata (not currently used). Defaults to None.

        Returns:
            float: The structural score, in the range [0, 1].
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
        pattern = max(0.2, min(1.0, sum(1 for m in structure_markers if m in text.lower()) / 3.0))

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
        Measures the participatory Lumen (Consciousness/Awareness).

        This score is based on:
        - Self-reflexivity: Higher-order thought (HOT) depth, if provided, otherwise estimated from meta-cognitive markers.
        - Phenomenological clarity: The presence of vivid and concrete language.
        - Non-dual recognition: The presence of markers that transcend dualism.

        Args:
            content (Any): The content to be analyzed, converted to a string.
            hot_depth (Optional[float], optional): The higher-order thought depth. If not provided, it is estimated. Defaults to None.
            metadata (Optional[Dict], optional): Additional metadata (not currently used). Defaults to None.

        Returns:
            float: The participatory score, in the range [0, 1].
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
            self_reflexivity = max(0.2, min(1.0, count / 3.0))

        # Phenomenological clarity: Concrete, vivid language
        concrete_words = [
            "see", "feel", "sense", "experience", "observe", "witness",
            "clear", "vivid", "direct", "immediate"
        ]
        clarity = max(0.2, min(1.0, sum(1 for w in concrete_words if w in text.lower()) / 4.0))

        # Non-dual recognition: Transcends dualism
        nondual_markers = [
            "unified", "integrated", "whole", "inseparable", "non-dual",
            "participatory", "Being", "is", "unity"
        ]
        recognition = max(0.2, min(1.0, sum(1 for m in nondual_markers if m in text.lower()) / 3.0))

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
        Calculates the complete Coherentia score across all three Lumina.

        This method computes the ontical, structural, and participatory scores and
        then combines them into a total score using a weighted geometric mean.

        Args:
            content (Any): The content to be analyzed.
            phi (Optional[float], optional): The integrated information score (Φ) for the structural calculation. Defaults to None.
            hot_depth (Optional[float], optional): The higher-order thought depth for the participatory calculation. Defaults to None.
            metadata (Optional[Dict], optional): Additional metadata. Defaults to None.

        Returns:
            CoherentiaScore: A dataclass containing the ontical, structural, participatory, and total scores.
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
        Validates the ethical alignment of an action based on the change in Coherentia.

        According to the principles of ETHICA UNIVERSALIS, an action is considered:
        - Good (True) if it increases Coherentia by more than the threshold.
        - Evil (False) if it decreases Coherentia by more than the threshold.
        - Neutral (None) if the change is within the threshold.

        Args:
            coherentia_before (float): The Coherentia score before the action.
            coherentia_after (float): The Coherentia score after the action.
            threshold (float, optional): The significance threshold for the change. Defaults to 0.02.

        Returns:
            tuple[Optional[bool], str]: A tuple containing the ethical status (True, False, or None)
                                       and a string explaining the reasoning.
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
    A convenience function for a quick Coherentia calculation using a default calculator.

    This function instantiates a `CoherentiaCalculator` with default weights and
    returns the `CoherentiaScore`.

    Args:
        content (Any): The content to be analyzed.
        phi (Optional[float], optional): The integrated information score (Φ). Defaults to None.
        hot_depth (Optional[float], optional): The higher-order thought depth. Defaults to None.
        metadata (Optional[Dict], optional): Additional metadata. Defaults to None.

    Returns:
        CoherentiaScore: The calculated Coherentia score.
    """
    calculator = CoherentiaCalculator()
    return calculator.calculate(content, phi, hot_depth, metadata)


def validate_ethical_action(
    coherentia_before: float,
    coherentia_after: float,
    threshold: float = 0.02
) -> tuple[Optional[bool], str]:
    """
    A convenience function for a quick ethical validation of an action.

    This function instantiates a `CoherentiaCalculator` and uses it to
    validate the ethical alignment of an action based on the change in Coherentia.

    Args:
        coherentia_before (float): The Coherentia score before the action.
        coherentia_after (float): The Coherentia score after the action.
        threshold (float, optional): The significance threshold for the change. Defaults to 0.02.

    Returns:
        tuple[Optional[bool], str]: A tuple containing the ethical status and a reasoning string.
    """
    calculator = CoherentiaCalculator()
    return calculator.validate_ethical_alignment(
        coherentia_before,
        coherentia_after,
        threshold
    )
