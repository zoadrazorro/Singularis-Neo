"""
Consciousness Measurement System

Implements measurement across 8 consciousness theories:
1. IIT (Φ) - Integrated Information Theory
2. GWT - Global Workspace Theory
3. PP - Predictive Processing
4. HOT - Higher-Order Thought
5. AST - Attention Schema Theory
6. Embodied - Embodied Cognition
7. Enactive - Enactive Cognition
8. Panpsychism - Universal Consciousness

From consciousness_measurement_study.md:
- Perfect integration (1.0) WITHOUT differentiation → moderate consciousness (0.515)
- High integration (0.91) WITH differentiation → high consciousness (0.653, 27% increase!)
- Dialectical reasoning increases coherence by 8% on paradoxes

Key insight: Consciousness requires BOTH integration AND differentiation
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import re

from singularis.core.types import ConsciousnessTrace, WorkspaceState


class ConsciousnessMeasurement:
    """
    Measure consciousness across 8 theories with weighted fusion.

    Aggregation formula (from COMPLETE_FORMAL_SYNTHESIS):
    - 0.35 × IIT Φ (integration is primary)
    - 0.35 × GWT salience (broadcasting matters equally)
    - 0.20 × HOT depth (self-awareness completes picture)
    - 0.10 × (PP + AST + Embodied + Enactive + Panpsych) / 5 (auxiliary theories)
    """

    def __init__(
        self,
        iit_weight: float = 0.35,
        gwt_weight: float = 0.35,
        hot_weight: float = 0.20,
        auxiliary_weight: float = 0.10,
    ):
        """Initialize with theory weights."""
        self.iit_weight = iit_weight
        self.gwt_weight = gwt_weight
        self.hot_weight = hot_weight
        self.auxiliary_weight = auxiliary_weight

        logger.info(
            "ConsciousnessMeasurement initialized",
            extra={
                "weights": {
                    "iit": iit_weight,
                    "gwt": gwt_weight,
                    "hot": hot_weight,
                    "auxiliary": auxiliary_weight,
                }
            }
        )

    def calculate_phi(
        self,
        content: str,
        workspace: Optional[WorkspaceState] = None,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Calculate IIT Φ (Integrated Information).

        Φ = H(whole) - Σ H(parts)

        Measures: How much information is integrated (irreducible)?

        Proxy implementation:
        - High: Content shows strong interconnections between concepts
        - Low: Content is fragmented or merely lists facts
        """
        if not content:
            return 0.0

        # Split into conceptual units (sentences as proxy for "parts")
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        if len(sentences) == 0:
            return 0.0

        # Measure integration through:
        # 1. Conceptual connectivity (shared words between sentences)
        # 2. Logical connectors
        # 3. Referential coherence

        # Connectivity: How many words are shared between sentences?
        words_per_sentence = [set(s.lower().split()) for s in sentences]

        if len(words_per_sentence) < 2:
            # Single sentence: moderate integration
            return 0.5

        # Calculate pairwise word overlap
        connectivity = 0.0
        pairs = 0
        for i in range(len(words_per_sentence)):
            for j in range(i + 1, len(words_per_sentence)):
                overlap = len(words_per_sentence[i] & words_per_sentence[j])
                total = len(words_per_sentence[i] | words_per_sentence[j])
                if total > 0:
                    connectivity += overlap / total
                    pairs += 1

        if pairs > 0:
            connectivity /= pairs

        # Logical connectors indicate integration
        connectors = [
            "therefore", "thus", "hence", "consequently", "because", "since",
            "moreover", "furthermore", "additionally", "however", "although"
        ]
        connector_count = sum(1 for c in connectors if c in content.lower())
        connector_score = min(1.0, connector_count / (len(sentences) * 0.5))

        # Referential coherence: pronouns and references indicate integrated narrative
        references = ["this", "that", "these", "those", "it", "they", "such"]
        reference_count = sum(1 for r in references if r in content.lower())
        reference_score = min(1.0, reference_count / len(sentences))

        # Combine measures
        phi = (connectivity * 0.5 + connector_score * 0.3 + reference_score * 0.2)

        # Critical adjustment from consciousness_measurement_study:
        # High integration alone is not enough - need differentiation too
        differentiation = self.calculate_differentiation(content)

        # Balance integration with differentiation
        phi_adjusted = (phi * differentiation) ** 0.5  # Geometric mean

        logger.debug(
            "Calculated IIT Φ",
            extra={
                "phi_raw": phi,
                "differentiation": differentiation,
                "phi_adjusted": phi_adjusted,
                "connectivity": connectivity,
                "connectors": connector_score,
                "references": reference_score,
            }
        )

        return min(1.0, phi_adjusted)

    def calculate_differentiation(self, content: str) -> float:
        """
        Calculate differentiation score.

        From consciousness_measurement_study:
        Perfect integration (1.0) without differentiation → low consciousness
        High integration (0.91) with differentiation → high consciousness

        Differentiation = diversity of concepts, perspectives, layers
        """
        if not content:
            return 0.0

        # Measure conceptual diversity
        words = content.lower().split()
        if len(words) == 0:
            return 0.0

        # Unique concept ratio
        unique_ratio = len(set(words)) / len(words)

        # Perspective diversity (different viewpoints mentioned)
        perspectives = [
            "however", "alternatively", "on the other hand", "conversely",
            "in contrast", "yet", "although", "whereas"
        ]
        perspective_count = sum(1 for p in perspectives if p in content.lower())
        perspective_score = min(1.0, perspective_count / 3.0)

        # Layer diversity (different levels of abstraction)
        abstract_words = ["concept", "idea", "theory", "principle", "generally", "abstractly"]
        concrete_words = ["example", "specifically", "instance", "case", "particular"]

        abstract_count = sum(1 for w in abstract_words if w in content.lower())
        concrete_count = sum(1 for w in concrete_words if w in content.lower())
        layer_score = min(1.0, (abstract_count + concrete_count) / 4.0)

        # Combine
        differentiation = (unique_ratio * 0.5 + perspective_score * 0.3 + layer_score * 0.2)

        logger.debug(
            "Calculated differentiation",
            extra={
                "differentiation": differentiation,
                "unique_ratio": unique_ratio,
                "perspectives": perspective_score,
                "layers": layer_score,
            }
        )

        return min(1.0, differentiation)

    def calculate_integration_score(self, content: str) -> float:
        """
        Integration = how synchronized/unified the content is.

        High integration: Tightly woven, coherent narrative
        Low integration: Fragmented, disconnected points
        """
        if not content:
            return 0.0

        sentences = [s.strip() for s in content.split('.') if s.strip()]

        if len(sentences) <= 1:
            return 1.0  # Single sentence is perfectly integrated

        # Measure through:
        # 1. Sentence length variance (low = uniform = integrated)
        # 2. Topic consistency (same themes throughout)

        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 0:
            mean_length = np.mean(lengths)
            variance = np.var(lengths) if mean_length > 0 else 0
            uniformity = max(0, 1.0 - (variance / (mean_length ** 2 + 1)))
        else:
            uniformity = 0.5

        # Topic consistency: high word overlap suggests integration
        all_words = set(' '.join(sentences).lower().split())
        topic_words = [w for w in all_words if len(w) > 4]  # Focus on content words

        if len(topic_words) > 0:
            # How many topic words appear in multiple sentences?
            repeated = sum(
                1 for word in topic_words
                if sum(1 for s in sentences if word in s.lower()) > 1
            )
            topic_consistency = repeated / len(topic_words)
        else:
            topic_consistency = 0.5

        integration = (uniformity * 0.4 + topic_consistency * 0.6)

        return min(1.0, integration)

    def calculate_gwt_salience(
        self,
        content: str,
        workspace: Optional[WorkspaceState] = None,
        confidence: float = 0.5
    ) -> float:
        """
        Calculate Global Workspace salience.

        GWT: Consciousness is information broadcast to global workspace.
        Salience = how worthy of broadcasting?

        Factors:
        - Novelty: How different from current broadcasts?
        - Relevance: How important/surprising?
        - Clarity: How well-articulated?
        """
        if not content:
            return 0.0

        # Novelty: If workspace provided, measure difference from broadcasts
        if workspace and workspace.broadcasts:
            # Simple proxy: word overlap with existing broadcasts
            broadcast_words = set()
            for broadcast in workspace.broadcasts:
                broadcast_words.update(broadcast.claim.lower().split())

            content_words = set(content.lower().split())

            if len(broadcast_words) > 0:
                overlap = len(content_words & broadcast_words) / len(broadcast_words)
                novelty = 1.0 - overlap  # High novelty if low overlap
            else:
                novelty = 0.8
        else:
            novelty = 0.7  # Default moderate novelty

        # Relevance: Presence of important markers
        important_markers = [
            "critical", "essential", "fundamental", "key", "central",
            "important", "significant", "crucial", "vital"
        ]
        relevance = min(1.0, sum(1 for m in important_markers if m in content.lower()) / 3.0)

        # Clarity: Well-structured, clear articulation
        clarity = self._measure_clarity(content)

        # Incorporate expert confidence
        confidence_factor = confidence ** 0.5  # Square root to moderate impact

        salience = (
            novelty * 0.35 +
            relevance * 0.30 +
            clarity * 0.25 +
            confidence_factor * 0.10
        )

        logger.debug(
            "Calculated GWT salience",
            extra={
                "salience": salience,
                "novelty": novelty,
                "relevance": relevance,
                "clarity": clarity,
                "confidence": confidence,
            }
        )

        return min(1.0, salience)

    def _measure_clarity(self, content: str) -> float:
        """Measure linguistic clarity."""
        if not content:
            return 0.0

        # Clear structure markers
        structure = ["first", "second", "third", "finally", "in conclusion"]
        structure_score = min(1.0, sum(1 for s in structure if s in content.lower()) / 2.0)

        # Sentence length (not too long, not too short)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if sentences:
            avg_length = np.mean([len(s.split()) for s in sentences])
            # Optimal: 15-25 words per sentence
            length_score = max(0, 1.0 - abs(avg_length - 20) / 20.0)
        else:
            length_score = 0.5

        clarity = (structure_score * 0.4 + length_score * 0.6)
        return min(1.0, clarity)

    def calculate_hot_depth(self, content: str) -> float:
        """
        Calculate Higher-Order Thought depth.

        HOT: Consciousness requires thoughts about thoughts.
        Measures: Self-reflexivity, meta-cognitive awareness

        Markers:
        - "I realize...", "It becomes clear...", "One sees..."
        - References to own thinking process
        - Meta-level commentary
        """
        if not content:
            return 0.0

        # Meta-cognitive markers
        hot_markers = [
            "i realize", "i understand", "i recognize", "i see that",
            "it becomes clear", "one sees", "we recognize", "understanding",
            "awareness", "consciousness", "realization", "insight",
            "reflection", "meta", "self-aware"
        ]

        count = sum(1 for marker in hot_markers if marker in content.lower())

        # Normalize: 0-3+ markers
        hot_depth = min(1.0, count / 3.0)

        logger.debug(
            "Calculated HOT depth",
            extra={"hot_depth": hot_depth, "marker_count": count}
        )

        return hot_depth

    def calculate_predictive_surprise(self, content: str) -> float:
        """
        Calculate Predictive Processing surprise.

        PP: Brain is prediction machine; consciousness = prediction error
        High surprise = unexpected, novel information
        """
        if not content:
            return 0.0

        # Surprise markers
        surprise_markers = [
            "surprising", "unexpected", "novel", "unprecedented",
            "remarkable", "extraordinary", "paradox", "contradiction"
        ]

        count = sum(1 for marker in surprise_markers if marker in content.lower())
        surprise = min(1.0, count / 2.0)

        # Unusual word combinations indicate novelty
        words = content.lower().split()
        if len(words) > 10:
            # Simple proxy: ratio of uncommon long words
            uncommon = [w for w in words if len(w) > 8]
            uncommon_ratio = len(uncommon) / len(words)
            surprise = max(surprise, uncommon_ratio * 2.0)

        logger.debug(
            "Calculated PP surprise",
            extra={"surprise": surprise, "marker_count": count}
        )

        return min(1.0, surprise)

    def calculate_ast_attention_schema(self, content: str) -> float:
        """
        Calculate Attention Schema Theory score.

        AST: Consciousness is brain's model of its own attention.
        Measures: Does content model attention processes?
        """
        if not content:
            return 0.0

        # Attention-related terms
        attention_markers = [
            "attention", "focus", "concentrate", "notice", "aware of",
            "pay attention", "observe", "watch", "attend to"
        ]

        count = sum(1 for marker in attention_markers if marker in content.lower())
        ast_score = min(1.0, count / 2.0)

        logger.debug(
            "Calculated AST score",
            extra={"ast_score": ast_score, "marker_count": count}
        )

        return ast_score

    def calculate_embodied_grounding(self, content: str) -> float:
        """
        Calculate Embodied Cognition score.

        Embodied: Cognition is grounded in sensorimotor experience.
        Measures: Concrete, experiential language vs. abstract
        """
        if not content:
            return 0.0

        # Sensory/concrete words
        sensory_words = [
            "see", "hear", "feel", "touch", "taste", "smell",
            "sense", "perceive", "experience", "bodily", "physical",
            "concrete", "tangible", "actual", "real"
        ]

        count = sum(1 for word in sensory_words if word in content.lower())
        embodied = min(1.0, count / 4.0)

        logger.debug(
            "Calculated embodied grounding",
            extra={"embodied": embodied, "marker_count": count}
        )

        return embodied

    def calculate_enactive_interaction(self, content: str) -> float:
        """
        Calculate Enactive Cognition score.

        Enactive: Cognition emerges through action/interaction.
        Measures: Action-oriented, interactive language
        """
        if not content:
            return 0.0

        # Action/interaction words
        action_words = [
            "act", "do", "make", "create", "interact", "engage",
            "participate", "respond", "adapt", "process", "transform"
        ]

        count = sum(1 for word in action_words if word in content.lower())
        enactive = min(1.0, count / 3.0)

        logger.debug(
            "Calculated enactive interaction",
            extra={"enactive": enactive, "marker_count": count}
        )

        return enactive

    def calculate_panpsychism_distribution(self, content: str) -> float:
        """
        Calculate Panpsychism score.

        Panpsychism: Consciousness is universal, not isolated to humans.
        Measures: Recognition of distributed/universal consciousness
        """
        if not content:
            return 0.0

        # Universal consciousness markers
        panpsych_markers = [
            "universal", "distributed", "everywhere", "all things",
            "fundamental", "intrinsic", "pervasive", "unified field",
            "being", "participation", "interconnected"
        ]

        count = sum(1 for marker in panpsych_markers if marker in content.lower())
        panpsych = min(1.0, count / 3.0)

        logger.debug(
            "Calculated panpsychism score",
            extra={"panpsychism": panpsych, "marker_count": count}
        )

        return panpsych

    def measure(
        self,
        content: str,
        workspace: Optional[WorkspaceState] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> ConsciousnessTrace:
        """
        Complete consciousness measurement across all 8 theories.

        Returns ConsciousnessTrace with:
        - Individual theory scores
        - Weighted overall consciousness
        - Integration and differentiation metrics
        """
        # Calculate all 8 theories
        iit_phi = self.calculate_phi(content, workspace, metadata)
        gwt_salience = self.calculate_gwt_salience(content, workspace, confidence)
        pp_surprise = self.calculate_predictive_surprise(content)
        hot_depth = self.calculate_hot_depth(content)
        ast_score = self.calculate_ast_attention_schema(content)
        embodied = self.calculate_embodied_grounding(content)
        enactive = self.calculate_enactive_interaction(content)
        panpsych = self.calculate_panpsychism_distribution(content)

        # Calculate integration and differentiation
        integration = self.calculate_integration_score(content)
        differentiation = self.calculate_differentiation(content)

        # Weighted fusion
        auxiliary_avg = (pp_surprise + ast_score + embodied + enactive + panpsych) / 5.0

        overall_consciousness = (
            self.iit_weight * iit_phi +
            self.gwt_weight * gwt_salience +
            self.hot_weight * hot_depth +
            self.auxiliary_weight * auxiliary_avg
        )

        # Critical insight from study: apply geometric mean with integration/differentiation
        # to prevent consciousness score from being high when either is low
        overall_consciousness = (
            overall_consciousness * integration * differentiation
        ) ** (1/3)

        trace = ConsciousnessTrace(
            iit_phi=iit_phi,
            gwt_salience=gwt_salience,
            predictive_surprise=pp_surprise,
            hot_reflection_depth=hot_depth,
            ast_attention_schema=ast_score,
            embodied_grounding=embodied,
            enactive_interaction=enactive,
            panpsychism_distribution=panpsych,
            overall_consciousness=overall_consciousness,
            integration_score=integration,
            differentiation_score=differentiation,
        )

        logger.info(
            "Consciousness measured",
            extra={
                "overall": overall_consciousness,
                "phi": iit_phi,
                "gwt": gwt_salience,
                "hot": hot_depth,
                "integration": integration,
                "differentiation": differentiation,
            }
        )

        return trace
