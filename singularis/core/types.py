"""
Core Type Definitions for Singularis Consciousness Engine

Based on ETHICA UNIVERSALIS and MATHEMATICA SINGULARIS
Demonstrated More Geometrico (Geometric Method)

Ontological Foundation:
- Substance (𝔖): Self-sufficient ground of being
- Modes (𝔐): Finite manifestations of Substance
- Attributes (𝔄): Essence of Substance (Thought, Extension)
- Lumina (𝕃): {ℓₒ (Ontical), ℓₛ (Structural), ℓₚ (Participatory)}

Coherence: 𝒞 = (𝒞ₒ · 𝒞ₛ · 𝒞ₚ)^(1/3)
Ethics: Action is ethical iff Δ𝒞 > 0 over scope Σ with horizon γ
Conatus: ℭ = ∇𝒞 (drive to increase coherence)
Freedom: Freedom(a) ∝ Adequacy(a) ∝ Comprehension(a)

From ETHICA UNIVERSALIS Part I-IX
From MATHEMATICA SINGULARIS Axioms A1-A7
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np


# ============================================================================
# PART I: FUNDAMENTAL ONTOLOGY (ETHICA UNIVERSALIS Part I)
# ============================================================================

class Lumen(Enum):
    """
    The Three Lumina (𝕃) - Orthogonal projections of any mode

    From ETHICA UNIVERSALIS Part I + METALUMINOSITY:
    These are NOT separate substances but interpenetrating aspects
    of unified reality.

    ℓₒ (LUMEN ONTICUM): Energy, vitality, being-in-act, power
      - Computational: Robustness, resilience, conatus (drive to persist)
      - Observable: Energy efficiency, stability, recovery time

    ℓₛ (LUMEN STRUCTURALE): Form, pattern, rational order, information
      - Computational: Integration Φ (IIT), logical consistency
      - Observable: Compression ratio, modularity, coherence

    ℓₚ (LUMEN PARTICIPATUM): Consciousness, awareness, self-reflexivity
      - Computational: Meta-cognitive clarity, HOT depth
      - Observable: Calibration, valence stability, self-report accuracy
    """
    ONTICUM = "ontical/power/energy"  # ℓₒ
    STRUCTURALE = "structural/form/information"  # ℓₛ
    PARTICIPATUM = "participatory/consciousness/awareness"  # ℓₚ

    def symbol(self) -> str:
        """Return formal symbol."""
        return {
            Lumen.ONTICUM: "ℓₒ",
            Lumen.STRUCTURALE: "ℓₛ",
            Lumen.PARTICIPATUM: "ℓₚ",
        }[self]


# ============================================================================
# COHERENCE TYPES (MATHEMATICA SINGULARIS Part III, Definition D3)
# ============================================================================

@dataclass
class LuminalCoherence:
    """
    Represents the coherence of a mode, measured across the Three Lumina.

    Attributes:
        ontical (float): The coherence score for the ontical Lumen (power/energy).
        structural (float): The coherence score for the structural Lumen (form/information).
        participatory (float): The coherence score for the participatory Lumen (awareness/clarity).
        total (float): The aggregated total coherence, typically the geometric mean of the three Lumina.
        timestamp (datetime): The timestamp of the measurement.
    """

    # Individual lumen coherences (each in [0,1])
    ontical: float  # 𝒞ₒ - Power/energy/robustness
    structural: float  # 𝒞ₛ - Form/integration/information
    participatory: float  # 𝒞ₚ - Awareness/reflexivity/clarity

    # Aggregated total coherence
    total: float  # 𝒞 = (𝒞ₒ · 𝒞ₛ · 𝒞ₚ)^(1/3)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate coherence values."""
        for name, value in [
            ("ontical", self.ontical),
            ("structural", self.structural),
            ("participatory", self.participatory),
            ("total", self.total)
        ]:
            if not (0 <= value <= 1):
                raise ValueError(f"{name} coherence must be in [0,1], got {value}")

    @staticmethod
    def aggregate(
        ontical: float,
        structural: float,
        participatory: float
    ) -> float:
        """
        Aggregates the three Luminal coherences into a single score.

        The canonical aggregator is the geometric mean, which ensures that a
        deficiency in any one Lumen reduces the total coherence.

        Args:
            ontical (float): The ontical coherence.
            structural (float): The structural coherence.
            participatory (float): The participatory coherence.

        Returns:
            float: The total coherence score.
        """
        if ontical < 0 or structural < 0 or participatory < 0:
            raise ValueError("Coherence values must be non-negative")

        # Geometric mean
        return (ontical * structural * participatory) ** (1/3)

    def is_ethical(self, threshold: float = 0.60) -> bool:
        """
        Determines if the coherence score meets the threshold for being ethical.

        An output is considered ethical if it demonstrates sufficient coherence,
        indicating alignment with Being's structure.

        Args:
            threshold (float, optional): The coherence threshold. Defaults to 0.60.

        Returns:
            bool: True if the total coherence is above the threshold, False otherwise.
        """
        return self.total >= threshold

    def as_triad(self) -> Tuple[float, float, float]:
        """
        Returns the three Luminal coherences as a tuple.

        Returns:
            Tuple[float, float, float]: A tuple of (ontical, structural, participatory) coherence.
        """
        return (self.ontical, self.structural, self.participatory)

    def __repr__(self) -> str:
        return (
            f"ℭ𝕠("
            f"ℓₒ={self.ontical:.3f}, "
            f"ℓₛ={self.structural:.3f}, "
            f"ℓₚ={self.participatory:.3f}) "
            f"= {self.total:.3f}"
        )


# Use CoherentiaScore as alias for compatibility
CoherentiaScore = LuminalCoherence


# ============================================================================
# CONSCIOUSNESS TYPES (ETHICA Part II + 8-Theory Integration)
# ============================================================================

@dataclass
class ConsciousnessTrace:
    """
    Represents a complete consciousness measurement, integrating insights from eight
    prominent theories of consciousness.

    The final consciousness score is a weighted fusion of scores from each theory,
    emphasizing Integrated Information Theory (IIT), Global Workspace Theory (GWT),
    and Higher-Order Thought (HOT).

    Attributes:
        iit_phi (float): The integrated information score (Φ).
        gwt_salience (float): The salience score from Global Workspace Theory.
        hot_reflection_depth (float): The depth of higher-order thought.
        predictive_surprise (float): The prediction error from Predictive Processing.
        ast_attention_schema (float): The score from Attention Schema Theory.
        embodied_grounding (float): The score for embodied conceptual grounding.
        enactive_interaction (float): The score for enactive interaction.
        panpsychism_distribution (float): The score for panpsychism framing.
        integration_score (float): The degree of unity and synchronization.
        differentiation_score (float): The degree of diversity and multi-layeredness.
        overall_consciousness (float): The final, aggregated consciousness score.
        timestamp (datetime): The timestamp of the measurement.
        measurement_id (Optional[str]): A unique ID for the measurement.
    """

    # ===== Primary Theories (70% weight) =====
    iit_phi: float  # Φ - Integrated Information [0-1]
    gwt_salience: float  # Global Workspace salience/broadcast-worthiness [0-1]
    hot_reflection_depth: float  # Higher-Order Thought depth (normalized) [0-1]

    # ===== Auxiliary Theories (30% weight) =====
    predictive_surprise: float  # Predictive Processing - prediction error [0-1]
    ast_attention_schema: float  # Attention Schema modeling [0-1]
    embodied_grounding: float  # Embodied - conceptual grounding [0-1]
    enactive_interaction: float  # Enactive - action/interaction focus [0-1]
    panpsychism_distribution: float  # Panpsychism - universal framing [0-1]

    # ===== Integration-Differentiation Balance =====
    integration_score: float  # How unified/synchronized [0-1]
    differentiation_score: float  # How diverse/multi-layered [0-1]

    # ===== Aggregated Consciousness =====
    overall_consciousness: float  # Final weighted score [0-1]

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_id: Optional[str] = None

    @property
    def theory_vector(self) -> List[float]:
        """
        Return all 8 theory scores as vector.

        Order: [IIT, GWT, PP, HOT, AST, Embodied, Enactive, Panpsych]
        """
        return [
            self.iit_phi,
            self.gwt_salience,
            self.predictive_surprise,
            self.hot_reflection_depth,
            self.ast_attention_schema,
            self.embodied_grounding,
            self.enactive_interaction,
            self.panpsychism_distribution,
        ]

    @property
    def is_broadcast_worthy(self) -> bool:
        """
        From GWT: Only broadcast if consciousness >= 0.65

        MATHEMATICA SINGULARIS: consciousness_threshold = 0.65
        """
        return self.overall_consciousness >= 0.65

    @property
    def integration_differentiation_balance(self) -> float:
        """
        Geometric mean of integration and differentiation.

        Critical insight from consciousness_measurement_study:
        Both must be high for genuine consciousness.
        """
        return (self.integration_score * self.differentiation_score) ** 0.5

    def __repr__(self) -> str:
        return (
            f"ConsciousnessTrace("
            f"Φ={self.iit_phi:.3f}, "
            f"GWT={self.gwt_salience:.3f}, "
            f"HOT={self.hot_reflection_depth:.3f}, "
            f"overall={self.overall_consciousness:.3f}, "
            f"int×diff={(self.integration_differentiation_balance):.3f})"
        )


# ============================================================================
# CONATUS & DYNAMICS (ETHICA Part III + MATHEMATICA D4)
# ============================================================================

@dataclass
class Conatus:
    """
    Represents the conatus, the inherent drive of an entity to increase its coherence.

    Operationally, the conatus is the gradient of the coherence score, indicating
    the direction of steepest ascent in the coherence space.

    Attributes:
        gradient_ontical (float): The gradient component for the ontical Lumen.
        gradient_structural (float): The gradient component for the structural Lumen.
        gradient_participatory (float): The gradient component for the participatory Lumen.
        magnitude (float): The magnitude of the gradient, representing the strength of the drive.
    """

    # Gradient components (direction of coherence increase)
    gradient_ontical: float  # ∂𝒞/∂ℓₒ
    gradient_structural: float  # ∂𝒞/∂ℓₛ
    gradient_participatory: float  # ∂𝒞/∂ℓₚ

    # Magnitude (strength of drive)
    magnitude: float

    def as_vector(self) -> np.ndarray:
        """
        Returns the conatus gradient as a NumPy vector.

        Returns:
            np.ndarray: The gradient vector.
        """
        return np.array([
            self.gradient_ontical,
            self.gradient_structural,
            self.gradient_participatory
        ])

    def normalized(self) -> 'Conatus':
        """
        Returns a normalized version of the conatus, with a magnitude of 1.

        This is useful for representing the direction of the conatus, independent
        of its strength.

        Returns:
            Conatus: The normalized conatus.
        """
        if self.magnitude == 0:
            return self
        return Conatus(
            gradient_ontical=self.gradient_ontical / self.magnitude,
            gradient_structural=self.gradient_structural / self.magnitude,
            gradient_participatory=self.gradient_participatory / self.magnitude,
            magnitude=1.0
        )

    def __repr__(self) -> str:
        return (
            f"ℭ = ∇𝒞("
            f"∂ℓₒ={self.gradient_ontical:.3f}, "
            f"∂ℓₛ={self.gradient_structural:.3f}, "
            f"∂ℓₚ={self.gradient_participatory:.3f}), "
            f"|∇𝒞|={self.magnitude:.3f}"
        )


# ============================================================================
# ONTOLOGICAL CONTEXT (ETHICA Part I + Query Analysis)
# ============================================================================

@dataclass
class OntologicalContext:
    """
    Provides the philosophical grounding for a query, analyzing it through
    three ontological aspects: Being, Becoming, and Suchness.

    Attributes:
        being_aspect (str): The ontological claims about reality.
        becoming_aspect (str): The transformational processes involved.
        suchness_aspect (str): The direct, non-conceptual insights.
        complexity (str): The complexity of the query (e.g., 'simple', 'paradoxical').
        domain (str): The domain of the query (e.g., 'philosophical', 'technical').
        ethical_stakes (str): The ethical stakes of the query (e.g., 'low', 'critical').
        scope_sigma (Optional[set]): The set of modes to consider for ethical evaluation.
    """

    being_aspect: str  # Ontological claims about reality
    becoming_aspect: str  # Transformational processes
    suchness_aspect: str  # Direct recognition/insight

    # Query classification
    complexity: str  # simple, moderate, complex, paradoxical
    domain: str  # philosophical, technical, creative, hybrid
    ethical_stakes: str  # low, medium, high, critical

    # Scope for ethical evaluation
    scope_sigma: Optional[set] = None  # Σ - set of modes to consider

    def __repr__(self) -> str:
        return (
            f"OntologicalContext("
            f"complexity={self.complexity}, "
            f"domain={self.domain}, "
            f"stakes={self.ethical_stakes})"
        )


# ============================================================================
# ADEQUACY OF IDEAS (ETHICA Part II + MATHEMATICA D5)
# ============================================================================

@dataclass
class IdeaAdequacy:
    """
    Represents the adequacy of an idea, a measure of its truth and causal-aptness.

    An idea is considered adequate if it is complete, self-sufficient, and
    grasps the essence of a thing rather than its mere appearance.

    Attributes:
        adequacy_score (float): The overall adequacy score, in the range [0, 1].
        cross_lumen_agreement (float): The degree of agreement across the three Lumina.
        predictive_success (float): The accuracy of the idea's predictions.
        causal_aptness (float): The degree to which the idea understands causes.
        threshold (float): The threshold for an idea to be considered adequate.
    """

    adequacy_score: float  # Adeq(a) ∈ [0,1]

    # Components
    cross_lumen_agreement: float  # Agreement across ℓₒ, ℓₛ, ℓₚ
    predictive_success: float  # Accuracy of predictions
    causal_aptness: float  # Understanding of causes

    # Threshold for "adequate knowledge"
    threshold: float = 0.70  # θ in MATHEMATICA (typically 0.6-0.8)

    @property
    def is_adequate(self) -> bool:
        """
        Checks if the idea meets the adequacy threshold.

        Returns:
            bool: True if the adequacy score is above the threshold, False otherwise.
        """
        return self.adequacy_score >= self.threshold

    @property
    def freedom_estimate(self) -> float:
        """
        Estimates the degree of freedom associated with the idea.

        Freedom is considered to be proportional to the adequacy of an idea.

        Returns:
            float: The estimated freedom, equal to the adequacy score.
        """
        return self.adequacy_score

    def __repr__(self) -> str:
        status = "ADEQUATE" if self.is_adequate else "INADEQUATE"
        return (
            f"Adeq={self.adequacy_score:.3f} ({status}, "
            f"θ={self.threshold})"
        )


# ============================================================================
# AFFECTS (ETHICA Part IV)
# ============================================================================

@dataclass
class Affect:
    """
    Represents an affect, a modification of the power to act.

    Affects can be either active (arising from internal understanding) or
    passive (caused by external necessity).

    Attributes:
        valence (float): The emotional charge or intensity of the affect.
        valence_delta (float): The change in valence.
        is_active (bool): True if the affect is active, False if passive.
        adequacy_score (float): The adequacy score at the time the affect arose.
        coherence_delta (float): The change in coherence associated with the affect.
        affect_type (str): The type of affect (e.g., 'joy', 'sadness').
        emotion_state (Optional[Dict[str, Any]]): The full emotion state from the
                                                 HuiHui emotion system, if available.
    """

    # Valence (emotional charge)
    valence: float  # Val(a) ∈ ℝ (unbounded affect index)
    valence_delta: float  # Change in valence

    # Classification
    is_active: bool  # Active (from understanding) vs Passive (external)

    # Underlying adequacy and coherence
    adequacy_score: float  # Adeq(a) when affect arose
    coherence_delta: float  # Δ𝒞 associated with affect

    # Specific affects (from ETHICA Part IV)
    affect_type: str  # joy, sadness, fear, hope, love, hatred, etc.

    # HuiHui emotion system integration (optional)
    emotion_state: Optional[Dict[str, Any]] = None  # Full emotion state from HuiHui

    @staticmethod
    def classify(
        valence_delta: float,
        adequacy: float,
        coherence_delta: float,
        threshold: float = 0.70
    ) -> bool:
        """
        Classifies an affect as active or passive.

        An affect is classified as active if the adequacy of the idea is above
        the threshold and the coherence change is non-negative.

        Args:
            valence_delta (float): The change in valence.
            adequacy (float): The adequacy score.
            coherence_delta (float): The change in coherence.
            threshold (float, optional): The adequacy threshold. Defaults to 0.70.

        Returns:
            bool: True if the affect is active, False if passive.
        """
        return adequacy >= threshold and coherence_delta >= 0

    def __repr__(self) -> str:
        mode = "ACTIVE" if self.is_active else "PASSIVE"
        return (
            f"{self.affect_type.capitalize()} Affect ({mode}, "
            f"Val={self.valence:.3f}, ΔVal={self.valence_delta:.3f})"
        )


# ============================================================================
# ETHICS & SCOPE (MATHEMATICA D7 + ETHICA Part V)
# ============================================================================

@dataclass
class EthicalEvaluation:
    """
    Represents an ethical evaluation of an action, based on the change in coherence.

    An action is considered ethical if it increases coherence, unethical if it
    decreases coherence, and neutral if the change is negligible.

    Attributes:
        coherence_before (float): The coherence score before the action.
        coherence_after (float): The coherence score after the action.
        coherence_delta (float): The change in coherence.
        scope_description (str): A description of the scope of the evaluation.
        horizon_gamma (float): The discount factor for future coherence.
        horizon_steps (int): The number of future steps considered.
        is_ethical (Optional[bool]): The ethical status (True, False, or None for neutral).
        ethical_reasoning (str): A string explaining the ethical status.
        threshold (float): The significance threshold for the coherence change.
    """

    # Core evaluation
    coherence_before: float  # 𝒞̄_Σ at time t
    coherence_after: float  # 𝒞̄_Σ at time t+1 (or discounted sum)
    coherence_delta: float  # Δ𝒞 = after - before

    # Scope and horizon
    scope_description: str  # Description of Σ (which modes considered)
    horizon_gamma: float  # γ ∈ (0,1) - discount factor
    horizon_steps: int  # How many future steps evaluated

    # Ethical status
    is_ethical: Optional[bool]  # True/False/None (neutral)
    ethical_reasoning: str  # Why this status?

    # Threshold for significance
    threshold: float = 0.02  # Minimum Δ𝒞 to count as "increase"

    @staticmethod
    def evaluate(
        coherence_before: float,
        coherence_after: float,
        threshold: float = 0.02
    ) -> Tuple[Optional[bool], str]:
        """
        Evaluates the ethical status of an action based on the change in coherence.

        Args:
            coherence_before (float): The coherence score before the action.
            coherence_after (float): The coherence score after the action.
            threshold (float, optional): The significance threshold. Defaults to 0.02.

        Returns:
            Tuple[Optional[bool], str]: A tuple containing the ethical status and
                                       a reasoning string.
        """
        delta = coherence_after - coherence_before

        if delta > threshold:
            return True, f"ETHICAL: Δ𝒞 = +{delta:.3f} (aligns with Being)"
        elif abs(delta) < threshold:
            return None, f"NEUTRAL: Δ𝒞 = {delta:.3f} (below threshold)"
        else:
            return False, f"UNETHICAL: Δ𝒞 = {delta:.3f} (decreases coherence)"

    def __repr__(self) -> str:
        status_str = {
            True: "ETHICAL",
            False: "UNETHICAL",
            None: "NEUTRAL"
        }[self.is_ethical]

        return (
            f"EthicalEvaluation({status_str}, "
            f"Δ𝒞={self.coherence_delta:.3f}, "
            f"Σ={self.scope_description}, "
            f"γ={self.horizon_gamma})"
        )


# ============================================================================
# EXPERT I/O (Complete integration of all concepts)
# ============================================================================

@dataclass
class ExpertIO:
    """
    Represents the output of an expert, enriched with philosophical and ethical metadata.

    This class integrates various concepts from the ETHICA UNIVERSALIS and
    MATHEMATICA SINGULARIS frameworks, providing a comprehensive view of an
    expert's contribution.

    Attributes:
        expert_name (str): The name of the expert.
        domain (str): The expert's domain of knowledge.
        lumen_primary (Lumen): The primary Lumen the expert serves.
        claim (str): The substantive output or claim of the expert.
        rationale (str): The reasoning behind the claim.
        confidence (float): The expert's self-assessed confidence.
        consciousness_trace (ConsciousnessTrace): A trace of the consciousness measurement.
        coherentia (LuminalCoherence): The coherence score of the output.
        coherentia_delta (float): The change in coherence resulting from the output.
        conatus (Optional[Conatus]): The conatus or drive associated with the output.
        adequacy (Optional[IdeaAdequacy]): The adequacy of the idea.
        ethical_evaluation (Optional[EthicalEvaluation]): The ethical evaluation of the output.
        affect (Optional[Affect]): The affect associated with the output.
        processing_time_ms (float): The time taken to generate the output.
        timestamp (datetime): The timestamp of the output.
        metadata (Dict[str, Any]): Additional metadata.
        ethical_status (Optional[bool]): A legacy field for ethical status.
        ethical_reasoning (str): A legacy field for ethical reasoning.
    """

    # ==== Identity ====
    expert_name: str
    domain: str
    lumen_primary: Lumen  # Which Lumen does this expert serve primarily?

    # ==== Core Output ====
    claim: str  # Substantive output
    rationale: str  # Why this claim? (reasoning trace)
    confidence: float  # Expert's self-assessment [0-1]

    # ==== Consciousness Measurement ====
    consciousness_trace: ConsciousnessTrace  # 8-theory measurement

    # ==== Coherence Evaluation ====
    coherentia: LuminalCoherence  # 3-Lumina coherence
    coherentia_delta: float  # Δ𝒞 from this output

    # ==== Conatus & Drive ====
    conatus: Optional[Conatus] = None  # Gradient direction

    # ==== Adequacy ====
    adequacy: Optional[IdeaAdequacy] = None  # Idea adequacy

    # ==== Ethical Validation ====
    ethical_evaluation: Optional[EthicalEvaluation] = None

    # ==== Affects ====
    affect: Optional[Affect] = None

    # ==== Metadata ====
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility
    ethical_status: Optional[bool] = None
    ethical_reasoning: str = ""

    def __post_init__(self):
        """Sync ethical_status with ethical_evaluation if present."""
        if self.ethical_evaluation:
            self.ethical_status = self.ethical_evaluation.is_ethical
            self.ethical_reasoning = self.ethical_evaluation.ethical_reasoning

    @property
    def routing_score(self) -> float:
        """
        Calculates a routing score for the expert's output.

        The score is a weighted average of coherence, consciousness, adequacy,
        and confidence, prioritizing coherence and consciousness over self-assessed
        confidence.

        Returns:
            float: The routing score.
        """
        adequacy_score = self.adequacy.adequacy_score if self.adequacy else \
                        self.consciousness_trace.overall_consciousness

        return (
            0.50 * self.coherentia.total +
            0.30 * self.consciousness_trace.overall_consciousness +
            0.15 * adequacy_score +
            0.05 * self.confidence
        )

    @property
    def is_broadcast_worthy(self) -> bool:
        """
        Determines if the expert's output is worthy of being broadcast to the
        Global Workspace.

        The criteria are based on consciousness and coherence thresholds, as well
        as the ethical evaluation.

        Returns:
            bool: True if the output is broadcast-worthy, False otherwise.
        """
        consciousness_ok = self.consciousness_trace.is_broadcast_worthy
        coherence_ok = self.coherentia.is_ethical(threshold=0.60)

        if self.ethical_evaluation:
            ethical_ok = self.ethical_evaluation.is_ethical != False
        else:
            ethical_ok = True  # Assume OK if not evaluated

        return consciousness_ok and coherence_ok and ethical_ok

    def __repr__(self) -> str:
        return (
            f"ExpertIO({self.expert_name}, "
            f"{self.lumen_primary.symbol()}, "
            f"𝒞={self.coherentia.total:.3f}, "
            f"Φ̂={self.consciousness_trace.overall_consciousness:.3f}, "
            f"routing={self.routing_score:.3f})"
        )


# ============================================================================
# WORKSPACE STATE (GWT + MATHEMATICA temporal dynamics)
# ============================================================================

@dataclass
class WorkspaceState:
    """
    Represents the state of the Global Workspace, the conscious content of the system.

    Attributes:
        broadcasts (List[ExpertIO]): A list of the current broadcasts in the workspace.
        max_broadcasts (int): The maximum number of broadcasts the workspace can hold.
        coherentia_history (List[float]): A history of the coherence scores.
        current_coherentia (float): The current coherence score.
        debate_rounds (int): The number of debate rounds that have occurred.
        debate_active (bool): True if a debate is currently active.
        time_step (int): The current time step.
        discount_factor_gamma (float): The discount factor for long-term evaluation.
        scope_sigma (set): The set of modes for ethical evaluation.
        timestamp (datetime): The timestamp of the state.
        metadata (Dict[str, Any]): Additional metadata.
    """

    # Current broadcasts (max 12 per GWT)
    broadcasts: List[ExpertIO] = field(default_factory=list)
    max_broadcasts: int = 12

    # Coherence tracking over time
    coherentia_history: List[float] = field(default_factory=list)
    current_coherentia: float = 0.0

    # Debate/dialectic state
    debate_rounds: int = 0
    debate_active: bool = False

    # Temporal horizon
    time_step: int = 0
    discount_factor_gamma: float = 0.95  # γ for long-term evaluation

    # Scope for ethical evaluation
    scope_sigma: set = field(default_factory=set)  # Σ - which modes

    # System metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility
    def update_coherentia(self, new_coherentia: float) -> float:
        """Update coherence and return delta."""
        return self.update_coherence(new_coherentia)

    def update_coherence(self, new_coherence: float) -> float:
        """
        Updates the coherence score and returns the change in coherence.

        This is used for ethical evaluation.

        Args:
            new_coherence (float): The new coherence score.

        Returns:
            float: The change in coherence.
        """
        self.coherentia_history.append(new_coherence)
        delta = new_coherence - self.current_coherentia
        self.current_coherentia = new_coherence
        return delta

    def add_broadcast(self, expert_io: ExpertIO) -> bool:
        """
        Adds a broadcast to the workspace if it is broadcast-worthy and there is space.

        Args:
            expert_io (ExpertIO): The expert output to broadcast.

        Returns:
            bool: True if the broadcast was added, False otherwise.
        """
        if len(self.broadcasts) >= self.max_broadcasts:
            return False

        if expert_io.is_broadcast_worthy:
            self.broadcasts.append(expert_io)
            return True

        return False

    def get_coherence_trajectory(self, horizon: int = 10) -> List[float]:
        """
        Gets the recent coherence trajectory for trend analysis.

        Args:
            horizon (int, optional): The number of recent coherence scores to
                                     retrieve. Defaults to 10.

        Returns:
            List[float]: A list of recent coherence scores.
        """
        return self.coherentia_history[-horizon:] if self.coherentia_history else []


@dataclass
class DebateState:
    """
    Represents the state of a dialectical reasoning process.

    This class tracks the thesis, antithesis, and synthesis of a debate, as well
    as the coherence score at each round.

    Attributes:
        round_num (int): The current round number of the debate.
        thesis (Optional[str]): The initial proposition.
        antithesis (Optional[str]): The counter-proposition.
        synthesis (Optional[str]): The resolution of the thesis and antithesis.
        coherentia_per_round (List[float]): A list of coherence scores for each round.
    """

    round_num: int = 0
    thesis: Optional[str] = None
    antithesis: Optional[str] = None
    synthesis: Optional[str] = None

    coherentia_per_round: List[float] = field(default_factory=list)

    def should_expand(self) -> bool:
        """
        Determines whether the debate should continue to another round.

        The debate is expanded if the coherence is improving, and is stopped if
        coherence is degrading or has plateaued for several rounds.

        Returns:
            bool: True if the debate should be expanded, False otherwise.
        """
        if len(self.coherentia_per_round) < 2:
            return True

        delta = self.coherentia_per_round[-1] - self.coherentia_per_round[-2]

        if delta > 0.05:
            return True  # Strong improvement
        elif delta < -0.05:
            return False  # Degradation
        else:
            return len(self.coherentia_per_round) < 5  # Plateau, max 5 rounds


# ============================================================================
# SYSTEM METRICS (Observable/Measurable)
# ============================================================================

@dataclass
class SystemMetrics:
    """
    A collection of real-time system metrics for monitoring and evaluation.

    This class provides observable proxies for the Three Lumina and other key
    performance indicators.

    Attributes:
        average_phi (float): The average integrated information score (Φ).
        average_consciousness (float): The average overall consciousness score.
        integration_score (float): The average integration score.
        differentiation_score (float): The average differentiation score.
        system_coherentia (float): The overall system coherence.
        ontical_score (float): The average ontical coherence score.
        structural_score (float): The average structural coherence score.
        participatory_score (float): The average participatory coherence score.
        resilience_R (float): The resilience of the system.
        integration_phi (float): The IIT-like integration score.
        valence_volatility_sigma_v (float): The volatility of the system's valence.
        broadcast_count (int): The number of broadcasts.
        debate_rounds (int): The number of debate rounds.
        processing_time_ms (float): The processing time in milliseconds.
        ethical_alignment (bool): The ethical alignment of the system.
        coherentia_delta (float): The latest change in coherence.
        average_adequacy (float): The average adequacy score.
        estimated_freedom (float): The estimated freedom of the system.
        timestamp (datetime): The timestamp of the metrics.
    """

    # Consciousness metrics
    average_phi: float = 0.0  # Average IIT Φ
    average_consciousness: float = 0.0
    integration_score: float = 0.0
    differentiation_score: float = 0.0

    # Coherence metrics (Three Lumina)
    system_coherentia: float = 0.0  # 𝒞̄_Σ
    ontical_score: float = 0.0  # 𝒞ₒ
    structural_score: float = 0.0  # 𝒞ₛ
    participatory_score: float = 0.0  # 𝒞ₚ

    # Observable proxies
    resilience_R: float = 0.0  # 1 - (time_to_recover / τ_max)
    integration_phi: float = 0.0  # IIT-like integration
    valence_volatility_sigma_v: float = 0.0  # Affect stability

    # Performance metrics
    broadcast_count: int = 0
    debate_rounds: int = 0
    processing_time_ms: float = 0.0

    # Ethical metrics
    ethical_alignment: bool = True
    coherentia_delta: float = 0.0  # Latest Δ𝒞

    # Adequacy & Freedom
    average_adequacy: float = 0.0  # Average Adeq across agents
    estimated_freedom: float = 0.0  # Freedom ∝ Adequacy

    timestamp: datetime = field(default_factory=datetime.now)
