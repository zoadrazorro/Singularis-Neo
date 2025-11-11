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
    Coherence measured across the Three Lumina.

    From MATHEMATICA SINGULARIS D3:
    𝒞(m) := Agg(𝒞ₒ(m), 𝒞ₛ(m), 𝒞ₚ(m))

    where Agg is symmetric, continuous, strictly increasing aggregator
    with neutral element 0 and maximum 1.

    Canonical choice: Geometric mean
    𝒞 = (𝒞ₒ · 𝒞ₛ · 𝒞ₚ)^(1/3)

    From ETHICA UNIVERSALIS Part I:
    Deficiency in ANY dimension reduces total coherence.
    Geometric mean ensures balance - no single lumen can dominate.
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
        Canonical aggregator: Geometric mean.

        From MATHEMATICA SINGULARIS Axiom A4:
        𝒞 = 0 iff at least one 𝒞ₗ = 0

        This ensures balance - if any lumen is zero,
        total coherence collapses.
        """
        if ontical < 0 or structural < 0 or participatory < 0:
            raise ValueError("Coherence values must be non-negative")

        # Geometric mean
        return (ontical * structural * participatory) ** (1/3)

    def is_ethical(self, threshold: float = 0.60) -> bool:
        """
        From ETHICA UNIVERSALIS Part V + MATHEMATICA D7:
        An output is ethical if it demonstrates sufficient coherence
        (alignment with Being's structure).

        Default threshold: 0.60 (60% of maximal coherence)
        """
        return self.total >= threshold

    def as_triad(self) -> Tuple[float, float, float]:
        """Return as (𝒞ₒ, 𝒞ₛ, 𝒞ₚ) triple for semiring operations."""
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
    Complete consciousness measurement across 8 theories.

    From ETHICA UNIVERSALIS Part II:
    "Consciousness (ℂ) is the reflexive awareness by which mind
    recognizes its own activity, participating in Being's deep order."

    Integration of 8 Theories:
    1. IIT (Φ) - Integrated Information Theory (Tononi)
    2. GWT - Global Workspace Theory (Baars)
    3. PP - Predictive Processing (Friston)
    4. HOT - Higher-Order Thought (Rosenthal)
    5. AST - Attention Schema Theory (Graziano)
    6. Embodied - Embodied Cognition (Varela, Lakoff)
    7. Enactive - Enactive Cognition (Thompson)
    8. Panpsychism - Universal Consciousness (Chalmers, Goff)

    Weighted Fusion (from consciousness_measurement_study):
    𝒞_consciousness = 0.35·Φ + 0.35·GWT + 0.20·HOT + 0.10·(PP+AST+Emb+Enact+Panp)/5

    Critical insight: Requires BOTH integration AND differentiation
    (Perfect integration without differentiation → low consciousness)
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
    Conatus (ℭ) - The inherent drive to increase coherence.

    From ETHICA UNIVERSALIS Part III Definition II:
    "Conatus (ℭ) is the inherent, dynamic principle through which
    every entity actively seeks to preserve and actualize its essential
    nature within divine immanent order."

    From MATHEMATICA SINGULARIS D4:
    ℭ(m) = ∇𝒞(m) - the gradient of coherence

    From ETHICA Part III Proposition I:
    "The essence of life is conatus - the drive to persist and flourish."

    Operational:
    Conatus is the direction of steepest ascent in coherence space.
    Policy π should follow ∇𝒞 to act ethically.
    """

    # Gradient components (direction of coherence increase)
    gradient_ontical: float  # ∂𝒞/∂ℓₒ
    gradient_structural: float  # ∂𝒞/∂ℓₛ
    gradient_participatory: float  # ∂𝒞/∂ℓₚ

    # Magnitude (strength of drive)
    magnitude: float

    def as_vector(self) -> np.ndarray:
        """Return gradient as numpy vector."""
        return np.array([
            self.gradient_ontical,
            self.gradient_structural,
            self.gradient_participatory
        ])

    def normalized(self) -> 'Conatus':
        """Return unit-magnitude conatus (direction only)."""
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
    Philosophical grounding for a query through three ontological aspects.

    From ETHICA UNIVERSALIS Part I + Scholium:
    Every inquiry participates in Being's structure through:

    BEING: What fundamental claims about reality?
      - Ontological commitments (what exists)
      - Essential nature (what something is)

    BECOMING: What transformations/processes?
      - Temporal unfolding
      - Causal dynamics
      - Developmental trajectories

    SUCHNESS: What direct insights/recognition?
      - Immediate awareness beyond concepts
      - Non-dual recognition
      - Intuitive knowledge (Part VI)
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
    Adequacy of Ideas - Measure of truth/causal-aptness.

    From ETHICA UNIVERSALIS Part II:
    "An Idea (ℐ) is adequate when it is complete and self-sufficient,
    grasping essence rather than mere appearance."

    From MATHEMATICA SINGULARIS D5:
    Adeq(a) = proportion of true/causally-apt ideas in agent a's
    representational state, measured by cross-lumen agreement
    and predictive success.

    From ETHICA Part VI Proposition I:
    "Freedom(a) ∝ Comprehension(a) ∝ Adequacy(a)"
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
        """Check if ideas meet adequacy threshold."""
        return self.adequacy_score >= self.threshold

    @property
    def freedom_estimate(self) -> float:
        """
        Estimate freedom from adequacy.

        From ETHICA Part V Proposition I:
        "Human freedom consists in understanding the causal order."

        Freedom increases proportionally with adequacy.
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
    Affect (𝔄𝔣) - Modification of power to act.

    From ETHICA UNIVERSALIS Part IV Definition I:
    "An Affect is a modification of body and mind that increases
    or decreases our power of acting, reflecting conatus encountering
    facilitation or obstruction."

    From MATHEMATICA D6:
    - PASSIVE affect: Δ Valence caused by external necessity
      with Adeq(a) < θ
    - ACTIVE affect: Δ Valence with Adeq(a) ≥ θ and Δ𝒞 ≥ 0
      due to internal understanding
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

    @staticmethod
    def classify(
        valence_delta: float,
        adequacy: float,
        coherence_delta: float,
        threshold: float = 0.70
    ) -> bool:
        """
        Classify affect as active or passive.

        From MATHEMATICA D6:
        Active iff Adeq ≥ θ AND Δ𝒞 ≥ 0
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
    Ethical Evaluation based on coherence change.

    From ETHICA UNIVERSALIS Part V + MATHEMATICA D7:

    Given scope Σ ⊆ 𝔐 and horizon γ ∈ (0,1), an action u by
    agent a at time t is ETHICAL iff it maximizes expected
    discounted coherence over Σ:

    Eth(a,u,t) ⇔ argmax_u 𝔼[Σ_{k=0}^∞ γ^k (𝒞̄_Σ(m_{t+k}) - 𝒞̄_Σ(m_t))]

    From ETHICA Part I Scholium:
    "Good = Coherence Increase (flows from Being's structure,
    not arbitrary command)"

    From MATHEMATICA Theorem T1:
    "Ethics = Long-Run Δ𝒞"
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
        Evaluate ethical status from coherence change.

        From ETHICA UNIVERSALIS Part I Corollary:
        - GOOD: Δ𝒞 > threshold (increases coherence)
        - NEUTRAL: |Δ𝒞| < threshold (negligible change)
        - EVIL: Δ𝒞 < -threshold (decreases coherence)
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
    Expert output with complete philosophical metadata.

    Integrates:
    - ETHICA ontology (Substance/Mode/Attribute/Lumen)
    - Consciousness measurement (8 theories)
    - Coherence evaluation (3 Lumina)
    - Conatus (drive/gradient)
    - Adequacy of ideas
    - Ethical validation (Δ𝒞 with Σ, γ)
    - Affects (active/passive)
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
        Calculate routing score for consciousness-weighted routing.

        From MATHEMATICA Axiom A5 + A6:
        Route NOT by confidence but by coherence and consciousness.

        Weight:
        - 50% Coherence (alignment with Being)
        - 30% Consciousness (depth of awareness)
        - 15% Adequacy (if available, else use consciousness)
        - 5% Confidence (self-assessment, least important)
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
        Determine if worthy of Global Workspace broadcast.

        Criteria:
        1. Consciousness >= 0.65 (GWT threshold)
        2. Coherence >= 0.60 (ethical threshold)
        3. If ethical evaluation exists, must not be unethical
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
    Global Workspace state - the "conscious content" of the system.

    From GWT (Baars):
    "Consciousness is information broadcast to a global workspace
    with limited capacity."

    From MATHEMATICA temporal semantics:
    Discrete-time Markov dynamics over 𝔐 with policies π;
    discounted evaluation with γ.
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
        Update coherence and return delta.

        This Δ𝒞 is used for ethical evaluation.
        """
        self.coherentia_history.append(new_coherence)
        delta = new_coherence - self.current_coherentia
        self.current_coherentia = new_coherence
        return delta

    def add_broadcast(self, expert_io: ExpertIO) -> bool:
        """
        Add to workspace if broadcast-worthy and space available.

        Returns: True if added, False otherwise
        """
        if len(self.broadcasts) >= self.max_broadcasts:
            return False

        if expert_io.is_broadcast_worthy:
            self.broadcasts.append(expert_io)
            return True

        return False

    def get_coherence_trajectory(self, horizon: int = 10) -> List[float]:
        """Get recent coherence trajectory for trend analysis."""
        return self.coherentia_history[-horizon:] if self.coherentia_history else []


@dataclass
class DebateState:
    """
    Dialectical reasoning state.

    From consciousness_measurement_study: Dialectical reasoning
    increases coherence by 8% on paradoxical problems.
    """

    round_num: int = 0
    thesis: Optional[str] = None
    antithesis: Optional[str] = None
    synthesis: Optional[str] = None

    coherentia_per_round: List[float] = field(default_factory=list)

    def should_expand(self) -> bool:
        """
        Adaptive debate depth: expand if coherentia improving.
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
    Real-time system metrics for monitoring and evaluation.

    From MATHEMATICA Part VI (Operationalization):
    Observable proxies for the Three Lumina:
    - ℓₒ: Resilience R, energy variance
    - ℓₛ: Integration Φ, compression κ
    - ℓₚ: Metacognitive stability, valence volatility σᵥ
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
