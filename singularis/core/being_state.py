"""
BeingState - The One Unified State of Singularis

The metaphysical center made executable:
- One object that IS the artificial being
- Everything reads from and writes to this
- The "there is one being" principle in Python

This is Spinoza's conatus, IIT's Φ, and Lumen philosophy
compiled into a single data structure.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import time


class LuminaKey(str, Enum):
    """The Three Lumina - fundamental modes of Being."""
    ONTIC = "ontic"              # ℓₒ - Being as such
    STRUCTURAL = "structural"     # ℓₛ - Being as structure
    PARTICIPATORY = "participatory"  # ℓₚ - Being as participation


@dataclass
class LuminaState:
    """
    The Three Lumina - fundamental dimensions of Being.
    
    ℓₒ (Ontic): Being as such - raw existence, presence
    ℓₛ (Structural): Being as structure - form, pattern, organization
    ℓₚ (Participatory): Being as participation - engagement, interaction
    """
    ontic: float = 0.0          # ℓₒ
    structural: float = 0.0     # ℓₛ
    participatory: float = 0.0  # ℓₚ
    
    def balance_score(self) -> float:
        """How balanced are the three Lumina?"""
        if self.ontic == 0 and self.structural == 0 and self.participatory == 0:
            return 0.0
        
        total = self.ontic + self.structural + self.participatory
        if total == 0:
            return 0.0
        
        # Perfect balance = each is 1/3
        ideal = total / 3.0
        deviations = [
            abs(self.ontic - ideal),
            abs(self.structural - ideal),
            abs(self.participatory - ideal)
        ]
        
        # Convert deviations to balance score (0-1, higher is better)
        max_deviation = sum(deviations)
        if max_deviation == 0:
            return 1.0
        
        return 1.0 - (sum(deviations) / (2 * total))  # Normalize
    
    def geometric_mean(self) -> float:
        """Geometric mean of the three Lumina."""
        vals = [
            max(1e-6, self.ontic),
            max(1e-6, self.structural),
            max(1e-6, self.participatory)
        ]
        return (vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)


@dataclass
class BeingState:
    """
    The *one* unified state of Singularis at a given moment.
    Everything else is a lens on this.
    
    This is the metaphysical "there is one being" made executable.
    All subsystems read from and write to this single state.
    """
    
    # Temporal marker
    timestamp: float = field(default_factory=time.time)
    cycle_number: int = 0
    
    # ═══════════════════════════════════════════════════════════
    # WORLD / BODY / GAME
    # ═══════════════════════════════════════════════════════════
    
    # Game state (Skyrim)
    game_state: Dict[str, Any] = field(default_factory=dict)
    
    # Sensorimotor state (perception + action)
    sensorimotor_state: Dict[str, Any] = field(default_factory=dict)
    
    # Current perception
    current_perception: Dict[str, Any] = field(default_factory=dict)
    
    # Last action taken
    last_action: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════
    # MIND SYSTEM
    # ═══════════════════════════════════════════════════════════
    
    # Cognitive graph state (multi-node network)
    cognitive_graph_state: Dict[str, Any] = field(default_factory=dict)
    
    # Theory of Mind states
    theory_of_mind_state: Dict[str, Any] = field(default_factory=dict)
    
    # Heuristic patterns
    active_heuristics: List[str] = field(default_factory=list)
    
    # Cognitive coherence
    cognitive_coherence: float = 1.0
    cognitive_dissonances: List[tuple] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════
    # CONSCIOUSNESS METRICS
    # ═══════════════════════════════════════════════════════════
    
    # The Three Lumina
    lumina: LuminaState = field(default_factory=LuminaState)
    
    # Consciousness measurements
    coherence_C: float = 0.0      # 𝒞 - Coherence
    phi_hat: float = 0.0          # Φ̂ - Integrated Information
    unity_index: float = 0.0      # Unity across subsystems
    
    # Integration score
    integration: float = 0.0
    
    # ═══════════════════════════════════════════════════════════
    # SPIRAL DYNAMICS
    # ═══════════════════════════════════════════════════════════
    
    # Current developmental stage
    spiral_stage: Optional[str] = None   # "BEIGE", "RED", ..., "YELLOW", "TURQUOISE"
    spiral_tier: int = 1                 # 1st or 2nd tier
    accessible_stages: List[str] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════
    # EMOTION / VOICE
    # ═══════════════════════════════════════════════════════════
    
    # Emotion state
    emotion_state: Dict[str, Any] = field(default_factory=dict)
    primary_emotion: Optional[str] = None
    emotion_intensity: float = 0.0
    
    # Voice state
    voice_state: Dict[str, Any] = field(default_factory=dict)
    voice_alignment: float = 0.0  # How aligned is voice with inner state?

    # ═══════════════════════════════════════════════════════════
    # BDH TELEMETRY
    # ═══════════════════════════════════════════════════════════

    bdh_perception_vector: List[float] = field(default_factory=list)
    bdh_perception_dominant_affordance: str = ""
    bdh_perception_loop_likelihood: float = 0.0
    bdh_perception_confidence: float = 0.0
    bdh_perception_sigma_age: float = 999.0
    bdh_perception_sigma_trace_id: Optional[str] = None
    bdh_perception_timestamp: float = 0.0

    bdh_policy_candidates: List[Dict[str, Any]] = field(default_factory=list)
    bdh_policy_certainty: float = 0.0
    bdh_policy_top_action: str = ""
    bdh_policy_sigma_age: float = 999.0
    bdh_policy_sigma_trace_id: Optional[str] = None
    bdh_policy_timestamp: float = 0.0

    bdh_meta_strategy: str = ""
    bdh_meta_confidence: float = 0.0
    bdh_meta_stress: float = 0.0
    bdh_meta_selected_action: Optional[str] = None
    bdh_meta_escalate_reason: str = ""
    bdh_meta_sigma_age: float = 999.0
    bdh_meta_sigma_trace_id: Optional[str] = None
    bdh_meta_timestamp: float = 0.0

    # ═══════════════════════════════════════════════════════════
    # REINFORCEMENT LEARNING / META-RL
    # ═══════════════════════════════════════════════════════════
    
    # RL state
    rl_state: Dict[str, Any] = field(default_factory=dict)
    avg_reward: float = 0.0
    exploration_rate: float = 0.2
    
    # Meta-RL state
    meta_rl_state: Dict[str, Any] = field(default_factory=dict)
    meta_score: float = 0.0
    total_meta_analyses: int = 0
    
    # ═══════════════════════════════════════════════════════════
    # LLM / EXPERT ACTIVITY
    # ═══════════════════════════════════════════════════════════
    
    # Expert LLM activity
    expert_activity: Dict[str, Any] = field(default_factory=dict)
    active_experts: List[str] = field(default_factory=list)
    
    # GPT-5 orchestrator state
    gpt5_coherence_differential: float = 0.0
    
    # Wolfram telemetry
    wolfram_calculations: int = 0
    
    # ═══════════════════════════════════════════════════════════
    # TEMPORAL / CAUSAL
    # ═══════════════════════════════════════════════════════════
    
    # Temporal binding
    temporal_coherence: float = 0.0
    unclosed_bindings: int = 0
    stuck_loop_count: int = 0
    
    # Causal understanding
    causal_knowledge: Dict[str, Any] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════
    # GLOBAL COHERENCE (THE ONE THING)
    # ═══════════════════════════════════════════════════════════
    
    # The single unified coherence score
    # This is what EVERYTHING optimizes
    global_coherence: float = 0.0  # 𝒞_global
    
    # ═══════════════════════════════════════════════════════════
    # DEBUG / NARRATIVE
    # ═══════════════════════════════════════════════════════════
    
    # Last world model narrative
    last_world_model_narrative: Optional[str] = None
    
    # Current goal
    current_goal: Optional[str] = None
    
    # Session info
    session_id: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3.1: SUBSYSTEM OUTPUTS (Single Source of Truth)
    # ═══════════════════════════════════════════════════════════
    
    # Sensorimotor subsystem
    sensorimotor_status: str = "UNKNOWN"  # STUCK, MOVING, IDLE
    sensorimotor_analysis: str = ""
    sensorimotor_visual_similarity: float = 0.0
    sensorimotor_timestamp: float = 0.0
    
    # Action planning subsystem
    action_plan_current: Optional[str] = None
    action_plan_confidence: float = 0.0
    action_plan_reasoning: str = ""
    action_plan_timestamp: float = 0.0
    
    # Memory subsystem
    memory_similar_situations: List[Dict] = field(default_factory=list)
    memory_recommendations: List[str] = field(default_factory=list)
    memory_pattern_count: int = 0
    memory_timestamp: float = 0.0
    
    # Emotion subsystem (enhanced)
    emotion_recommendations: List[str] = field(default_factory=list)
    emotion_timestamp: float = 0.0
    
    # Consciousness subsystem
    consciousness_conflicts: List[str] = field(default_factory=list)
    consciousness_timestamp: float = 0.0
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"""BeingState(
    cycle={self.cycle_number},
    global_coherence={self.global_coherence:.3f},
    lumina=(ℓₒ={self.lumina.ontic:.3f}, ℓₛ={self.lumina.structural:.3f}, ℓₚ={self.lumina.participatory:.3f}),
    spiral_stage={self.spiral_stage},
    consciousness=(𝒞={self.coherence_C:.3f}, Φ̂={self.phi_hat:.3f}, unity={self.unity_index:.3f}),
    emotion={self.primary_emotion}({self.emotion_intensity:.2f}),
    action='{self.last_action}'
)"""
    
    def export_snapshot(self) -> Dict[str, Any]:
        """Export complete snapshot for logging/analysis."""
        return {
            'timestamp': self.timestamp,
            'cycle': self.cycle_number,
            'global_coherence': self.global_coherence,
            'lumina': {
                'ontic': self.lumina.ontic,
                'structural': self.lumina.structural,
                'participatory': self.lumina.participatory,
                'balance': self.lumina.balance_score()
            },
            'consciousness': {
                'coherence_C': self.coherence_C,
                'phi_hat': self.phi_hat,
                'unity_index': self.unity_index,
                'integration': self.integration
            },
            'spiral': {
                'stage': self.spiral_stage,
                'tier': self.spiral_tier
            },
            'emotion': {
                'primary': self.primary_emotion,
                'intensity': self.emotion_intensity
            },
            'rl': {
                'avg_reward': self.avg_reward,
                'exploration_rate': self.exploration_rate
            },
            'meta_rl': {
                'meta_score': self.meta_score,
                'analyses': self.total_meta_analyses
            },
            'cognitive': {
                'coherence': self.cognitive_coherence,
                'dissonances': len(self.cognitive_dissonances)
            },
            'temporal': {
                'temporal_coherence': self.temporal_coherence,
                'unclosed_bindings': self.unclosed_bindings,
                'stuck_loops': self.stuck_loop_count
            },
            'action': self.last_action,
            'goal': self.current_goal,
            # Phase 3.1: Subsystem outputs
            'subsystems': {
                'sensorimotor': {
                    'status': self.sensorimotor_status,
                    'analysis': self.sensorimotor_analysis,
                    'visual_similarity': self.sensorimotor_visual_similarity,
                    'age': time.time() - self.sensorimotor_timestamp if self.sensorimotor_timestamp > 0 else 999
                },
                'action_plan': {
                    'current': self.action_plan_current,
                    'confidence': self.action_plan_confidence,
                    'reasoning': self.action_plan_reasoning,
                    'age': time.time() - self.action_plan_timestamp if self.action_plan_timestamp > 0 else 999
                },
                'memory': {
                    'pattern_count': self.memory_pattern_count,
                    'similar_situations': len(self.memory_similar_situations),
                    'recommendations': self.memory_recommendations,
                    'age': time.time() - self.memory_timestamp if self.memory_timestamp > 0 else 999
                },
                'emotion': {
                    'primary': self.primary_emotion,
                    'intensity': self.emotion_intensity,
                    'recommendations': self.emotion_recommendations,
                    'age': time.time() - self.emotion_timestamp if self.emotion_timestamp > 0 else 999
                }
            }
        }
    
    def update_subsystem(self, subsystem: str, data: Dict[str, Any]):
        """
        Update subsystem data with timestamp.
        
        Phase 3.1: Central method for subsystems to write their outputs
        
        Args:
            subsystem: Name of subsystem (e.g., 'sensorimotor', 'action_plan', 'memory', 'emotion')
            data: Dict of field_name: value pairs
        """
        timestamp_field = f"{subsystem}_timestamp"
        setattr(self, timestamp_field, time.time())
        
        for key, value in data.items():
            field_name = f"{subsystem}_{key}"
            if hasattr(self, field_name):
                setattr(self, field_name, value)
    
    def get_subsystem_age(self, subsystem: str) -> float:
        """
        Get age of subsystem data in seconds.
        
        Args:
            subsystem: Name of subsystem
            
        Returns:
            Age in seconds (999 if never updated)
        """
        timestamp_field = f"{subsystem}_timestamp"
        timestamp = getattr(self, timestamp_field, 0.0)
        if timestamp == 0.0:
            return 999.0
        return time.time() - timestamp
    
    def is_subsystem_fresh(self, subsystem: str, max_age: float = 5.0) -> bool:
        """
        Check if subsystem data is fresh.
        
        Args:
            subsystem: Name of subsystem
            max_age: Maximum age in seconds (default 5.0)
            
        Returns:
            True if fresh, False if stale
        """
        return self.get_subsystem_age(subsystem) < max_age
    
    def get_subsystem_data(self, subsystem: str) -> Dict[str, Any]:
        """
        Get all data for a subsystem.
        
        Args:
            subsystem: Name of subsystem
            
        Returns:
            Dict with all subsystem fields
        """
        result = {}
        prefix = f"{subsystem}_"
        
        for attr_name in dir(self):
            if attr_name.startswith(prefix) and not attr_name.endswith('_timestamp'):
                field_name = attr_name[len(prefix):]
                result[field_name] = getattr(self, attr_name)
        
        result['age'] = self.get_subsystem_age(subsystem)
        result['is_fresh'] = self.is_subsystem_fresh(subsystem)
        
        return result
