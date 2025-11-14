"""
OMEGA DNA Hyperhelix

A second, higher-order helix that maps to the existing Double Helix and
coordinates meta-reasoning across systems using a phase-fluctuating 4D model.

Principles implemented (scaffolding + hooks):
- Symbolic Logic Gating linkage for LLM MoE calls (neural-symbolic)
- Modular, Composable Architecture (MoE specialization, mission-driven layers)
- World Model Foundation (hierarchical predictive processing, active inference)
- Multimodal Integration (progressive fusion, cross-modal attention, alignment)
- Curriculum Development (automated curriculum with return-gain signals)
- Test-Time Adaptation (TTT: fast weights, large chunk updates)
- Hybrid Architectures (SSM + Transformer memory hybrid)
- Continual Learning (EWC-style selective plasticity, wake-sleep consolidation)

This module is intentionally light-weight and non-invasive: it exposes hooks
that wire into existing systems when present, and computes meta-metrics that
can be surfaced in dashboards.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


@dataclass
class Phase4D:
    """Phase-fluctuating 4D state (integration, temporal, causal, predictive)."""
    integration: float = 0.6
    temporal: float = 0.6
    causal: float = 0.6
    predictive: float = 0.6
    t: float = 0.0
    
    def as_array(self) -> np.ndarray:
        return np.array([self.integration, self.temporal, self.causal, self.predictive], dtype=float)


@dataclass
class OmegaNode:
    """Omega meta-node that maps to a Double-Helix node."""
    node_id: str
    name: str
    phase: Phase4D = field(default_factory=Phase4D)
    mission_role: str = "coordinator"
    last_update: float = field(default_factory=time.time)
    improvement_rate: float = 0.0
    contribution_weight: float = 0.5


class OmegaHyperhelix:
    """
    OMEGA DNA Hyperhelix
    - Maps to DoubleHelixArchitecture nodes
    - Maintains 4D phase state and modulates gating/weights
    - Exposes hooks into symbolic gating, MoE, curriculum, multimodal, TTA, CL
    """
    
    def __init__(
        self,
        double_helix,
        *,
        symbolic_bridge=None,
        curriculum=None,
        task_sampler=None,
        memory=None,
        world_model=None,
        perception=None,
        video_interpreter=None,
        voice_system=None,
        moe=None,
        hybrid_llm=None,
    ):
        self.double_helix = double_helix
        self.symbolic_bridge = symbolic_bridge
        self.curriculum = curriculum
        self.task_sampler = task_sampler
        self.memory = memory
        self.world_model = world_model
        self.perception = perception
        self.video_interpreter = video_interpreter
        self.voice_system = voice_system
        self.moe = moe
        self.hybrid_llm = hybrid_llm
        
        # Node mapping (1:1 with Double Helix nodes)
        self.nodes: Dict[str, OmegaNode] = {}
        if self.double_helix and getattr(self.double_helix, 'nodes', None):
            for node_id, node in self.double_helix.nodes.items():
                self.nodes[node_id] = OmegaNode(node_id=node_id, name=node.name)
        
        # Global 4D phase
        self.phase = Phase4D()
        
        # Test-Time Adaptation (fast weights)
        self.fast_weights: Dict[str, Any] = {}
        self.last_tta_time: float = 0.0
        
        # Continual learning (EWC-style placeholders)
        self.ewc_fisher: Dict[str, np.ndarray] = {}
        self.ewc_lambda: float = 0.1
        self.last_consolidation_time: float = time.time()
        
        # Metrics
        self.stats = {
            'init_time': time.time(),
            'gating_events': 0,
            'moe_calls': 0,
            'hybrid_calls': 0,
            'cost_saved_estimate': 0.0,
            'phase_updates': 0,
            'multimodal_alignments': 0,
            'avg_alignment_score': 0.0,
            'tta_updates': 0,
            'curriculum_suggestions': 0,
            'ewc_consolidations': 0,
        }
    
    # ──────────────────────────────────────────────────────────────────────────
    # Phase model (4D oscillator) and modulation
    # ──────────────────────────────────────────────────────────────────────────
    def update_phase(self, dt: float = 0.5):
        """Update 4D phase via smooth oscillators (bounded in [0.3, 0.95])."""
        self.phase.t += dt
        # Use coupled sinusoids for gentle fluctuations
        self.phase.integration = 0.65 + 0.15 * math.sin(self.phase.t * 0.31)
        self.phase.temporal    = 0.65 + 0.15 * math.sin(self.phase.t * 0.27 + 1.1)
        self.phase.causal      = 0.65 + 0.15 * math.sin(self.phase.t * 0.23 + 2.0)
        self.phase.predictive  = 0.65 + 0.15 * math.sin(self.phase.t * 0.19 + 2.7)
        
        self.phase.integration = float(np.clip(self.phase.integration, 0.3, 0.95))
        self.phase.temporal    = float(np.clip(self.phase.temporal,    0.3, 0.95))
        self.phase.causal      = float(np.clip(self.phase.causal,      0.3, 0.95))
        self.phase.predictive  = float(np.clip(self.phase.predictive,  0.3, 0.95))
        
        self.stats['phase_updates'] += 1
    
    def phase_modulate_weight(self, base_weight: float) -> float:
        """Modulate a weight using current 4D phase (average influence)."""
        phase_gain = float(np.mean(self.phase.as_array()))  # 0.3..0.95
        return float(np.clip(base_weight * (0.75 + 0.5 * (phase_gain - 0.5)), 0.0, 1.5))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Symbolic gating and MoE linkage
    # ──────────────────────────────────────────────────────────────────────────
    def record_gating_event(self, decision: Dict[str, Any], context: Dict[str, Any]):
        """Called when symbolic gate makes a decision for LLM usage."""
        self.stats['gating_events'] += 1
        # Estimate cost saved when no LLM used
        if not decision.get('should_invoke_llm', False):
            self.stats['cost_saved_estimate'] += 0.01
        
        # Optionally modulate thresholds/phases based on scene dynamics
        health = float(context.get('health', 100))
        in_combat = bool(context.get('in_combat', False))
        if in_combat and health < 40:
            # Increase temporal/causal weighting under stress
            self.phase.temporal = float(np.clip(self.phase.temporal + 0.05, 0.3, 0.99))
            self.phase.causal   = float(np.clip(self.phase.causal   + 0.05, 0.3, 0.99))
    
    def record_moe_query(self, mode: str = 'reasoning'):
        """Called when MoE is queried (reasoning/vision)."""
        if mode == 'reasoning':
            self.stats['moe_calls'] += 1
        else:
            self.stats['hybrid_calls'] += 1
    
    # ──────────────────────────────────────────────────────────────────────────
    # World model: predictive processing + active inference (scaffold)
    # ──────────────────────────────────────────────────────────────────────────
    def predictive_error(self, prediction: np.ndarray, observation: np.ndarray) -> float:
        if prediction is None or observation is None:
            return 0.0
        diff = prediction - observation
        return float(np.sqrt(np.sum(diff * diff)))
    
    def active_inference_adjust(self, belief_state: Dict[str, Any], error: float) -> Dict[str, Any]:
        # Simple proportional adjustment (placeholder for real active inference)
        adjusted = dict(belief_state)
        adjusted['uncertainty'] = max(0.0, min(1.0, belief_state.get('uncertainty', 0.5) * (1.0 - 0.1 * np.tanh(error))))
        return adjusted
    
    # ──────────────────────────────────────────────────────────────────────────
    # Multimodal integration: alignment + progressive fusion (scaffold)
    # ──────────────────────────────────────────────────────────────────────────
    def compute_alignment(self, emb_a: Optional[np.ndarray], emb_b: Optional[np.ndarray]) -> float:
        if emb_a is None or emb_b is None:
            return 0.0
        a = np.array(emb_a, dtype=float)
        b = np.array(emb_b, dtype=float)
        if a.shape != b.shape:
            n = min(a.shape[-1], b.shape[-1])
            a = a[..., :n]
            b = b[..., :n]
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(np.clip(num / den, -1.0, 1.0))
    
    def record_multimodal_alignment(self, scores: List[float]):
        if not scores:
            return
        self.stats['multimodal_alignments'] += 1
        prev = self.stats['avg_alignment_score']
        n = self.stats['multimodal_alignments']
        mean_score = float(np.mean(scores))
        self.stats['avg_alignment_score'] = (prev * (n - 1) + mean_score) / n
    
    # ──────────────────────────────────────────────────────────────────────────
    # Curriculum development (automated, return-gain)
    # ──────────────────────────────────────────────────────────────────────────
    def propose_curriculum_tasks(self, count: int = 5) -> List[Any]:
        suggestions = []
        if self.task_sampler:
            try:
                suggestions = self.task_sampler.get_recommended_tasks(count=count)
                self.stats['curriculum_suggestions'] += len(suggestions)
            except Exception:
                pass
        return suggestions
    
    # ──────────────────────────────────────────────────────────────────────────
    # Test-Time Adaptation (TTT): fast weights (scaffold)
    # ──────────────────────────────────────────────────────────────────────────
    def apply_tta_update(self, key: str, delta: Any):
        """Apply a small fast-weight update to adapt at test-time."""
        self.fast_weights[key] = delta
        self.last_tta_time = time.time()
        self.stats['tta_updates'] += 1
    
    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid SSM + Transformer memory (scaffold)
    # ──────────────────────────────────────────────────────────────────────────
    def sequence_hybrid_score(self, short_seq_emb: np.ndarray, long_ctx_emb: np.ndarray) -> float:
        """Combine SSM-like short sequence efficiency with long-range retrieval."""
        # Simple mixture as placeholder
        short_scale = 0.6
        long_scale = 0.4
        s = float(np.linalg.norm(short_seq_emb)) if short_seq_emb is not None else 0.0
        l = float(np.linalg.norm(long_ctx_emb)) if long_ctx_emb is not None else 0.0
        return short_scale * s + long_scale * l
    
    # ──────────────────────────────────────────────────────────────────────────
    # Continual learning: EWC + wake-sleep consolidation (scaffold)
    # ──────────────────────────────────────────────────────────────────────────
    def ewc_consolidate(self, params: Dict[str, np.ndarray]):
        """Update Fisher info estimates (placeholder)."""
        for k, v in params.items():
            fisher = self.ewc_fisher.get(k, np.zeros_like(v))
            self.ewc_fisher[k] = 0.95 * fisher + 0.05 * (v * v)
        self.stats['ewc_consolidations'] += 1
        self.last_consolidation_time = time.time()
    
    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def tick(self, dt: float = 0.5):
        """Advance OMEGA hyperhelix state (call each cycle)."""
        self.update_phase(dt)
    
    def get_phase_state(self) -> Dict[str, float]:
        return {
            'integration': self.phase.integration,
            'temporal': self.phase.temporal,
            'causal': self.phase.causal,
            'predictive': self.phase.predictive,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        phase = self.get_phase_state()
        return {
            'nodes_mapped': len(self.nodes),
            'phase': phase,
            **self.stats,
        }
