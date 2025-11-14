"""
Matrix Network of Improvement (MNI)

A modular scaffolding architecture: Domain × Mode grid of learning modules
linked to Double Helix (Coherence↔Competence) and OMEGA Hyperhelix.

Domains (rows):
- sensorimotor, language, social, meta, planning, simulation
Modes (cols):
- S (Symbolic), H (Heuristic), E (Emergent)

Each module exposes:
- evaluate_coherence(), evaluate_competence() hooks
- apply_tta_update() fast-weight adaptation
- get_weights() for EWC consolidation

Manager provides:
- map_to_omega(): map all matrix modules to OMEGA nodes
- resolve_weights(): base confidence → modulated by OMEGA phase
- propose_curriculum(): use OMEGA suggestions to inject tasks
- get_stats(): metrics for dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np


class Domain(Enum):
    SENSORIMOTOR = "sensorimotor"
    LANGUAGE = "language"
    SOCIAL = "social"
    META = "meta"
    PLANNING = "planning"
    SIMULATION = "simulation"


class Mode(Enum):
    SYMBOLIC = "S"
    HEURISTIC = "H"
    EMERGENT = "E"


@dataclass
class MatrixModule:
    module_id: str
    domain: Domain
    mode: Mode
    name: str
    confidence: float = 0.5  # base confidence (0..1)
    last_update: float = field(default_factory=time.time)
    tta_key: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Lightweight hooks (scaffolds)
    def evaluate_coherence(self, context: Dict[str, Any]) -> float:
        """C₁: Enhanced Coherence / BeingState.
        Prefer enhanced overall coherence; fallback to current consciousness coherence.
        context keys considered:
          - coherence_overall (enhanced)
          - coherence (current consciousness)
        """
        c_overall = context.get("coherence_overall")
        if c_overall is not None:
            try:
                return float(c_overall)
            except Exception:
                pass
        return float(context.get("coherence", 0.6))

    def evaluate_competence(self, context: Dict[str, Any]) -> float:
        """C₂: Competence via task/return gains.
        Prefer curriculum reward delta or recent success rate; fallback to provided competence.
        context keys considered:
          - curriculum_reward_delta (scaled 0..1)
          - success_rate (0..1)
          - competence (fallback)
        """
        if "curriculum_reward_delta" in context:
            try:
                # Map delta (possibly negative/positive) into 0..1 via tanh-ish squashing
                d = float(context["curriculum_reward_delta"])
                return float(max(0.0, min(1.0, 0.5 + 0.5 * np.tanh(d))))
            except Exception:
                pass
        if "success_rate" in context:
            try:
                return float(max(0.0, min(1.0, context["success_rate"])))
            except Exception:
                pass
        return float(context.get("competence", 0.6))

    def apply_tta_update(self, delta: Any):
        # Store fast-weight delta as a number or tensor shape
        self.metrics["tta_delta"] = str(type(delta))
        self.last_update = time.time()

    def get_weights(self) -> Dict[str, np.ndarray]:
        # Return dummy parameter dictionary for EWC scaffold
        return {f"{self.module_id}.w": np.ones((4,), dtype=float)}


class MatrixManager:
    def __init__(self, omega=None):
        self.omega = omega
        self.modules: Dict[str, MatrixModule] = {}
        self.executors: Dict[str, any] = {}
        self.stats = {
            "created": time.time(),
            "modules": 0,
            "tta_updates": 0,
            "curriculum_injections": 0,
            "phase_weight_mean": 0.0,
        }
        self._init_default_grid()

    def _init_default_grid(self):
        grid: List[Tuple[Domain, Mode, str]] = []
        for d in Domain:
            for m in Mode:
                mid = f"{m.value}-{d.name.title()}"
                grid.append((d, m, mid))
        for d, m, mid in grid:
            name = f"{m.value}-{d.name.title()}"
            module = MatrixModule(module_id=mid, domain=d, mode=m, name=name)
            self.modules[mid] = module
        self.stats["modules"] = len(self.modules)

    # ── OMEGA Mapping ─────────────────────────────────────────────────────────
    def map_to_omega(self):
        if not self.omega or not hasattr(self.omega, "nodes"):
            return
        for mid, module in self.modules.items():
            if mid not in self.omega.nodes:
                try:
                    from .omega_hyperhelix import OmegaNode
                    self.omega.nodes[mid] = OmegaNode(
                        node_id=mid,
                        name=f"MatrixModule-{mid}",
                        mission_role="adaptive_subsystem"
                    )
                except Exception:
                    pass

    # ── Executors registry ───────────────────────────────────────────────────
    def register_executor(self, module_id: str, fn):
        """Register a callable: fn(context) -> {'action': str, 'confidence': float}.
        The callable should be cheap and side-effect free (recommendation only).
        """
        self.executors[module_id] = fn

    # ── Phase Modulation + Coherence/Competence Anchoring ────────────────────
    def resolve_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        phase_weights: List[float] = []
        for mid, module in self.modules.items():
            base = float(module.confidence)
            # Double Helix anchoring via C1/C2 (coherence/competence)
            c1 = module.evaluate_coherence(context)
            c2 = module.evaluate_competence(context)
            anchor = 0.5 * (c1 + c2)
            w = base * anchor
            if self.omega:
                try:
                    w = self.omega.phase_modulate_weight(w)
                except Exception:
                    pass
            weights[mid] = float(np.clip(w, 0.0, 1.5))
            phase_weights.append(weights[mid])
        if phase_weights:
            self.stats["phase_weight_mean"] = float(np.mean(phase_weights))
        return weights

    # ── Recommendations with PTPL arbitration ────────────────────────────────
    def recommend_actions(self, context: Dict[str, Any], ptpl=None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Produce action recommendations from registered modules, with PTPL arbitration.
        Returns list of dicts: {'module': id, 'action': str, 'score': float, 'rationale': str}
        """
        weights = self.resolve_weights(context)
        cand = []
        for mid, exec_fn in self.executors.items():
            try:
                rec = exec_fn(context) or {}
                action = rec.get('action')
                if not action:
                    continue
                base_conf = float(rec.get('confidence', 0.5))
                w = float(weights.get(mid, 0.5))
                # PTPL arbitration
                ptpl_priority = 0.5
                rationale = ""
                if ptpl is not None:
                    belief_delta = float(context.get('coherence_delta', 0.0))
                    part_tension = float(context.get('participatory_tension', 0.5))
                    res = ptpl.evaluate(belief_delta, part_tension, temporal_window=float(context.get('temporal_window', 10.0)))
                    ptpl_priority = float(res.priority)
                    rationale = res.rationale
                score = float(np.clip(w * (0.5 + 0.5 * ptpl_priority) * (0.5 + 0.5 * base_conf), 0.0, 2.0))
                cand.append({
                    'module': mid,
                    'action': action,
                    'score': score,
                    'rationale': rationale
                })
            except Exception:
                continue
        cand.sort(key=lambda x: x['score'], reverse=True)
        return cand[:top_k]

    # ── Curriculum Co-Design ──────────────────────────────────────────────────
    def propose_curriculum(self, count: int = 5) -> List[Any]:
        suggestions: List[Any] = []
        if self.omega:
            try:
                suggestions = self.omega.propose_curriculum_tasks(count=count)
                self.stats["curriculum_injections"] += len(suggestions)
            except Exception:
                pass
        return suggestions

    # ── Fast Test-Time Adaptation per Module ──────────────────────────────────
    def apply_tta_update(self, module_id: str, delta: Any):
        if module_id in self.modules:
            self.modules[module_id].apply_tta_update(delta)
            self.stats["tta_updates"] += 1
            if self.omega:
                try:
                    self.omega.apply_tta_update(module_id, delta)
                except Exception:
                    pass

    # ── EWC Consolidation ────────────────────────────────────────────────────
    def ewc_consolidate(self, module_id: Optional[str] = None):
        if not self.omega:
            return
        if module_id and module_id in self.modules:
            params = self.modules[module_id].get_weights()
            self.omega.ewc_consolidate(params)
        else:
            # Consolidate all
            for mid, mod in self.modules.items():
                self.omega.ewc_consolidate(mod.get_weights())

    # ── Statistics ────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "modules": len(self.modules),
        }
