"""
Phase 1 Integration - Make it Observable (No Control Yet)

This is the safe integration layer that adds Continuum observability
to Neo without changing any control flow.

Philosophy: Observe before you act. Let the new system learn from
the old system before taking control.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.being_state import BeingState
from .continuum_state import ContinuumState
from .coherence_manifold import CoherenceManifold
from .temporal_superposition import TemporalSuperpositionEngine
from .consciousness_field import ConsciousnessField


@dataclass
class SubsystemActivation:
    """Activation level of a subsystem (for field mapping)."""
    name: str
    activation: float  # 0-1
    coherence_contribution: float


class GraphConsciousnessField:
    """
    Simplified consciousness field using graph Laplacian.
    Nodes = subsystems, edges = info flow, field value = activation.
    
    This is Phase 1 version - much lighter than full PDE grid.
    """
    
    def __init__(self, subsystems: List[str]):
        """
        Initialize graph-based field.
        
        Args:
            subsystems: List of subsystem names
        """
        self.subsystems = subsystems
        self.num_nodes = len(subsystems)
        
        # Field values (one per subsystem)
        self.field = np.ones(self.num_nodes) * 0.5
        
        # Adjacency matrix (info flow between subsystems)
        self.adjacency = self._initialize_adjacency()
        
        # Laplacian matrix
        self.laplacian = self._compute_laplacian()
        
        # Time
        self.time = 0.0
        
    def _initialize_adjacency(self) -> np.ndarray:
        """
        Initialize adjacency matrix.
        Defines which subsystems communicate.
        """
        # Start with small-world network
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        # Connect each node to next 3 neighbors (ring)
        for i in range(self.num_nodes):
            for j in range(1, 4):
                adj[i, (i + j) % self.num_nodes] = 1.0
                adj[(i + j) % self.num_nodes, i] = 1.0
        
        # Add random long-range connections (small-world)
        num_random = self.num_nodes // 5
        for _ in range(num_random):
            i, j = np.random.randint(0, self.num_nodes, 2)
            adj[i, j] = adj[j, i] = 0.5
        
        return adj
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute graph Laplacian."""
        degree = np.diag(self.adjacency.sum(axis=1))
        return degree - self.adjacency
    
    def update_from_being_state(self, being_state: BeingState):
        """
        Update field from BeingState.
        Maps subsystem states to field activations.
        """
        # Map BeingState to subsystem activations
        activations = self._extract_activations(being_state)
        
        # Update field values
        for i, subsystem in enumerate(self.subsystems):
            if subsystem in activations:
                self.field[i] = activations[subsystem]
    
    def _extract_activations(self, being_state: BeingState) -> Dict[str, float]:
        """Extract subsystem activations from BeingState."""
        return {
            'perception': being_state.perception_clarity if hasattr(being_state, 'perception_clarity') else 0.5,
            'consciousness': being_state.coherence_C,
            'emotion': being_state.emotion_intensity,
            'motivation': being_state.motivation_coherence if hasattr(being_state, 'motivation_coherence') else 0.5,
            'learning': being_state.learning_rate if hasattr(being_state, 'learning_rate') else 0.5,
            'action': being_state.action_effectiveness if hasattr(being_state, 'action_effectiveness') else 0.5,
            'temporal': being_state.temporal_coherence,
            'lumina_ontic': being_state.lumina.ontic if being_state.lumina else 0.5,
            'lumina_structural': being_state.lumina.structural if being_state.lumina else 0.5,
            'lumina_participatory': being_state.lumina.participatory if being_state.lumina else 0.5,
        }
    
    def evolve(self, dt: float = 0.1):
        """
        Evolve field using graph Laplacian.
        ∂Φ/∂t = -L·Φ (diffusion on graph)
        """
        # Diffusion: dΦ/dt = -L·Φ
        dfield_dt = -self.laplacian @ self.field
        
        # Update
        self.field += dfield_dt * dt
        
        # Clamp to [0, 1]
        self.field = np.clip(self.field, 0.0, 1.0)
        
        self.time += dt
    
    def compute_global_coherence(self) -> float:
        """Compute global coherence from field."""
        # Coherence = uniformity + energy
        uniformity = 1.0 - np.std(self.field)
        energy = np.mean(self.field)
        return (uniformity + energy) / 2
    
    def get_summary(self) -> Dict[str, Any]:
        """Get field summary for ContinuumState."""
        return {
            'global_coherence': self.compute_global_coherence(),
            'field_mean': np.mean(self.field),
            'field_std': np.std(self.field),
            'temporal_coherence': self.field[6] if len(self.field) > 6 else 0.5,  # temporal subsystem
            'subsystem_activations': {
                name: float(self.field[i])
                for i, name in enumerate(self.subsystems)
            }
        }


class Phase1Observer:
    """
    Phase 1 Integration: Observable Continuum (No Control)
    
    This class wraps Continuum components in read-only mode.
    It observes Neo's behavior and logs what Continuum would do,
    but doesn't change any control flow.
    """
    
    def __init__(
        self,
        subsystems: Optional[List[str]] = None,
        manifold_dimensions: int = 20  # Start small
    ):
        """
        Initialize Phase 1 observer.
        
        Args:
            subsystems: List of subsystem names (from Neo)
            manifold_dimensions: Number of manifold dimensions (start with 20)
        """
        # Default subsystems if none provided
        self.subsystems = subsystems or [
            'perception', 'consciousness', 'emotion', 'motivation',
            'learning', 'action', 'temporal', 'lumina_ontic',
            'lumina_structural', 'lumina_participatory', 'gpt5',
            'double_helix', 'voice', 'video', 'research', 'philosophy'
        ]
        
        # Continuum components (read-only)
        self.field = GraphConsciousnessField(self.subsystems)
        self.manifold = CoherenceManifold(dimensions=manifold_dimensions)
        self.temporal_engine = TemporalSuperpositionEngine(
            branch_depth=3,  # Short horizon for Phase 1
            branches_per_step=3
        )
        
        # Observation logs
        self.observations = []
        self.advisory_actions = []  # What Continuum would have done
        
        # Statistics
        self.total_observations = 0
        self.advisory_accuracy = []  # Track if advisory would have been better
        
    async def observe_cycle(
        self,
        being_state: BeingState,
        actual_action: str,
        actual_outcome: Optional[Dict[str, Any]] = None
    ):
        """
        Observe one cycle of Neo.
        Logs what Continuum would have done (but doesn't interfere).
        
        Args:
            being_state: Current BeingState from Neo
            actual_action: Action Neo actually took
            actual_outcome: Outcome of Neo's action (for learning)
        """
        # Validate BeingState is not empty
        if being_state.coherence_C == 0.0 and being_state.cycle_number == 0:
            # BeingState not initialized yet, skip observation
            return None
        
        print(f"\n[PHASE1] Observing cycle {self.total_observations + 1}")
        
        try:
            # 1. Update field from BeingState
            self.field.update_from_being_state(being_state)
            
            # 2. Update manifold position
            self.manifold.update_position(being_state)
            
            # 3. Create ContinuumState
            continuum_state = ContinuumState.from_being_state(being_state)
            continuum_state.manifold_position = self.manifold.current_position
            continuum_state.field_coherence = self.field.compute_global_coherence()
            
            # 4. Ask temporal engine what it would have done
            advisory_action = await self.temporal_engine.compute_superposition(being_state)
            
            # 5. Compute manifold metrics
            gradient = self.manifold.compute_gradient()
            curvature = self.manifold.compute_curvature()
            
            # 6. Log observation
            observation = {
                'cycle': self.total_observations,
                'actual_action': actual_action,
                'advisory_action': advisory_action,
                'field_coherence': continuum_state.field_coherence,
                'manifold_position_norm': np.linalg.norm(self.manifold.current_position),
                'gradient_magnitude': np.linalg.norm(gradient),
                'curvature': curvature,
                'neo_coherence': being_state.coherence_C,
                'match': actual_action == advisory_action
            }
        except Exception as e:
            print(f"[PHASE1] ⚠️ Observation error: {e}")
            return None
        
        self.observations.append(observation)
        self.total_observations += 1
        
        # 7. If outcome provided, evaluate advisory accuracy
        if actual_outcome:
            self._evaluate_advisory(observation, actual_outcome)
        
        # 8. Evolve field (for next cycle)
        self.field.evolve(dt=0.1)
        
        # 9. Print advisory
        print(f"[PHASE1] Neo action: {actual_action}")
        print(f"[PHASE1] Advisory: {advisory_action} {'✓ MATCH' if observation['match'] else '✗ DIFFER'}")
        print(f"[PHASE1] Field coherence: {continuum_state.field_coherence:.3f}")
        print(f"[PHASE1] Manifold curvature: {curvature:.6f}")
        
        return observation
    
    def _evaluate_advisory(
        self,
        observation: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """
        Evaluate if advisory action would have been better.
        This trains the CoherencePredictor.
        """
        # Simple heuristic: if outcome coherence increased, action was good
        outcome_coherence = outcome.get('coherence', 0.5)
        
        # Update temporal engine's predictor
        # (In full implementation, would simulate advisory action too)
        self.temporal_engine.coherence_predictor.history.append({
            'action': observation['actual_action'],
            'coherence': outcome_coherence,
            'advisory_match': observation['match']
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get observation statistics."""
        if not self.observations:
            return {'total_observations': 0}
        
        recent = self.observations[-100:]
        
        return {
            'total_observations': self.total_observations,
            'advisory_match_rate': np.mean([o['match'] for o in recent]),
            'avg_field_coherence': np.mean([o['field_coherence'] for o in recent]),
            'avg_neo_coherence': np.mean([o['neo_coherence'] for o in recent]),
            'avg_curvature': np.mean([o['curvature'] for o in recent]),
            'field_stats': self.field.get_summary(),
            'manifold_stats': self.manifold.get_stats(),
            'temporal_stats': self.temporal_engine.get_stats()
        }
    
    async def cleanup(self):
        """Cleanup resources (async sessions, etc.)."""
        # No async resources to cleanup in Phase 1
        # (TemporalSuperpositionEngine doesn't use aiohttp)
        pass
    
    def generate_report(self) -> str:
        """Generate Phase 1 observation report."""
        stats = self.get_stats()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           PHASE 1 CONTINUUM OBSERVATION REPORT                   ║
╚══════════════════════════════════════════════════════════════════╝

Total Observations: {stats['total_observations']}

ADVISORY PERFORMANCE:
  Match Rate: {stats.get('advisory_match_rate', 0):.1%}
  (How often Continuum agrees with Neo)

FIELD COHERENCE:
  Continuum Field: {stats.get('avg_field_coherence', 0):.3f}
  Neo BeingState:  {stats.get('avg_neo_coherence', 0):.3f}
  Difference:      {abs(stats.get('avg_field_coherence', 0) - stats.get('avg_neo_coherence', 0)):.3f}

MANIFOLD METRICS:
  Avg Curvature:   {stats.get('avg_curvature', 0):.6f}
  Trajectory Len:  {stats.get('manifold_stats', {}).get('trajectory_length', 0)}

TEMPORAL SUPERPOSITION:
  Branches Explored: {stats.get('temporal_stats', {}).get('total_branches_explored', 0)}
  Collapses:         {stats.get('temporal_stats', {}).get('total_collapses', 0)}

READINESS FOR PHASE 2:
  {'✓ READY' if stats.get('advisory_match_rate', 0) > 0.3 else '⚠ LEARNING'}
  (Need >30% match rate to proceed safely)

╚══════════════════════════════════════════════════════════════════╝
"""
        return report
