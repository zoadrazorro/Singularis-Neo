"""
State Coordination System

Addresses the critical state mismatch problem where different subsystems
have contradictory beliefs about reality.

Problem Example:
- Game State: "scene: inventory"
- Sensorimotor: "outdoor environment with wooden structures"

This is epistemic incoherence - the system doesn't know what reality it's in.

Solution:
- Track all subsystem views of state
- Detect mismatches
- Resolve conflicts via voting, recency, or authority
- Provide unified canonical state to all subsystems
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class ConflictType(Enum):
    """Enumerates the types of state conflicts that can occur between subsystems."""
    SCENE_MISMATCH = "scene_mismatch"  # Different scenes reported
    LOCATION_MISMATCH = "location_mismatch"  # Different locations
    COMBAT_MISMATCH = "combat_mismatch"  # Disagreement on combat status
    MENU_MISMATCH = "menu_mismatch"  # Menu vs non-menu
    TEMPORAL_LAG = "temporal_lag"  # One subsystem using stale data


@dataclass
class StateConflict:
    """Represents a detected conflict between the states reported by different subsystems.

    Attributes:
        conflict_type: The type of conflict, as defined by the `ConflictType` enum.
        subsystems: A list of the names of the subsystems involved in the conflict.
        values: A dictionary mapping subsystem names to their conflicting state values.
        severity: A float from 0.0 to 1.0 indicating how critical the mismatch is.
        timestamp: The time when the conflict was detected.
    """
    conflict_type: ConflictType
    subsystems: List[str]
    values: Dict[str, Any]
    severity: float  # 0-1, how critical is this mismatch
    timestamp: float
    
    def __str__(self):
        return (f"{self.conflict_type.value}: "
                f"{', '.join(f'{k}={v}' for k, v in self.values.items())} "
                f"(severity={self.severity:.2f})")


@dataclass
class SubsystemView:
    """Represents a subsystem's view of the current state at a specific point in time.

    Attributes:
        subsystem_name: The name of the subsystem providing the state view.
        state: The dictionary representing the state as seen by this subsystem.
        timestamp: The time when this state view was generated.
        confidence: A float from 0.0 to 1.0 indicating the subsystem's confidence
                    in the accuracy of its view.
    """
    subsystem_name: str
    state: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0  # How confident is this subsystem about its view
    
    def age(self) -> float:
        """Calculates the age of this state view in seconds.

        Returns:
            The time elapsed since the view was created.
        """
        return time.time() - self.timestamp
    
    def is_stale(self, threshold: float = 2.0) -> bool:
        """Determines if this state view is too old to be considered reliable.

        Args:
            threshold: The age in seconds beyond which the view is considered stale.

        Returns:
            True if the view is stale, False otherwise.
        """
        return self.age() > threshold


class StateCoordinator:
    """Coordinates state across multiple subsystems to ensure epistemic coherence.

    This class acts as a central authority for truth about the game state. It
    receives state updates from various subsystems, detects and resolves any
    conflicts or disagreements, and maintains a single, unified "canonical"
    state that all other parts of the system can rely on. This prevents logical
    contradictions and ensures all components are working from a consistent
    understanding of reality.
    """
    
    def __init__(
        self,
        staleness_threshold: float = 2.0,
        conflict_history_size: int = 100
    ):
        """Initializes the StateCoordinator.

        Args:
            staleness_threshold: The time in seconds after which a subsystem's
                                 state view is considered too old and is discarded
                                 from conflict resolution.
            conflict_history_size: The maximum number of past conflicts to store
                                   in the history log.
        """
        self.staleness_threshold = staleness_threshold
        
        # Current views from each subsystem
        self.subsystem_views: Dict[str, SubsystemView] = {}
        
        # Canonical state (resolved truth)
        self.canonical_state: Dict[str, Any] = {}
        self.canonical_timestamp: float = 0.0
        
        # Conflict tracking
        self.active_conflicts: List[StateConflict] = []
        self.conflict_history: List[StateConflict] = []
        self.conflict_history_size = conflict_history_size
        
        # Subsystem authority weights (which subsystems to trust more)
        self.authority_weights: Dict[str, float] = {
            'perception': 1.0,  # Direct observation is authoritative
            'sensorimotor': 0.9,  # Visual analysis is very reliable
            'action_planning': 0.7,  # May use slightly stale state
            'game_state': 1.0,  # Direct game state read is ground truth
            'memory': 0.5,  # Memory may be outdated
        }
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'stale_views_discarded': 0,
        }
    
    def update(
        self,
        subsystem: str,
        state: Dict[str, Any],
        confidence: float = 1.0
    ):
        """Updates the state view for a given subsystem and triggers conflict resolution.

        This is the main entry point for subsystems to report their understanding
        of the current state. After receiving an update, the coordinator detects
        and resolves any resulting conflicts, then updates the canonical state.

        Args:
            subsystem: The name of the subsystem providing the update.
            state: The state dictionary from that subsystem.
            confidence: The subsystem's confidence in its provided state (0.0 to 1.0).
        """
        view = SubsystemView(
            subsystem_name=subsystem,
            state=state.copy(),
            timestamp=time.time(),
            confidence=confidence
        )
        
        self.subsystem_views[subsystem] = view
        self.stats['total_updates'] += 1
        
        # Check for conflicts
        conflicts = self.detect_conflicts()
        
        if conflicts:
            self.active_conflicts = conflicts
            self.stats['conflicts_detected'] += len(conflicts)
            
            # Log conflicts
            print(f"\n[STATE-COORD] ⚠️  EPISTEMIC CONFLICTS DETECTED:")
            for conflict in conflicts:
                print(f"[STATE-COORD]   • {conflict}")
            
            # Resolve conflicts
            self.resolve_conflicts(conflicts)
        else:
            self.active_conflicts = []
        
        # Update canonical state
        self._update_canonical_state()
    
    def detect_conflicts(self) -> List[StateConflict]:
        """Detects conflicts between the current, fresh views of all subsystems.

        It compares key state variables (like scene, combat status, and menu status)
        across all non-stale subsystem views.

        Returns:
            A list of `StateConflict` objects representing any detected discrepancies.
        """
        conflicts = []
        
        # Remove stale views first
        fresh_views = {
            name: view for name, view in self.subsystem_views.items()
            if not view.is_stale(self.staleness_threshold)
        }
        
        stale_count = len(self.subsystem_views) - len(fresh_views)
        if stale_count > 0:
            self.stats['stale_views_discarded'] += stale_count
            print(f"[STATE-COORD] Discarded {stale_count} stale views")
        
        if len(fresh_views) < 2:
            return conflicts  # Need at least 2 views to have a conflict
        
        # Check scene type mismatches
        scenes = {}
        for name, view in fresh_views.items():
            scene = view.state.get('scene') or view.state.get('scene_type')
            if scene:
                if hasattr(scene, 'value'):  # Enum
                    scene = scene.value
                scenes[name] = scene
        
        if len(set(scenes.values())) > 1:
            # Multiple different scenes reported
            conflict = StateConflict(
                conflict_type=ConflictType.SCENE_MISMATCH,
                subsystems=list(scenes.keys()),
                values=scenes,
                severity=0.9,  # High severity - fundamental disagreement
                timestamp=time.time()
            )
            conflicts.append(conflict)
        
        # Check combat status mismatches
        combat_states = {}
        for name, view in fresh_views.items():
            in_combat = view.state.get('in_combat')
            if in_combat is not None:
                combat_states[name] = in_combat
        
        if len(set(combat_states.values())) > 1:
            conflict = StateConflict(
                conflict_type=ConflictType.COMBAT_MISMATCH,
                subsystems=list(combat_states.keys()),
                values=combat_states,
                severity=0.8,
                timestamp=time.time()
            )
            conflicts.append(conflict)
        
        # Check menu status mismatches
        menu_states = {}
        for name, view in fresh_views.items():
            in_menu = view.state.get('in_menu') or view.state.get('in_dialogue')
            if in_menu is not None:
                menu_states[name] = in_menu
        
        if len(set(menu_states.values())) > 1:
            conflict = StateConflict(
                conflict_type=ConflictType.MENU_MISMATCH,
                subsystems=list(menu_states.keys()),
                values=menu_states,
                severity=0.7,
                timestamp=time.time()
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[StateConflict]):
        """Resolves a list of detected conflicts.

        The resolution strategy involves scoring each conflicting view based on the
        authority of its source subsystem, its recency, and its confidence. The
        view with the highest score is declared the "winner," and its version of
        the state is considered the truth for the conflicting variable.

        Args:
            conflicts: The list of `StateConflict` objects to resolve.
        """
        for conflict in conflicts:
            print(f"\n[STATE-COORD] Resolving: {conflict.conflict_type.value}")
            
            # Get views involved in conflict
            involved_views = [
                (name, self.subsystem_views[name])
                for name in conflict.subsystems
                if name in self.subsystem_views
            ]
            
            # Score each view
            scores = []
            for name, view in involved_views:
                authority = self.authority_weights.get(name, 0.5)
                recency = 1.0 / (1.0 + view.age())  # Decay with age
                confidence = view.confidence
                
                score = authority * 0.5 + recency * 0.3 + confidence * 0.2
                scores.append((name, view, score))
            
            # Sort by score (highest first)
            scores.sort(key=lambda x: x[2], reverse=True)
            
            # Winner is highest score
            winner_name, winner_view, winner_score = scores[0]
            
            print(f"[STATE-COORD] Resolution scores:")
            for name, view, score in scores:
                print(f"[STATE-COORD]   {name}: {score:.3f} "
                      f"(auth={self.authority_weights.get(name, 0.5):.2f}, "
                      f"age={view.age():.1f}s, conf={view.confidence:.2f})")
            
            print(f"[STATE-COORD] ✓ Resolved: trusting '{winner_name}' "
                  f"(score={winner_score:.3f})")
            
            self.stats['conflicts_resolved'] += 1
            
            # Store in history
            self.conflict_history.append(conflict)
            if len(self.conflict_history) > self.conflict_history_size:
                self.conflict_history.pop(0)
    
    def _update_canonical_state(self):
        """Updates the canonical state by performing a weighted vote for each state variable."""
        if not self.subsystem_views:
            return
        
        # Start with empty canonical state
        canonical = {}
        
        # Collect all fields across all views
        all_fields = set()
        for view in self.subsystem_views.values():
            all_fields.update(view.state.keys())
        
        # For each field, compute weighted vote
        for field in all_fields:
            values = []
            weights = []
            
            for name, view in self.subsystem_views.items():
                if field in view.state:
                    authority = self.authority_weights.get(name, 0.5)
                    recency = 1.0 / (1.0 + view.age())
                    confidence = view.confidence
                    
                    weight = authority * 0.5 + recency * 0.3 + confidence * 0.2
                    
                    values.append(view.state[field])
                    weights.append(weight)
            
            if not values:
                continue
            
            # If all values are the same, easy
            if len(set(str(v) for v in values)) == 1:
                canonical[field] = values[0]
            else:
                # Take value with highest weight
                max_idx = weights.index(max(weights))
                canonical[field] = values[max_idx]
        
        self.canonical_state = canonical
        self.canonical_timestamp = time.time()
    
    def get_canonical_state(self) -> Dict[str, Any]:
        """Returns the current, resolved canonical state.

        This state represents the system's single source of truth and should be
        used by all subsystems for decision-making.

        Returns:
            A copy of the canonical state dictionary.
        """
        return self.canonical_state.copy()
    
    def get_coherence(self) -> float:
        """Calculates the current epistemic coherence of the system.

        Coherence is a score from 0.0 (total disagreement) to 1.0 (perfect agreement).
        It is based on the number and severity of active conflicts, as well as the
        staleness of subsystem views.

        Returns:
            The coherence score.
        """
        if len(self.subsystem_views) < 2:
            return 1.0  # Can't have conflicts with <2 views
        
        # Conflict penalty
        if not self.active_conflicts:
            conflict_score = 1.0
        else:
            # Average severity of active conflicts
            avg_severity = sum(c.severity for c in self.active_conflicts) / len(self.active_conflicts)
            conflict_score = 1.0 - avg_severity
        
        # Staleness penalty
        stale_count = sum(
            1 for view in self.subsystem_views.values()
            if view.is_stale(self.staleness_threshold)
        )
        staleness_ratio = stale_count / len(self.subsystem_views)
        staleness_score = 1.0 - staleness_ratio
        
        # Overall coherence
        coherence = 0.7 * conflict_score + 0.3 * staleness_score
        
        return coherence
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the state coordinator's performance.

        Returns:
            A dictionary containing statistics on updates, conflicts, coherence,
            and the age of the canonical state.
        """
        return {
            **self.stats,
            'active_conflicts': len(self.active_conflicts),
            'subsystem_count': len(self.subsystem_views),
            'coherence': self.get_coherence(),
            'canonical_age': time.time() - self.canonical_timestamp if self.canonical_timestamp > 0 else 0
        }
    
    def __repr__(self):
        coherence = self.get_coherence()
        conflicts = len(self.active_conflicts)
        views = len(self.subsystem_views)
        
        return (f"StateCoordinator(views={views}, conflicts={conflicts}, "
                f"coherence={coherence:.3f})")
