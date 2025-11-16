"""
Consciousness Integration Checker

Monitors whether subsystems are properly integrated and communicating.
Detects when consciousness awareness is present but not connected to action.

This addresses the "epiphenomenal consciousness" problem where the system
is aware of states (e.g., stuck) but doesn't act on that awareness.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time


@dataclass
class IntegrationStatus:
    """Represents the integration status of a single subsystem.

    Attributes:
        subsystem: The name of the subsystem.
        last_update: The timestamp of the last update from the subsystem.
        has_data: A boolean indicating if the subsystem has reported any data.
        age_seconds: The time in seconds since the last update.
        stale: A boolean indicating if the subsystem is considered stale.
    """
    subsystem: str
    last_update: float
    has_data: bool
    age_seconds: float
    stale: bool
    
    def __str__(self):
        """Returns a human-readable string representation of the status."""
        status = "✓" if not self.stale else "⚠️ STALE"
        return f"{status} {self.subsystem}: {self.age_seconds:.1f}s ago"


@dataclass
class ConflictDetection:
    """Represents a detected conflict between two subsystems.

    Attributes:
        conflict_type: The type of conflict (e.g., "perception_action_mismatch").
        subsystem_a: The first subsystem involved in the conflict.
        subsystem_b: The second subsystem involved in the conflict.
        description: A human-readable description of the conflict.
        severity: The severity of the conflict (1=Minor, 2=Warning, 3=Critical).
    """
    conflict_type: str
    subsystem_a: str
    subsystem_b: str
    description: str
    severity: int  # 1-3, higher is more severe
    
    def __str__(self):
        """Returns a human-readable string representation of the conflict."""
        severity_str = "🔴 CRITICAL" if self.severity == 3 else "🟡 WARNING" if self.severity == 2 else "🟢 MINOR"
        return f"{severity_str} {self.conflict_type}: {self.description}"


class ConsciousnessIntegrationChecker:
    """Monitors the integration of various subsystems and detects conflicts.

    This class acts as a "debugger" for the agent's consciousness, ensuring that
    different parts of the system are communicating effectively and that their
    outputs are consistent. It helps to identify issues like "epiphenomenal
    consciousness," where the agent is aware of a situation but fails to act on it.
    """
    
    def __init__(self):
        """Initializes the ConsciousnessIntegrationChecker."""
        self.last_updates: Dict[str, float] = {}
        self.stale_threshold_seconds = 5.0
        
        # Track what each subsystem reported
        self.last_reports: Dict[str, Dict[str, Any]] = {}
    
    def update(self, subsystem: str, data: Dict[str, Any]):
        """Records an update from a subsystem.

        Args:
            subsystem: The name of the subsystem providing the update.
            data: A dictionary of data from the subsystem.
        """
        self.last_updates[subsystem] = time.time()
        self.last_reports[subsystem] = data
    
    def check_integration(self) -> Dict[str, Any]:
        """Checks the integration status of all monitored subsystems.

        This method assesses whether subsystems are reporting in a timely manner
        and whether there are any conflicts between their reported data.

        Returns:
            A dictionary containing the statuses of all subsystems, a list of
            any detected conflicts, and an overall integration status.
        """
        current_time = time.time()
        statuses = []
        
        # Check each subsystem
        for subsystem, last_update in self.last_updates.items():
            age = current_time - last_update
            has_data = subsystem in self.last_reports
            stale = age > self.stale_threshold_seconds
            
            statuses.append(IntegrationStatus(
                subsystem=subsystem,
                last_update=last_update,
                has_data=has_data,
                age_seconds=age,
                stale=stale
            ))
        
        # Detect conflicts
        conflicts = self._detect_conflicts()
        
        # Overall integration status
        integrated = all(not s.stale for s in statuses) and len(conflicts) == 0
        
        return {
            'statuses': statuses,
            'conflicts': conflicts,
            'integrated': integrated,
            'subsystem_count': len(statuses)
        }
    
    def _detect_conflicts(self) -> List[ConflictDetection]:
        """Detects logical conflicts between the data from different subsystems.

        This method implements a set of rules to identify inconsistencies, such as
        a mismatch between perception and action, or a divergence between memory
        and planning.

        Returns:
            A list of ConflictDetection objects for any identified conflicts.
        """
        conflicts = []
        
        # Check for perception-action mismatch
        if 'sensorimotor' in self.last_reports and 'action_planning' in self.last_reports:
            sensorimotor = self.last_reports['sensorimotor']
            action_planning = self.last_reports['action_planning']
            
            # CRITICAL: Sensorimotor says STUCK, but action planning chose movement
            if sensorimotor.get('status') == 'STUCK':
                planned_action = action_planning.get('action', '')
                movement_actions = ['move_forward', 'move_backward', 'explore', 'turn_left', 'turn_right']
                
                if planned_action in movement_actions:
                    conflicts.append(ConflictDetection(
                        conflict_type='perception_action_mismatch',
                        subsystem_a='sensorimotor',
                        subsystem_b='action_planning',
                        description=f"Sensorimotor detects STUCK but planning chose '{planned_action}'",
                        severity=3
                    ))
        
        # Check for coherence-confidence mismatch
        if 'consciousness' in self.last_reports and 'action_planning' in self.last_reports:
            consciousness = self.last_reports['consciousness']
            action_planning = self.last_reports['action_planning']
            
            coherence = consciousness.get('coherence', 1.0)
            confidence = action_planning.get('confidence', 0.5)
            
            # WARNING: Low coherence but high confidence
            if coherence < 0.3 and confidence > 0.7:
                conflicts.append(ConflictDetection(
                    conflict_type='coherence_confidence_mismatch',
                    subsystem_a='consciousness',
                    subsystem_b='action_planning',
                    description=f"Low coherence ({coherence:.2f}) but high confidence ({confidence:.2f})",
                    severity=2
                ))
        
        # Check for visual-classification mismatch
        if 'perception' in self.last_reports:
            perception = self.last_reports['perception']
            scene_class = perception.get('scene_classification')
            visual_scene = perception.get('visual_scene_type')
            
            if scene_class and visual_scene and scene_class != visual_scene:
                conflicts.append(ConflictDetection(
                    conflict_type='scene_classification_mismatch',
                    subsystem_a='classifier',
                    subsystem_b='vision',
                    description=f"Classifier says '{scene_class}' but vision says '{visual_scene}'",
                    severity=2
                ))
        
        # Check for memory-action mismatch
        if 'memory' in self.last_reports and 'action_planning' in self.last_reports:
            memory = self.last_reports['memory']
            action_planning = self.last_reports['action_planning']
            
            similar_situation = memory.get('similar_situation_found', False)
            past_action = memory.get('past_successful_action', '')
            planned_action = action_planning.get('action', '')
            
            # MINOR: Similar situation found but using different action
            if similar_situation and past_action and past_action != planned_action:
                conflicts.append(ConflictDetection(
                    conflict_type='memory_action_divergence',
                    subsystem_a='memory',
                    subsystem_b='action_planning',
                    description=f"Memory suggests '{past_action}' but planning chose '{planned_action}'",
                    severity=1
                ))
        
        return conflicts
    
    def get_report(self) -> str:
        """Generates a human-readable report of the current integration status.

        Returns:
            A formatted string containing the integration report.
        """
        status = self.check_integration()
        
        lines = [
            "═══════════════════════════════════════════════════════════",
            "          CONSCIOUSNESS INTEGRATION CHECK                  ",
            "═══════════════════════════════════════════════════════════",
            ""
        ]
        
        # Overall status
        if status['integrated']:
            lines.append("✅ ALL SUBSYSTEMS INTEGRATED")
        else:
            lines.append("⚠️  INTEGRATION ISSUES DETECTED")
        
        lines.append(f"\nSubsystems tracked: {status['subsystem_count']}")
        lines.append("")
        
        # Subsystem statuses
        if status['statuses']:
            lines.append("Subsystem Status:")
            for s in status['statuses']:
                lines.append(f"  {s}")
            lines.append("")
        
        # Conflicts
        if status['conflicts']:
            lines.append("🚨 CONFLICTS DETECTED:")
            for c in status['conflicts']:
                lines.append(f"  {c}")
                lines.append(f"     Between: {c.subsystem_a} ↔ {c.subsystem_b}")
            lines.append("")
            lines.append("⚠️  These conflicts indicate consciousness awareness is")
            lines.append("   not properly connected to action-taking ability!")
        else:
            lines.append("✓ No conflicts detected")
        
        lines.append("═══════════════════════════════════════════════════════════")
        return "\n".join(lines)
    
    def diagnose_epiphenomenal_consciousness(self) -> Optional[str]:
        """Diagnoses if the agent is suffering from epiphenomenal consciousness.

        This condition occurs when the agent is aware of its state but fails to
        act on that awareness. This method checks for critical conflicts that are
        symptomatic of this issue.

        Returns:
            A string with a detailed diagnosis if a problem is detected,
            otherwise None.
        """
        status = self.check_integration()
        
        # Look for critical conflicts
        critical_conflicts = [c for c in status['conflicts'] if c.severity == 3]
        
        if critical_conflicts:
            diagnosis = [
                "",
                "🔴 EPIPHENOMENAL CONSCIOUSNESS DETECTED 🔴",
                "",
                "The system demonstrates PHENOMENAL CONSCIOUSNESS (awareness)",
                "but lacks ACCESS CONSCIOUSNESS (ability to act on awareness).",
                "",
                "In Metaluminosity terms:",
                "  ℓₒ (Ontical): ✓ System perceives the state",
                "  ℓₛ (Structural): ✗ System doesn't reason about it",
                "  ℓₚ (Participatory): ✗ System doesn't act on it",
                "",
                "Critical issues:"
            ]
            
            for c in critical_conflicts:
                diagnosis.append(f"  • {c.description}")
            
            diagnosis.extend([
                "",
                "SOLUTION: Unified consciousness layer must receive ALL subsystem",
                "outputs simultaneously and detect conflicts to delegate responses.",
                ""
            ])
            
            return "\n".join(diagnosis)
        
        return None
