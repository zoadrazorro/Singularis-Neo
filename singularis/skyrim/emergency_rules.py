"""
Emergency Response Rules for Stuck Detection and Recovery

Implements fast-path rules that override normal planning when critical
situations are detected. These rules provide immediate responses to:
- Stuck/frozen states
- Low coherence (system confusion)
- Perception-action mismatches
- Repeated failures

This bridges the gap between consciousness (awareness) and agency (action).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class EmergencyLevel(Enum):
    """Enumerates the severity levels of an emergency situation."""
    CRITICAL = 3  # Immediate action override is required.
    HIGH = 2      # Strong recommendation for an alternative action.
    MEDIUM = 1    # Suggestion to modify behavior or re-evaluate.
    NONE = 0      # Normal operation, no emergency.


@dataclass
class EmergencyResponse:
    """Represents a recommended response to a detected emergency.

    Attributes:
        level: The severity of the emergency.
        action: The recommended action to take. Can be None if the response
                is to modify confidence rather than override the action.
        reason: A human-readable string explaining why the emergency was triggered.
        confidence_modifier: A float multiplier to apply to the confidence score
                             of the normally planned action.
        override: If True, the recommended action should immediately replace any
                  planned action.
    """
    level: EmergencyLevel
    action: str
    reason: str
    confidence_modifier: float
    override: bool


class EmergencyRules:
    """A system for detecting and responding to critical in-game situations.

    These rules are designed to be fast checks that run before more complex
    reasoning (like LLM calls). They can identify situations like being physically
    stuck, system confusion (low coherence), or repeated failures, and can
    override the normal planning process to force a recovery action.
    """
    
    def __init__(self):
        """Initializes the emergency rule system with default thresholds."""
        self.stuck_cycle_threshold = 3
        self.low_coherence_threshold = 0.25
        self.visual_similarity_stuck_threshold = 0.95
        self.confidence_reduction_threshold = 0.5
        
        # Tracking for stagnation detection
        self.cycles_since_strategy_change = 0
        self.last_strategy = None
    
    def evaluate_emergency_state(
        self,
        context: Dict[str, Any]
    ) -> Optional[EmergencyResponse]:
        """Evaluates the current system context against a set of emergency rules.

        The rules are checked in order of priority (critical, high, medium).
        The first rule that matches the context returns an EmergencyResponse.

        Args:
            context: A dictionary containing the current state of the system.
                     Expected keys include 'visual_similarity', 'recent_actions',
                     'coherence', 'action_confidence', 'sensorimotor_status',
                     and 'cycles_since_change'.

        Returns:
            An EmergencyResponse object if an emergency is detected, otherwise None.
        """
        # Check rules in priority order
        
        # CRITICAL: Stuck with high visual similarity
        response = self._rule_stuck_visual(context)
        if response:
            return response
        
        # HIGH: Perception-action mismatch
        response = self._rule_perception_action_mismatch(context)
        if response:
            return response
        
        # HIGH: Repeated action failure
        response = self._rule_repeated_failure(context)
        if response:
            return response
        
        # MEDIUM: Low coherence (system confusion)
        response = self._rule_low_coherence(context)
        if response:
            return response
        
        # MEDIUM: Long time without strategy change
        response = self._rule_strategy_stagnation(context)
        if response:
            return response
        
        return None
    
    def _rule_stuck_visual(self, context: Dict[str, Any]) -> Optional[EmergencyResponse]:
        """CRITICAL: Detects if the agent is physically stuck.

        This rule triggers if visual input has not changed significantly over
        several cycles despite repeated movement attempts.

        Args:
            context: The system state context.

        Returns:
            An EmergencyResponse with a recovery action, or None.
        """
        visual_sim = context.get('visual_similarity', 0.0)
        recent_actions = context.get('recent_actions', [])
        cycles_since_change = context.get('cycles_since_change', 0)
        
        if visual_sim > self.visual_similarity_stuck_threshold and len(recent_actions) >= 3:
            # Check for repeated movement actions
            movement_actions = ['move_forward', 'move_backward', 'explore']
            recent_movements = [a for a in recent_actions[-3:] if a in movement_actions]
            
            if len(recent_movements) >= 2 and cycles_since_change >= self.stuck_cycle_threshold:
                # Force unstuck action
                last_action = recent_actions[-1] if recent_actions else None
                
                if last_action in ['move_forward', 'explore']:
                    # Try interaction first, then rotation
                    return EmergencyResponse(
                        level=EmergencyLevel.CRITICAL,
                        action='activate',
                        reason='STUCK: High visual similarity with repeated forward movement. Try interaction.',
                        confidence_modifier=1.0,
                        override=True
                    )
                elif last_action == 'move_backward':
                    # Try rotation
                    return EmergencyResponse(
                        level=EmergencyLevel.CRITICAL,
                        action='turn_right',
                        reason='STUCK: High visual similarity with repeated backward movement. Try rotation.',
                        confidence_modifier=1.0,
                        override=True
                    )
                else:
                    # Default: jump to break state
                    return EmergencyResponse(
                        level=EmergencyLevel.CRITICAL,
                        action='jump',
                        reason='STUCK: High visual similarity. Break state with jump.',
                        confidence_modifier=1.0,
                        override=True
                    )
        
        return None
    
    def _rule_perception_action_mismatch(self, context: Dict[str, Any]) -> Optional[EmergencyResponse]:
        """HIGH: Detects a mismatch between low-level sensors and high-level plans.

        Triggers if sensorimotor feedback indicates a "STUCK" state, but the
        planner is still attempting to perform standard movements.

        Args:
            context: The system state context.

        Returns:
            An EmergencyResponse suggesting an interaction, or None.
        """
        sensorimotor_status = context.get('sensorimotor_status', '').upper()
        recent_actions = context.get('recent_actions', [])
        
        if 'STUCK' in sensorimotor_status and recent_actions:
            last_action = recent_actions[-1]
            movement_actions = ['move_forward', 'move_backward', 'explore', 'turn_left', 'turn_right']
            
            if last_action in movement_actions:
                # Perception says stuck, but we're trying to move
                # Try interaction to resolve
                return EmergencyResponse(
                    level=EmergencyLevel.HIGH,
                    action='activate',
                    reason='MISMATCH: Sensorimotor detects STUCK but planning chose movement. Try interaction.',
                    confidence_modifier=0.9,
                    override=True
                )
        
        return None
    
    def _rule_repeated_failure(self, context: Dict[str, Any]) -> Optional[EmergencyResponse]:
        """HIGH: Detects if the same action is being repeated without progress.

        Triggers if the last four actions were identical and system coherence
        has not improved, indicating a failure loop.

        Args:
            context: The system state context.

        Returns:
            An EmergencyResponse forcing an orthogonal action, or None.
        """
        recent_actions = context.get('recent_actions', [])
        coherence_history = context.get('coherence_history', [])
        
        if len(recent_actions) >= 4:
            last_four = recent_actions[-4:]
            if len(set(last_four)) == 1:  # All same action
                repeated_action = last_four[0]
                
                # Check if coherence improved
                coherence_improving = False
                if len(coherence_history) >= 2:
                    coherence_improving = coherence_history[-1] > coherence_history[-4] + 0.05
                
                if not coherence_improving:
                    # Repeated action with no improvement
                    # Suggest orthogonal action
                    if repeated_action in ['move_forward', 'explore']:
                        new_action = 'turn_right'
                    elif repeated_action in ['turn_left', 'turn_right']:
                        new_action = 'jump'
                    else:
                        new_action = 'move_backward'
                    
                    return EmergencyResponse(
                        level=EmergencyLevel.HIGH,
                        action=new_action,
                        reason=f'REPEATED FAILURE: {repeated_action} x4 with no progress. Force change.',
                        confidence_modifier=0.8,
                        override=True
                    )
        
        return None
    
    def _rule_low_coherence(self, context: Dict[str, Any]) -> Optional[EmergencyResponse]:
        """MEDIUM: Detects low system coherence, indicating confusion.

        If the system's internal state coherence is below a threshold but its
        action confidence is high, this rule reduces the confidence to prevent
        overconfident actions while confused.

        Args:
            context: The system state context.

        Returns:
            An EmergencyResponse that modifies confidence, or None.
        """
        coherence = context.get('coherence', 1.0)
        action_confidence = context.get('action_confidence', 0.5)
        
        if coherence < self.low_coherence_threshold and action_confidence > 0.7:
            # System is confused but overconfident
            return EmergencyResponse(
                level=EmergencyLevel.MEDIUM,
                action=None,  # Don't override action, just modify confidence
                reason=f'LOW COHERENCE: System coherence {coherence:.2f} < {self.low_coherence_threshold}. Reducing confidence.',
                confidence_modifier=0.5,  # Cut confidence in half
                override=False
            )
        
        return None
    
    def _rule_strategy_stagnation(self, context: Dict[str, Any]) -> Optional[EmergencyResponse]:
        """MEDIUM: Detects if the agent's strategy has not changed for a while.

        Triggers if the agent has been pursuing the same high-level strategy for
        too many cycles, suggesting it might be stuck in a strategic rut.

        Args:
            context: The system state context.

        Returns:
            An EmergencyResponse suggesting re-evaluation, or None.
        """
        cycles_since_change = context.get('cycles_since_change', 0)
        
        if cycles_since_change > 10:
            self.cycles_since_strategy_change = 0  # Reset
            
            return EmergencyResponse(
                level=EmergencyLevel.MEDIUM,
                action=None,
                reason=f'STAGNATION: {cycles_since_change} cycles without strategy change. Request re-evaluation.',
                confidence_modifier=0.7,
                override=False
            )
        
        return None
    
    def suggest_unstuck_action(
        self,
        last_action: Optional[str],
        available_actions: List[str]
    ) -> str:
        """Suggests a logical sequence of actions to try to get unstuck.

        Based on the last action attempted, it follows a predefined preference
        list (e.g., if moving forward failed, try 'activate', then 'turn_right').

        Args:
            last_action: The last action that was attempted.
            available_actions: A list of currently available actions.

        Returns:
            The name of the recommended unstuck action.
        """
        preferences = {
            'move_forward': ['activate', 'turn_right', 'jump'],
            'explore': ['activate', 'turn_around', 'jump'],
            'move_backward': ['turn_right', 'turn_left', 'jump'],
            'turn_left': ['jump', 'move_backward', 'activate'],
            'turn_right': ['jump', 'move_backward', 'activate'],
            'jump': ['turn_around', 'move_backward', 'activate'],
            'activate': ['turn_right', 'move_backward', 'jump'],
        }
        
        # Get preferences for last action
        preferred = preferences.get(last_action, ['activate', 'turn_right', 'jump'])
        
        # Return first available preference
        for action in preferred:
            if action in available_actions:
                return action
        
        # Fallback: any action except last
        for action in available_actions:
            if action != last_action:
                return action
        
        # Last resort: random available
        return available_actions[0] if available_actions else 'wait'
    
    def record_strategy_change(self, new_strategy: str):
        """Records a change in high-level strategy to reset the stagnation counter.

        Args:
            new_strategy: The name of the new strategy being adopted.
        """
        if new_strategy != self.last_strategy:
            self.cycles_since_strategy_change = 0
            self.last_strategy = new_strategy
        else:
            self.cycles_since_strategy_change += 1
