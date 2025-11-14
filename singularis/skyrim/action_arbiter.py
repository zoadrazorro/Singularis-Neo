"""
Action Arbiter - Single point of action execution with priority system.

Solves the multiple competing action executors problem by:
1. All action requests go through arbiter
2. Priority system (CRITICAL > HIGH > NORMAL > LOW)
3. Comprehensive validation before execution
4. Conflict resolution and preemption
5. Cancellation of outdated actions

Phase 2.1 & 2.2: ActionArbiter implementation
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable, List, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .skyrim_agi import SkyrimAGI
    from ..llm.gpt5_orchestrator import GPT5Orchestrator
    from ..core.being_state import BeingState


class ActionPriority(Enum):
    """Action priority levels."""
    CRITICAL = 4  # Survival (health <10%, falling, etc.)
    HIGH = 3      # Urgent (combat, stuck, etc.)
    NORMAL = 2    # Standard gameplay
    LOW = 1       # Background (exploration, idle)


@dataclass
class ActionRequest:
    """Request to execute an action."""
    action: str
    priority: ActionPriority
    source: str  # Which system requested it
    context: Dict[str, Any]  # Game state, perception, etc.
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: f"{time.time():.6f}")


@dataclass
class ActionResult:
    """Result of action execution."""
    action: str
    executed: bool
    success: bool
    reason: str  # Why it succeeded/failed/was rejected
    execution_time: float
    overrode_action: Optional[str] = None  # If we cancelled another action


class ActionArbiter:
    """
    Central arbiter for all action execution.
    
    Ensures:
    - Only one action executes at a time
    - Higher priority actions can preempt lower priority
    - Actions are validated before execution
    - Requesting systems get feedback
    
    Phase 2: Single point of control with priority system
    """
    
    def __init__(
        self,
        skyrim_agi: 'SkyrimAGI',
        gpt5_orchestrator: Optional['GPT5Orchestrator'] = None,
        enable_gpt5_coordination: bool = True
    ):
        """
        Initialize action arbiter.
        
        Args:
            skyrim_agi: Reference to SkyrimAGI instance
            gpt5_orchestrator: Optional GPT-5 orchestrator for coordination
            enable_gpt5_coordination: Whether to use GPT-5 for action coordination
        """
        self.agi = skyrim_agi
        self.gpt5 = gpt5_orchestrator
        self.enable_gpt5_coordination = enable_gpt5_coordination and gpt5_orchestrator is not None
        
        # Current execution
        self.current_action: Optional[ActionRequest] = None
        self.action_executing = False
        
        # Pending high-priority actions
        self.pending_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'executed': 0,
            'rejected': 0,
            'overridden': 0,
            'by_priority': {p: 0 for p in ActionPriority},
            'by_source': {},
            'rejection_reasons': {},
        }
        
        # Callbacks for requesting systems
        self.callbacks: Dict[str, Callable[[ActionResult], Awaitable[None]]] = {}
        
        # GPT-5 coordination stats (hybrid mode)
        self.gpt5_coordination_count = 0
        self.gpt5_coordination_time = 0.0
        self.local_coordination_count = 0  # Fast local arbitration count
        
        logger.info(
            f"[ARBITER] Action Arbiter initialized "
            f"(GPT-5 coordination: {'enabled' if self.enable_gpt5_coordination else 'disabled'})"
        )
    
    async def request_action(
        self,
        action: str,
        priority: ActionPriority,
        source: str,
        context: Dict[str, Any],
        callback: Optional[Callable[[ActionResult], Awaitable[None]]] = None
    ) -> ActionResult:
        """
        Request action execution.
        
        Args:
            action: Action to execute
            priority: Priority level
            source: Which system is requesting (e.g., 'reasoning_loop', 'fast_reactive')
            context: Game state, perception, etc.
            callback: Optional callback when action completes
            
        Returns:
            ActionResult with execution status
        """
        request = ActionRequest(
            action=action,
            priority=priority,
            source=source,
            context=context
        )
        
        self.stats['total_requests'] += 1
        self.stats['by_priority'][priority] += 1
        self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + 1
        
        if callback:
            self.callbacks[request.request_id] = callback
        
        # Validate request
        is_valid, reason = self._validate_request(request)
        if not is_valid:
            logger.warning(f"[ARBITER] Rejected {source} action '{action}': {reason}")
            self.stats['rejected'] += 1
            self.stats['rejection_reasons'][reason] = self.stats['rejection_reasons'].get(reason, 0) + 1
            
            result = ActionResult(
                action=action,
                executed=False,
                success=False,
                reason=f"Validation failed: {reason}",
                execution_time=0.0
            )
            await self._notify_callback(request.request_id, result)
            return result
        
        # Check if we should preempt current action
        if self.action_executing and self.current_action:
            if request.priority.value > self.current_action.priority.value:
                logger.warning(
                    f"[ARBITER] ⚡ {request.priority.name} action '{action}' "
                    f"preempting {self.current_action.priority.name} '{self.current_action.action}'"
                )
                # Cancel current action
                overridden_action = self.current_action.action
                await self._cancel_current_action()
                self.stats['overridden'] += 1
            else:
                # Lower or equal priority - reject
                result = ActionResult(
                    action=action,
                    executed=False,
                    success=False,
                    reason=f"Lower priority than current action ({self.current_action.action})",
                    execution_time=0.0
                )
                self.stats['rejected'] += 1
                await self._notify_callback(request.request_id, result)
                return result
        
        # Execute action
        result = await self._execute_action(request)
        
        # Notify callback
        await self._notify_callback(request.request_id, result)
        
        return result
    
    def _validate_request(self, request: ActionRequest) -> tuple[bool, str]:
        """
        Validate action request with comprehensive checks.
        
        Phase 2.2: Comprehensive validation
        
        Returns:
            (is_valid, reason)
        """
        # Check 1: Perception freshness
        perception_timestamp = request.context.get('perception_timestamp', time.time())
        age = time.time() - perception_timestamp
        
        # CRITICAL actions can use slightly older data (emergency)
        max_age = 5.0 if request.priority == ActionPriority.CRITICAL else 2.0
        
        if age > max_age:
            return (False, f"Perception too old ({age:.1f}s > {max_age}s)")
        
        # Check 2: Game state consistency
        if hasattr(self.agi, 'current_perception') and self.agi.current_perception:
            current_scene = str(self.agi.current_perception.get('scene_type', ''))
            requested_scene = str(request.context.get('scene_type', ''))
            
            # Allow scene mismatch for CRITICAL actions
            if request.priority != ActionPriority.CRITICAL:
                if current_scene != requested_scene and current_scene and requested_scene:
                    return (False, f"Scene changed: {requested_scene} → {current_scene}")
        
        # Check 3: Action availability
        action = request.action
        game_state = request.context.get('game_state')
        
        # Basic sanity checks
        if not action:
            return (False, "No action specified")
        
        # Can't move if in menu
        if game_state:
            in_menu = getattr(game_state, 'in_menu', False)
            movement_actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'jump', 'sprint']
            if in_menu and action in movement_actions:
                return (False, f"Can't {action} while in menu")
        
        # Check 4: Health-based validation
        if game_state:
            health = getattr(game_state, 'health', 100)
            
            # Don't allow non-critical actions when health is critical
            if health < 15 and request.priority != ActionPriority.CRITICAL:
                return (False, f"Health critical ({health:.0f}), only CRITICAL actions allowed")
            
            # Don't attack when health is very low
            if health < 25 and action in ['attack', 'power_attack']:
                return (False, f"Health too low ({health:.0f}) for offensive actions")
        
        # Check 5: Combat context validation
        if game_state:
            in_combat = getattr(game_state, 'in_combat', False)
            
            # Don't open menus during combat (unless CRITICAL for healing)
            menu_actions = ['open_inventory', 'open_map']
            if in_combat and action in menu_actions:
                if request.priority != ActionPriority.CRITICAL:
                    return (False, "Can't open menus during combat (unless CRITICAL)")
        
        # Check 6: Repeated action detection
        if hasattr(self.agi, 'action_history'):
            recent = self.agi.action_history[-5:] if len(self.agi.action_history) >= 5 else []
            
            # Don't allow same action 5 times in a row (stuck loop)
            if len(recent) >= 5 and all(a == action for a in recent):
                return (False, f"Action '{action}' repeated 5x (stuck loop prevention)")
        
        return (True, "Valid")
    
    async def _execute_action(self, request: ActionRequest) -> ActionResult:
        """Execute validated action request."""
        self.current_action = request
        self.action_executing = True
        
        start_time = time.time()
        success = False
        reason = "Unknown"
        
        try:
            logger.info(
                f"[ARBITER] ▶ Executing {request.priority.name} action: {request.action} "
                f"(from {request.source})"
            )
            
            # Execute through AGI's action system
            scene_type = request.context.get('scene_type', 'unknown')
            await self.agi._execute_action(request.action, scene_type)
            
            success = True
            reason = "Executed successfully"
            self.stats['executed'] += 1
            
        except Exception as e:
            success = False
            reason = f"Execution failed: {e}"
            logger.error(f"[ARBITER] Execution failed: {e}")
        
        finally:
            self.action_executing = False
            self.current_action = None
        
        execution_time = time.time() - start_time
        
        return ActionResult(
            action=request.action,
            executed=True,
            success=success,
            reason=reason,
            execution_time=execution_time
        )
    
    async def _cancel_current_action(self):
        """Cancel currently executing action (if possible)."""
        if self.current_action:
            logger.warning(f"[ARBITER] Cancelling action: {self.current_action.action}")
            # Note: Actual cancellation may not be possible for all actions
            # This sets the flag so the action loop knows it was overridden
            self.current_action = None
    
    def _should_use_gpt5_coordination(
        self,
        being_state: 'BeingState',
        candidate_actions: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if GPT-5 coordination is needed (hybrid mode).
        
        Use fast local arbitration when:
        - Single candidate action (no conflict)
        - All candidates have same priority
        - High subsystem consensus (>80%)
        - Low temporal issues (coherence >0.8, unclosed <5)
        
        Use GPT-5 coordination when:
        - Multiple conflicting actions
        - Low subsystem consensus
        - Temporal coherence issues
        - Stuck loop detected
        
        Returns:
            True if GPT-5 coordination needed, False for fast local arbitration
        """
        # Always use local for single action
        if len(candidate_actions) <= 1:
            return False
        
        # Check for priority conflicts
        priorities = [a.get('priority', 'NORMAL') for a in candidate_actions]
        if len(set(priorities)) > 1:
            # Mixed priorities - need coordination
            return True
        
        # Check subsystem consensus
        confidences = [a.get('confidence', 0.0) for a in candidate_actions]
        max_confidence = max(confidences) if confidences else 0.0
        confidence_spread = max_confidence - min(confidences) if confidences else 0.0
        
        # Use GPT-5 if: very low confidence OR very high spread (disagreement)
        # Lowered thresholds for more aggressive fast local arbitration
        if max_confidence < 0.4:
            # Very low confidence - need coordination
            return True
        
        if confidence_spread > 0.5:
            # Very high disagreement - need coordination
            return True
        
        # Check temporal coherence (more lenient)
        if being_state.temporal_coherence < 0.6 or being_state.unclosed_bindings > 10:
            # Serious temporal issues - need coordination
            return True
        
        # Check for stuck loops (more lenient)
        if being_state.stuck_loop_count >= 3:
            # Definitely stuck - need coordination
            return True
        
        # Check subsystem freshness
        stale_count = 0
        for subsystem in ['sensorimotor', 'action_plan', 'memory', 'emotion']:
            if not being_state.is_subsystem_fresh(subsystem, max_age=5.0):
                stale_count += 1
        
        if stale_count >= 2:
            # Multiple stale subsystems - need coordination
            return True
        
        # Otherwise, use fast local arbitration
        return False
    
    async def coordinate_action_decision(
        self,
        being_state: 'BeingState',
        candidate_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Coordinate action decision through GPT-5 orchestrator.
        
        Phase 3.3: GPT-5 Orchestrator Coordination (Hybrid Mode)
        
        Uses fast local arbitration for simple cases, GPT-5 only for complex
        decisions requiring meta-cognitive coordination.
        
        Args:
            being_state: Current unified state
            candidate_actions: List of candidate actions with metadata
            
        Returns:
            Selected action with reasoning, or None if coordination fails
        """
        if not self.enable_gpt5_coordination or not self.gpt5:
            logger.debug("[ARBITER] GPT-5 coordination disabled, skipping")
            return None
        
        # Hybrid mode: Check if GPT-5 coordination is needed
        if not self._should_use_gpt5_coordination(being_state, candidate_actions):
            # Fast local arbitration - select highest confidence action
            if candidate_actions:
                selected = max(candidate_actions, key=lambda x: x.get('confidence', 0.0))
                selected['coordination_method'] = 'local_fast'
                self.local_coordination_count += 1
                logger.info(
                    f"[ARBITER] ⚡ Fast local arbitration: {selected['action']} "
                    f"(confidence: {selected.get('confidence', 0.0):.2f})"
                )
                return selected
            return None
        
        start_time = time.time()
        
        try:
            # Gather subsystem states
            subsystem_states = {
                'sensorimotor': being_state.get_subsystem_data('sensorimotor'),
                'action_plan': being_state.get_subsystem_data('action_plan'),
                'memory': being_state.get_subsystem_data('memory'),
                'emotion': being_state.get_subsystem_data('emotion'),
                'consciousness': {
                    'coherence_C': being_state.coherence_C,
                    'phi_hat': being_state.phi_hat,
                    'unity_index': being_state.unity_index,
                    'conflicts': being_state.consciousness_conflicts
                },
                'temporal': {
                    'temporal_coherence': being_state.temporal_coherence,
                    'unclosed_bindings': being_state.unclosed_bindings,
                    'stuck_loop_count': being_state.stuck_loop_count
                },
                'global': {
                    'global_coherence': being_state.global_coherence,
                    'cycle_number': being_state.cycle_number,
                    'last_action': being_state.last_action
                }
            }
            
            # Format candidate actions
            actions_summary = []
            for i, action in enumerate(candidate_actions):
                actions_summary.append(
                    f"{i+1}. {action.get('action', 'unknown')} "
                    f"(priority: {action.get('priority', 'NORMAL')}, "
                    f"source: {action.get('source', 'unknown')}, "
                    f"confidence: {action.get('confidence', 0.0):.2f})"
                )
            
            # Build coordination request
            actions_text = "\n".join(actions_summary)
            content = f"""Action Coordination Request:
            
Current Cycle: {being_state.cycle_number}
Global Coherence: {being_state.global_coherence:.3f}
Temporal Coherence: {being_state.temporal_coherence:.3f}
Unclosed Bindings: {being_state.unclosed_bindings}
Stuck Loop Count: {being_state.stuck_loop_count}

Subsystem Status:
- Sensorimotor: {subsystem_states['sensorimotor'].get('status', 'UNKNOWN')} (age: {subsystem_states['sensorimotor'].get('age', 999):.1f}s)
- Action Plan: {subsystem_states['action_plan'].get('current', 'none')} (confidence: {subsystem_states['action_plan'].get('confidence', 0.0):.2f})
- Memory: {subsystem_states['memory'].get('pattern_count', 0)} patterns, {len(subsystem_states['memory'].get('recommendations', []))} recommendations
- Emotion: {being_state.primary_emotion} (intensity: {being_state.emotion_intensity:.2f})

Candidate Actions:
{actions_text}

Which action should be selected? Consider:
1. Subsystem consensus and conflicts
2. Temporal coherence and stuck loop prevention
3. Global coherence optimization
4. Freshness of subsystem data

Provide: Selected action number (or 0 for none), reasoning, and confidence."""
            
            # Query GPT-5
            logger.info("[ARBITER] Requesting GPT-5 action coordination...")
            
            response = await self.gpt5.send_message(
                system_id="action_arbiter",
                message_type="action_coordination",
                content=content,
                metadata={
                    'cycle': being_state.cycle_number,
                    'candidate_count': len(candidate_actions),
                    'global_coherence': being_state.global_coherence
                }
            )
            
            # Parse response
            selected_action = None
            
            # Try to extract action number from response
            response_text = response.response_text.lower()
            for i, action in enumerate(candidate_actions):
                if f"action {i+1}" in response_text or f"{i+1}." in response_text[:100]:
                    selected_action = action.copy()
                    selected_action['gpt5_reasoning'] = response.reasoning or response.response_text
                    selected_action['gpt5_confidence'] = response.confidence
                    break
            
            # Track stats
            elapsed = time.time() - start_time
            self.gpt5_coordination_count += 1
            self.gpt5_coordination_time += elapsed
            
            logger.info(
                f"[ARBITER] GPT-5 coordination complete: "
                f"{'selected' if selected_action else 'no selection'} "
                f"({elapsed:.2f}s)"
            )
            
            return selected_action
            
        except Exception as e:
            logger.error(f"[ARBITER] GPT-5 coordination failed: {e}")
            return None
    
    def prevent_conflicting_action(
        self,
        action: str,
        being_state: 'BeingState',
        priority: ActionPriority
    ) -> tuple[bool, str]:
        """
        Prevent conflicting actions before execution.
        
        Phase 3.4: Conflict Prevention
        
        Checks for conflicts with:
        - Current system state (stuck loops, low health, etc.)
        - Temporal binding state (unclosed loops)
        - Subsystem recommendations
        - Recent action history
        
        Args:
            action: Action to check
            being_state: Current unified state
            priority: Action priority
            
        Returns:
            (is_allowed, reason) - True if action should proceed
        """
        # Check 1: Stuck loop prevention
        if being_state.stuck_loop_count >= 3:
            # Only allow actions that break the loop
            loop_breaking_actions = ['turn_left', 'turn_right', 'jump', 'move_backward']
            if action not in loop_breaking_actions and priority != ActionPriority.CRITICAL:
                return (False, f"Stuck loop detected ({being_state.stuck_loop_count} cycles), action '{action}' would continue loop")
        
        # Check 2: Temporal coherence check
        if being_state.temporal_coherence < 0.5 and being_state.unclosed_bindings > 5:
            # System is losing temporal coherence - be conservative
            if priority == ActionPriority.LOW:
                return (False, f"Low temporal coherence ({being_state.temporal_coherence:.2f}), rejecting LOW priority actions")
        
        # Check 3: Subsystem conflict detection
        conflicts = []
        
        # Sensorimotor conflict
        if being_state.is_subsystem_fresh('sensorimotor'):
            sensorimotor_data = being_state.get_subsystem_data('sensorimotor')
            if sensorimotor_data.get('status') == 'STUCK':
                movement_actions = ['move_forward', 'sprint']
                if action in movement_actions:
                    conflicts.append("sensorimotor: system is stuck, movement may not work")
        
        # Action plan conflict
        if being_state.is_subsystem_fresh('action_plan'):
            action_plan_data = being_state.get_subsystem_data('action_plan')
            planned_action = action_plan_data.get('current')
            if planned_action and planned_action != action:
                confidence = action_plan_data.get('confidence', 0.0)
                if confidence > 0.7 and priority != ActionPriority.CRITICAL:
                    conflicts.append(f"action_plan: recommends '{planned_action}' (confidence: {confidence:.2f})")
        
        # Memory conflict
        if being_state.is_subsystem_fresh('memory'):
            memory_data = being_state.get_subsystem_data('memory')
            recommendations = memory_data.get('recommendations', [])
            if recommendations and action not in recommendations:
                if priority == ActionPriority.LOW:
                    conflicts.append(f"memory: recommends {recommendations}, not '{action}'")
        
        # Emotion conflict
        if being_state.is_subsystem_fresh('emotion'):
            emotion_data = being_state.get_subsystem_data('emotion')
            emotion_recs = emotion_data.get('recommendations', [])
            if emotion_recs and action not in emotion_recs:
                if being_state.emotion_intensity > 0.8:
                    conflicts.append(f"emotion: high intensity ({being_state.emotion_intensity:.2f}), recommends {emotion_recs}")
        
        # Check 4: Health-based conflicts
        if hasattr(being_state, 'game_state') and being_state.game_state:
            health = being_state.game_state.get('health', 100)
            if health < 20:
                aggressive_actions = ['attack', 'power_attack', 'sprint']
                if action in aggressive_actions and priority != ActionPriority.CRITICAL:
                    conflicts.append(f"health: critical health ({health:.0f}), aggressive action risky")
        
        # Evaluate conflicts
        if conflicts:
            # CRITICAL actions override all conflicts
            if priority == ActionPriority.CRITICAL:
                logger.warning(
                    f"[ARBITER] CRITICAL action '{action}' proceeding despite conflicts: {conflicts}"
                )
                return (True, "CRITICAL priority overrides conflicts")
            
            # HIGH priority can override 1-2 conflicts
            if priority == ActionPriority.HIGH and len(conflicts) <= 2:
                logger.info(
                    f"[ARBITER] HIGH priority action '{action}' proceeding with {len(conflicts)} conflicts"
                )
                return (True, f"HIGH priority overrides {len(conflicts)} conflicts")
            
            # Otherwise, block the action
            conflict_summary = "; ".join(conflicts)
            logger.warning(
                f"[ARBITER] Blocking action '{action}' due to conflicts: {conflict_summary}"
            )
            return (False, f"Conflicts detected: {conflict_summary}")
        
        return (True, "No conflicts detected")
    
    def ensure_temporal_binding_closure(
        self,
        being_state: 'BeingState',
        temporal_tracker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Ensure temporal binding loops close properly.
        
        Phase 3.5: Temporal Binding Closure
        
        Tracks closure rate and provides recommendations to improve it.
        Target: >95% closure rate
        
        Args:
            being_state: Current unified state
            temporal_tracker: Optional temporal coherence tracker
            
        Returns:
            Dict with closure metrics and recommendations
        """
        # Get temporal binding stats from BeingState
        temporal_coherence = being_state.temporal_coherence
        unclosed_bindings = being_state.unclosed_bindings
        stuck_loop_count = being_state.stuck_loop_count
        
        # Calculate closure rate (inverse of unclosed ratio)
        # If we have temporal_tracker, use its stats
        closure_rate = 0.0
        if temporal_tracker and hasattr(temporal_tracker, 'get_statistics'):
            stats = temporal_tracker.get_statistics()
            unclosed_ratio = stats.get('unclosed_ratio', 0.0)
            closure_rate = 1.0 - unclosed_ratio
        else:
            # Estimate from BeingState
            # Assume we want <5 unclosed bindings for good closure
            if unclosed_bindings <= 5:
                closure_rate = 0.95
            elif unclosed_bindings <= 10:
                closure_rate = 0.85
            else:
                closure_rate = max(0.0, 1.0 - (unclosed_bindings / 20.0))
        
        # Determine status
        status = "EXCELLENT" if closure_rate >= 0.95 else \
                 "GOOD" if closure_rate >= 0.85 else \
                 "FAIR" if closure_rate >= 0.70 else \
                 "POOR"
        
        # Generate recommendations
        recommendations = []
        
        if closure_rate < 0.95:
            recommendations.append("Increase action execution frequency to close loops faster")
        
        if unclosed_bindings > 10:
            recommendations.append(f"High unclosed bindings ({unclosed_bindings}), prioritize loop closure")
        
        if stuck_loop_count >= 3:
            recommendations.append(f"Stuck loop detected ({stuck_loop_count} cycles), force loop-breaking action")
        
        if temporal_coherence < 0.7:
            recommendations.append(f"Low temporal coherence ({temporal_coherence:.2f}), improve perception-action linkage")
        
        # Check for stale subsystem data (can prevent loop closure)
        stale_subsystems = []
        for subsystem in ['sensorimotor', 'action_plan', 'memory', 'emotion']:
            if not being_state.is_subsystem_fresh(subsystem, max_age=5.0):
                age = being_state.get_subsystem_age(subsystem)
                stale_subsystems.append(f"{subsystem} ({age:.1f}s)")
        
        if stale_subsystems:
            recommendations.append(f"Stale subsystems may prevent closure: {', '.join(stale_subsystems)}")
        
        result = {
            'closure_rate': closure_rate,
            'status': status,
            'unclosed_bindings': unclosed_bindings,
            'temporal_coherence': temporal_coherence,
            'stuck_loop_count': stuck_loop_count,
            'recommendations': recommendations,
            'meets_target': closure_rate >= 0.95
        }
        
        # Log if below target
        if closure_rate < 0.95:
            logger.warning(
                f"[ARBITER] Temporal binding closure below target: "
                f"{closure_rate:.1%} (target: 95%), "
                f"unclosed: {unclosed_bindings}, "
                f"recommendations: {len(recommendations)}"
            )
        else:
            logger.debug(
                f"[ARBITER] Temporal binding closure: {closure_rate:.1%} ✓"
            )
        
        return result
    
    async def _notify_callback(self, request_id: str, result: ActionResult):
        """Notify requesting system of result."""
        if request_id in self.callbacks:
            callback = self.callbacks.pop(request_id)
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"[ARBITER] Callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arbiter statistics."""
        total = max(self.stats['total_requests'], 1)
        executed = max(self.stats['executed'], 1)
        
        return {
            **self.stats,
            'rejection_rate': self.stats['rejected'] / total,
            'override_rate': self.stats['overridden'] / executed,
            'success_rate': self.stats['executed'] / total,
            'currently_executing': self.action_executing,
        }
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("ACTION ARBITER STATISTICS")
        print(f"{'='*60}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Executed: {stats['executed']} ({stats['success_rate']:.1%})")
        print(f"Rejected: {stats['rejected']} (Rate: {stats['rejection_rate']:.1%})")
        print(f"Overridden: {stats['overridden']} (Rate: {stats['override_rate']:.1%})")
        print(f"Currently Executing: {stats['currently_executing']}")
        
        print(f"\nBy Priority:")
        for priority, count in stats['by_priority'].items():
            print(f"  {priority.name}: {count}")
        
        print(f"\nBy Source:")
        for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {source}: {count}")
        
        if stats['rejection_reasons']:
            print(f"\nTop Rejection Reasons:")
            for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {reason}: {count}")
        
        if self.enable_gpt5_coordination:
            total_coordinations = self.gpt5_coordination_count + self.local_coordination_count
            print(f"\nHybrid Coordination (Speed Optimized):")
            print(f"  Total decisions: {total_coordinations}")
            print(f"  ⚡ Fast local: {self.local_coordination_count} ({self.local_coordination_count/total_coordinations*100:.1f}%)" if total_coordinations > 0 else "  ⚡ Fast local: 0")
            print(f"  🧠 GPT-5 Mini: {self.gpt5_coordination_count} ({self.gpt5_coordination_count/total_coordinations*100:.1f}%)" if total_coordinations > 0 else "  🧠 GPT-5 Mini: 0")
            if self.gpt5_coordination_count > 0:
                avg_time = self.gpt5_coordination_time / self.gpt5_coordination_count
                print(f"  Avg GPT-5 time: {avg_time:.2f}s")
            if total_coordinations > 0:
                speed_improvement = (self.local_coordination_count / total_coordinations) * 100
                print(f"  Speed improvement: ~{speed_improvement:.0f}% decisions instant")
        
        print(f"{'='*60}\n")
