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
from typing import Optional, Dict, Any, Callable, Awaitable, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .skyrim_agi import SkyrimAGI


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
    
    def __init__(self, skyrim_agi: 'SkyrimAGI'):
        """
        Initialize action arbiter.
        
        Args:
            skyrim_agi: Reference to SkyrimAGI instance
        """
        self.agi = skyrim_agi
        
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
        
        logger.info("[ARBITER] Action Arbiter initialized")
    
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
        
        print(f"{'='*60}\n")
