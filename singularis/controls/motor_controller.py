"""MotorController: translates HighLevelAction → actual button presses."""

import asyncio
import random
from typing import Any
from .action_space import HighLevelAction


class MotorController:
    """
    Maps abstract actions to concrete controller/keyboard inputs.
    
    This is the "hands and feet" layer - deterministic, reproducible
    physical behavior for each abstract action.
    """
    
    def __init__(self, actions_interface: Any, verbose: bool = True):
        """
        Initialize motor controller.
        
        Args:
            actions_interface: SkyrimActions or similar, with execute() method
            verbose: Print execution logs
        """
        self.actions = actions_interface
        self.verbose = verbose
        self.stats = {
            'total_executions': 0,
            'action_counts': {},
        }
    
    async def execute(self, action: HighLevelAction, duration: float = None):
        """
        Execute a high-level action via low-level controls.
        
        Args:
            action: Action to execute
            duration: Optional duration override
        """
        if self.verbose:
            print(f"[MOTOR] Executing {action.name}")
        
        self.stats['total_executions'] += 1
        self.stats['action_counts'][action.name] = self.stats['action_counts'].get(action.name, 0) + 1
        
        # Map to actual controller/keyboard actions
        try:
            if action == HighLevelAction.IDLE:
                await asyncio.sleep(0.5)
            
            elif action == HighLevelAction.LOOK_AROUND:
                await self._look_around()
            
            elif action == HighLevelAction.STEP_FORWARD:
                await self._step_forward(duration or 0.8)
            
            elif action == HighLevelAction.STEP_BACKWARD:
                await self._step_backward(duration or 0.6)
            
            elif action == HighLevelAction.STRAFE_LEFT:
                await self._strafe_left(duration or 0.5)
            
            elif action == HighLevelAction.STRAFE_RIGHT:
                await self._strafe_right(duration or 0.5)
            
            elif action == HighLevelAction.TURN_LEFT_SMALL:
                await self._turn_left_small()
            
            elif action == HighLevelAction.TURN_RIGHT_SMALL:
                await self._turn_right_small()
            
            elif action == HighLevelAction.TURN_LEFT_LARGE:
                await self._turn_left_large()
            
            elif action == HighLevelAction.TURN_RIGHT_LARGE:
                await self._turn_right_large()
            
            elif action == HighLevelAction.JUMP:
                await self._jump()
            
            elif action == HighLevelAction.SPRINT_FORWARD:
                await self._sprint_forward(duration or 1.5)
            
            elif action == HighLevelAction.QUICK_ATTACK:
                await self._quick_attack()
            
            elif action == HighLevelAction.POWER_ATTACK:
                await self._power_attack()
            
            elif action == HighLevelAction.BLOCK:
                await self._block(duration or 0.8)
            
            elif action == HighLevelAction.DODGE:
                await self._dodge()
            
            elif action == HighLevelAction.BASH:
                await self._bash()
            
            elif action == HighLevelAction.APPROACH_TARGET:
                await self._approach_target()
            
            elif action == HighLevelAction.RETREAT_FROM_TARGET:
                await self._retreat()
            
            elif action == HighLevelAction.CIRCLE_LEFT:
                await self._circle_left()
            
            elif action == HighLevelAction.CIRCLE_RIGHT:
                await self._circle_right()
            
            elif action == HighLevelAction.ACTIVATE:
                await self._activate()
            
            elif action == HighLevelAction.INTERACT:
                await self._activate()
            
            elif action == HighLevelAction.OPEN_INVENTORY:
                await self._open_inventory()
            
            elif action == HighLevelAction.OPEN_MAP:
                await self._open_map()
            
            elif action == HighLevelAction.CLOSE_MENU:
                await self._close_menu()
            
            elif action == HighLevelAction.USE_POTION_HEALTH:
                await self._use_health_potion()
            
            elif action == HighLevelAction.SNEAK:
                await self._sneak()
            
            else:
                print(f"[MOTOR] Unimplemented action: {action.name}, defaulting to STEP_FORWARD")
                await self._step_forward(0.5)
        
        except Exception as e:
            print(f"[MOTOR] Execution error for {action.name}: {e}")
    
    # === Implementation of individual actions ===
    
    async def _look_around(self):
        """Sweep camera left-right."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.LOOK_AROUND, duration=1.0))
    
    async def _step_forward(self, duration: float):
        """Walk forward."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.MOVE_FORWARD, duration=duration))
    
    async def _step_backward(self, duration: float):
        """Walk backward."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.MOVE_BACKWARD, duration=duration))
    
    async def _strafe_left(self, duration: float):
        """Strafe left."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.STRAFE_LEFT, duration=duration))
    
    async def _strafe_right(self, duration: float):
        """Strafe right."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.STRAFE_RIGHT, duration=duration))
    
    async def _turn_left_small(self):
        """Small left turn."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.LOOK_LEFT, duration=0.3))
    
    async def _turn_right_small(self):
        """Small right turn."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.LOOK_RIGHT, duration=0.3))
    
    async def _turn_left_large(self):
        """Large left turn."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.LOOK_LEFT, duration=0.8))
    
    async def _turn_right_large(self):
        """Large right turn."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.LOOK_RIGHT, duration=0.8))
    
    async def _jump(self):
        """Jump."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.JUMP, duration=0.3))
    
    async def _sprint_forward(self, duration: float):
        """Sprint forward."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.SPRINT, duration=duration))
    
    async def _quick_attack(self):
        """Quick attack."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.ATTACK, duration=0.3))
    
    async def _power_attack(self):
        """Power attack (hold attack button)."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.POWER_ATTACK, duration=0.6))
    
    async def _block(self, duration: float):
        """Hold block."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.BLOCK, duration=duration))
    
    async def _dodge(self):
        """Quick dodge (step back + small turn)."""
        await self._step_backward(0.3)
        if random.random() < 0.5:
            await self._turn_left_small()
        else:
            await self._turn_right_small()
    
    async def _bash(self):
        """Bash (block + attack)."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.BASH, duration=0.4))
    
    async def _approach_target(self):
        """Move toward target."""
        await self._step_forward(1.0)
    
    async def _retreat(self):
        """Retreat from target."""
        await self._step_backward(0.8)
    
    async def _circle_left(self):
        """Circle strafe left."""
        await self._strafe_left(0.6)
    
    async def _circle_right(self):
        """Circle strafe right."""
        await self._strafe_right(0.6)
    
    async def _activate(self):
        """Activate/interact."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.ACTIVATE, duration=0.2))
    
    async def _open_inventory(self):
        """Open inventory menu."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.INVENTORY, duration=0.2))
    
    async def _open_map(self):
        """Open map."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.MAP, duration=0.2))
    
    async def _close_menu(self):
        """Close menu/dialogue."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.BACK, duration=0.1))
    
    async def _use_health_potion(self):
        """Use health potion via quick-access."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.HEAL, duration=0.5))
    
    async def _sneak(self):
        """Toggle sneak."""
        from singularis.skyrim.actions import Action, ActionType
        await self.actions.execute(Action(ActionType.SNEAK, duration=0.2))
    
    def get_stats(self):
        """Get execution statistics."""
        return self.stats.copy()
