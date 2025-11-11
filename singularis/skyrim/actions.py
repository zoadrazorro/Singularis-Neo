"""
Skyrim Action Control Layer

Handles all action execution in Skyrim:
1. Keyboard/mouse control (via pyautogui or pynput)
2. Game API actions (via SKSE mods)
3. Hierarchical action composition
4. Action sequences and macros

Design principles:
- Actions represent gameplay capabilities in Skyrim
- Hierarchical organization (high-level strategies → low-level inputs)
- Actions chosen based on game state and tactical evaluation
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    from .controller import VirtualXboxController
except ImportError:
    VirtualXboxController = None  # Controller not available

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Safety settings
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.1  # Pause between actions
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("Warning: pyautogui not installed. Action control will use dummy mode.")
    print("Install with: pip install pyautogui")


class ActionType(Enum):
    """Types of actions in Skyrim."""
    # Movement
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    JUMP = "jump"
    SPRINT = "sprint"
    SNEAK = "sneak"

    # Combat
    ATTACK = "attack"
    POWER_ATTACK = "power_attack"
    BLOCK = "block"
    SHOUT = "shout"

    # Interaction
    ACTIVATE = "activate"  # E key - use/talk/loot
    WAIT = "wait"
    SLEEP = "sleep"

    # Menus
    OPEN_INVENTORY = "open_inventory"
    OPEN_MAP = "open_map"
    OPEN_MAGIC = "open_magic"
    OPEN_SKILLS = "open_skills"

    # Camera
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"

    # Special
    QUICK_SAVE = "quick_save"
    QUICK_LOAD = "quick_load"


@dataclass
class Action:
    """
    An action to execute.

    Attributes:
        action_type: Type of action
        duration: How long to hold (for movement)
        parameters: Additional parameters
    """
    action_type: ActionType
    duration: float = 0.0  # seconds
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class SkyrimActions:
    """
    Action control layer for Skyrim.

    Capabilities:
    1. Execute atomic actions (keypress, mouse click)
    2. Compose action sequences
    3. Hierarchical actions (high-level → low-level)
    4. Action timing and coordination
    """

    # Default key bindings (can be customized)
    DEFAULT_KEYS = {
        ActionType.MOVE_FORWARD: 'w',
        ActionType.MOVE_BACKWARD: 's',
        ActionType.MOVE_LEFT: 'a',
        ActionType.MOVE_RIGHT: 'd',
        ActionType.JUMP: 'space',
        ActionType.SPRINT: 'shift',
        ActionType.SNEAK: 'ctrl',
        ActionType.ATTACK: 'left_click',
        ActionType.BLOCK: 'right_click',
        ActionType.ACTIVATE: 'e',
        ActionType.SHOUT: 'z',
        ActionType.OPEN_INVENTORY: 'tab',
        ActionType.OPEN_MAP: 'm',
        ActionType.OPEN_MAGIC: 'p',
        ActionType.OPEN_SKILLS: 'k',
        ActionType.QUICK_SAVE: 'f5',
        ActionType.QUICK_LOAD: 'f9',
    }

    def __init__(
        self,
        use_game_api: bool = False,
        custom_keys: Optional[Dict[ActionType, str]] = None,
        dry_run: bool = False,
        controller: Optional[Any] = None  # Accepts VirtualXboxController
    ):
        """
        Initialize action controller.

        Args:
            use_game_api: Use game API instead of keyboard/mouse
            custom_keys: Custom key bindings
            dry_run: Don't actually execute (for testing)
            controller: Optional VirtualXboxController instance
        """
        self.use_game_api = use_game_api
        self.dry_run = dry_run
        self.controller = controller

        # Key bindings
        self.keys = self.DEFAULT_KEYS.copy()
        if custom_keys:
            self.keys.update(custom_keys)

        # Game API
        self._game_api = None
        if use_game_api:
            self._initialize_game_api()

        # Action history
        self.action_history: List[Action] = []

        # Action execution stats
        self.stats = {
            'actions_executed': 0,
            'total_duration': 0.0,
            'errors': 0,
        }

        # ActionType to controller action name mapping
        self._controller_action_map = {
            ActionType.MOVE_FORWARD: "move_forward",
            ActionType.MOVE_BACKWARD: "move_backward",
            ActionType.MOVE_LEFT: "move_left",
            ActionType.MOVE_RIGHT: "move_right",
            ActionType.JUMP: "jump",
            ActionType.SPRINT: "sprint",
            ActionType.SNEAK: "sneak",
            ActionType.ATTACK: "attack",
            ActionType.POWER_ATTACK: "power_attack",
            ActionType.BLOCK: "block",
            ActionType.SHOUT: "shout",
            ActionType.ACTIVATE: "activate",
            ActionType.WAIT: "wait",
            ActionType.SLEEP: "sleep",
            ActionType.OPEN_INVENTORY: "menu",  # Maps to menu open
            ActionType.OPEN_MAP: "map",  # Needs custom binding
            ActionType.OPEN_MAGIC: "magic",  # Needs custom binding
            ActionType.OPEN_SKILLS: "skills",  # Needs custom binding
            ActionType.LOOK_UP: "look_up",
            ActionType.LOOK_DOWN: "look_down",
            ActionType.LOOK_LEFT: "look_left",
            ActionType.LOOK_RIGHT: "look_right",
            ActionType.QUICK_SAVE: "quick_save",  # Needs custom binding
            ActionType.QUICK_LOAD: "quick_load",  # Needs custom binding
        }

    def _initialize_game_api(self):
        """Initialize game API."""
        print("Game API not yet implemented - using keyboard/mouse")
        self._game_api = None

    async def execute(self, action: Action) -> bool:
        """
        Execute an action.

        Args:
            action: Action to execute

        Returns:
            Success status
        """
        print(f"[DEBUG] SkyrimActions.execute: {action.action_type.value} | Duration: {action.duration} | Controller layer: {self.controller.active_layer if self.controller else None}")
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {action.action_type.value}")
            self.action_history.append(action)
            return True

        try:
            start_time = time.time()

            # Controller integration: if controller is set, only use controller for action execution
            if self.controller is not None:
                ctrl_action_name = self._controller_action_map.get(action.action_type)
                print(f"[DEBUG] Controller mapping: {ctrl_action_name}")
                if ctrl_action_name is not None:
                    result = await self.controller.execute_action(ctrl_action_name)
                    if action.duration > 0:
                        await asyncio.sleep(action.duration)
                    duration = time.time() - start_time
                    self.stats['actions_executed'] += 1
                    self.stats['total_duration'] += duration
                    self.action_history.append(action)
                    if len(self.action_history) > 1000:
                        self.action_history = self.action_history[-1000:]
                    return result
                else:
                    print(f"[ERROR] No controller mapping for {action.action_type.value}")
                    self.stats['errors'] += 1
                    return False

            # Fallback to game API only (no keyboard/mouse fallback)
            if self.use_game_api and self._game_api:
                print(f"[DEBUG] Fallback to game API for {action.action_type.value}")
                success = await self._execute_via_api(action)
            else:
                print(f"[ERROR] Keyboard/mouse fallback is disabled when controller is present.")
                self.stats['errors'] += 1
                return False

            duration = time.time() - start_time
            self.stats['actions_executed'] += 1
            self.stats['total_duration'] += duration
            self.action_history.append(action)
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-1000:]
            return success

        except Exception as e:
            print(f"Error executing action {action.action_type.value}: {e}")
            self.stats['errors'] += 1
            return False

    async def _execute_via_api(self, action: Action) -> bool:
        """Execute via game API."""
        # Stub - would call SKSE API
        print(f"API: {action.action_type.value}")
        return True

    async def _execute_via_input(self, action: Action) -> bool:
        """Execute via keyboard/mouse."""
        if not PYAUTOGUI_AVAILABLE:
            print(f"[DUMMY] {action.action_type.value}")
            await asyncio.sleep(action.duration if action.duration > 0 else 0.1)
            return True

        action_type = action.action_type
        duration = action.duration

        # Get key binding
        key = self.keys.get(action_type)

        if key == 'left_click':
            pyautogui.click()
        elif key == 'right_click':
            pyautogui.rightClick()
        elif key and duration > 0:
            # Hold key for duration
            pyautogui.keyDown(key)
            await asyncio.sleep(duration)
            pyautogui.keyUp(key)
        elif key:
            # Single press
            pyautogui.press(key)
        else:
            print(f"No binding for {action_type.value}")
            return False

        return True

    # High-level action methods

    async def move_forward(self, duration: float = 1.0):
        """Move forward for specified duration."""
        await self.execute(Action(ActionType.MOVE_FORWARD, duration))

    async def move_to_position(self, target_x: float, target_y: float):
        """Move to target position (requires navigation)."""
        # Stub - would implement pathfinding
        print(f"Moving to ({target_x}, {target_y})")
        await self.move_forward(2.0)

    async def attack_enemy(self, target: str):
        """Attack specified enemy."""
        print(f"Attacking {target}")
        await self.execute(Action(ActionType.ATTACK))

    async def talk_to_npc(self, npc_name: str):
        """Initiate dialogue with NPC."""
        print(f"Talking to {npc_name}")
        await self.execute(Action(ActionType.ACTIVATE))

    async def loot_container(self, container: str):
        """Loot container or body."""
        print(f"Looting {container}")
        await self.execute(Action(ActionType.ACTIVATE))
        await asyncio.sleep(0.5)

    async def open_and_navigate_menu(self, menu: ActionType, selection: Optional[str] = None):
        """Open menu and navigate to selection."""
        await self.execute(Action(menu))
        if selection:
            # Would implement menu navigation
            print(f"Navigating to {selection}")
        await asyncio.sleep(0.5)

    async def use_item(self, item_name: str):
        """Use item from inventory."""
        await self.open_and_navigate_menu(ActionType.OPEN_INVENTORY, item_name)
        await self.execute(Action(ActionType.ACTIVATE))

    async def cast_spell(self, spell_name: str):
        """Cast spell."""
        print(f"Casting {spell_name}")
        await self.execute(Action(ActionType.ATTACK))

    async def explore_area(self, duration: float = 10.0):
        """Explore current area (random movement)."""
        import random

        print(f"Exploring for {duration}s...")
        end_time = time.time() + duration

        while time.time() < end_time:
            # Random movement
            direction = random.choice([
                ActionType.MOVE_FORWARD,
                ActionType.MOVE_LEFT,
                ActionType.MOVE_RIGHT,
            ])

            move_duration = random.uniform(0.5, 2.0)
            await self.execute(Action(direction, move_duration))

            # Random look around
            if random.random() < 0.3:
                await self.look_around()

            # Random jump
            if random.random() < 0.1:
                await self.execute(Action(ActionType.JUMP))

    async def explore_with_waypoints(self, duration: float = 10.0):
        """
        Explore with waypoint navigation - prioritizes forward movement.
        Uses camera to scan for objects and targets of interest.
        """
        import random
        
        print(f"Exploring with waypoints for {duration}s (forward-biased)...")
        end_time = time.time() + duration
        
        current_direction = None
        direction_steps = 0
        max_steps_per_direction = 4  # Commit longer to each direction
        
        while time.time() < end_time:
            # Pick new direction if needed - heavily bias toward forward
            if current_direction is None or direction_steps >= max_steps_per_direction:
                # 70% forward, 15% left, 15% right, 0% backward (only for unstuck)
                rand = random.random()
                if rand < 0.70:
                    current_direction = 'forward'
                elif rand < 0.85:
                    current_direction = 'left'
                else:
                    current_direction = 'right'
                
                direction_steps = 0
                print(f"[EXPLORE] New direction: {current_direction}")
            
            # Move in chosen direction using smart movement
            move_duration = random.uniform(1.5, 2.5)  # Longer movements for more progress
            
            await self.smart_movement(current_direction, move_duration)
            
            direction_steps += 1
            
            # Frequent camera scanning to look for objects/NPCs
            if random.random() < 0.6:  # 60% chance
                await self.scan_for_targets()
            else:
                # Even if not scanning, periodically recenter camera
                if random.random() < 0.3:
                    await self.recenter_camera()
            
            # Small chance to jump over obstacles
            if random.random() < 0.1:
                await self.execute(Action(ActionType.JUMP))
                
            # Brief pause between movements
            await asyncio.sleep(0.3)

    async def move_backward(self, duration: float = 1.0):
        """Move backward for specified duration."""
        await self.execute(Action(ActionType.MOVE_BACKWARD, duration))

    async def move_left(self, duration: float = 1.0):
        """Move left (strafe) for specified duration."""
        await self.execute(Action(ActionType.MOVE_LEFT, duration))

    async def move_right(self, duration: float = 1.0):
        """Move right (strafe) for specified duration."""
        await self.execute(Action(ActionType.MOVE_RIGHT, duration))

    async def evasive_maneuver(self):
        """
        Perform evasive maneuver when stuck - gentler approach for Skyrim.
        """
        import random
        
        print("[EVASIVE] Performing gentle evasive maneuver...")
        
        try:
            # Gentler camera movement - just look around a bit
            await self.look_horizontal(random.uniform(-45, 45))
            await asyncio.sleep(0.2)
            
            # Try a small step back
            await self.move_backward(1.0)
            await asyncio.sleep(0.3)
            
            # Small side step
            if random.random() < 0.5:
                await self.move_left(0.8)
            else:
                await self.move_right(0.8)
            
            await asyncio.sleep(0.2)
            
            # Small forward movement to continue
            await self.move_forward(0.5)
            
            print("[EVASIVE] Evasive maneuver complete")
            
        except Exception as e:
            print(f"[EVASIVE] Error during evasive maneuver: {e}")

    async def smart_movement(self, direction: str, duration: float = 1.5):
        """
        Smart movement with obstacle detection and recovery.
        
        Args:
            direction: 'forward', 'backward', 'left', 'right'
            duration: How long to move
        """
        try:
            print(f"[MOVEMENT] Smart {direction} movement for {duration}s")
            
            if direction == 'forward':
                await self.move_forward(duration)
            elif direction == 'backward':
                await self.move_backward(duration)
            elif direction == 'left':
                await self.move_left(duration)
            elif direction == 'right':
                await self.move_right(duration)
            
            # Brief pause to let game respond
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"[MOVEMENT] Error during {direction} movement: {e}")
            # Try a small recovery movement
            await asyncio.sleep(0.5)

    async def recenter_camera(self):
        """
        Recenter camera to look straight ahead (horizontal).
        This prevents getting stuck looking at ground or sky.
        """
        print("[CAMERA] Recentering to horizontal view...")
        # Small upward adjustment to ensure we're looking forward
        await self.look_vertical(5)
        await asyncio.sleep(0.1)

    async def scan_for_targets(self):
        """
        Use right stick to scan environment for targets/objects.
        Performs a smooth camera sweep to detect moving objects or points of interest.
        Always recenters camera after scanning.
        """
        import random
        
        # Random scanning pattern - favor horizontal scans to avoid ground-staring
        scan_type = random.choice(['horizontal_sweep', 'horizontal_sweep', 'quick_glance'])
        
        if scan_type == 'horizontal_sweep':
            # Sweep camera left to right to scan horizon
            print("[SCAN] Horizontal sweep for targets...")
            await self.look_horizontal(-30)  # Look left
            await asyncio.sleep(0.2)
            await self.look_horizontal(60)   # Sweep right
            await asyncio.sleep(0.2)
            await self.look_horizontal(-30)  # Return to center
            
        else:  # quick_glance
            # Quick glance to one side
            direction = random.choice([-45, 45])
            print(f"[SCAN] Quick glance {('left' if direction < 0 else 'right')}...")
            await self.look_horizontal(direction)
            await asyncio.sleep(0.1)
            await self.look_horizontal(-direction)  # Return to center
        
        # Always recenter camera after scanning
        await self.recenter_camera()

    async def track_moving_target(self, horizontal_offset: float = 0, vertical_offset: float = 0):
        """
        Track a moving target by adjusting camera with right stick.
        
        Args:
            horizontal_offset: Degrees to adjust horizontally (+ = right, - = left)
            vertical_offset: Degrees to adjust vertically (+ = up, - = down)
        """
        print(f"[TRACK] Tracking target: H={horizontal_offset:.1f}°, V={vertical_offset:.1f}°")
        
        # Smooth tracking movements
        if abs(horizontal_offset) > 5:
            await self.look_horizontal(horizontal_offset * 0.5)  # Smooth tracking
            await asyncio.sleep(0.05)
        
        if abs(vertical_offset) > 5:
            await self.look_vertical(vertical_offset * 0.5)
            await asyncio.sleep(0.05)

    async def look_horizontal(self, degrees: float):
        """
        Look left/right by specified degrees.
        
        Args:
            degrees: Positive = right, negative = left
        """
        if self.controller is not None:
            # Map degrees to right stick X axis: assume max degrees = 45 maps to stick 1.0
            magnitude = max(-1.0, min(1.0, degrees / 45.0))
            await self.controller.look(magnitude, 0, duration=abs(degrees) / 45.0)
            await self.controller.look(0, 0, duration=0.05)
            return
        if not PYAUTOGUI_AVAILABLE:
            print(f"[DUMMY] Looking {degrees} degrees")
            return
        # Convert degrees to pixel movement (approximate)
        # Skyrim sensitivity varies, but ~3 pixels per degree is common
        pixels = int(degrees * 3)
        # Move in small steps for smooth camera movement
        steps = max(10, abs(pixels) // 20)
        step_size = pixels / steps
        for _ in range(steps):
            pyautogui.moveRel(step_size, 0, duration=0.02)
            await asyncio.sleep(0.01)
    
    async def look_vertical(self, degrees: float):
        """
        Look up/down by specified degrees.
        
        Args:
            degrees: Positive = up, negative = down
        """
        if self.controller is not None:
            # Map degrees to right stick Y axis: assume max degrees = 30 maps to stick 1.0
            magnitude = max(-1.0, min(1.0, degrees / 30.0))
            await self.controller.look(0, magnitude, duration=abs(degrees) / 30.0)
            await self.controller.look(0, 0, duration=0.05)
            return
        if not PYAUTOGUI_AVAILABLE:
            print(f"[DUMMY] Looking {degrees} degrees vertically")
            return
        # Convert degrees to pixel movement
        pixels = int(degrees * 3)
        # Move in small steps
        steps = max(10, abs(pixels) // 20)
        step_size = pixels / steps
        for _ in range(steps):
            pyautogui.moveRel(0, -step_size, duration=0.02)  # Negative because screen Y is inverted
            await asyncio.sleep(0.01)

    async def look_around(self):
        """Look around (move camera randomly)."""
        import random
        if self.controller is not None:
            # Random right stick movement
            h = random.uniform(-0.7, 0.7)
            v = random.uniform(-0.4, 0.4)
            await self.controller.look(h, v, duration=0.5)
            await self.controller.look(0, 0, duration=0.05)
            return
        if not PYAUTOGUI_AVAILABLE:
            print("[DUMMY] Looking around")
            return
        # Random horizontal look
        h_degrees = random.uniform(-45, 45)
        await self.look_horizontal(h_degrees)
        # Random vertical look (smaller range to avoid looking at sky/ground too much)
        v_degrees = random.uniform(-20, 20)
        await self.look_vertical(v_degrees)

    # Action sequences

    async def execute_sequence(self, actions: List[Action], delay: float = 0.1):
        """
        Execute sequence of actions.

        Args:
            actions: List of actions to execute
            delay: Delay between actions
        """
        for action in actions:
            await self.execute(action)
            if delay > 0:
                await asyncio.sleep(delay)

    async def combat_sequence(self, enemy: str):
        """Execute combat sequence."""
        sequence = [
            Action(ActionType.ATTACK),
            Action(ActionType.MOVE_BACKWARD, 0.5),
            Action(ActionType.BLOCK, 1.0),
            Action(ActionType.ATTACK),
        ]
        await self.execute_sequence(sequence, delay=0.2)

    async def stealth_approach(self, target: str):
        """Approach target stealthily."""
        await self.execute(Action(ActionType.SNEAK))
        await self.move_forward(3.0)
        await self.execute(Action(ActionType.SNEAK))  # Toggle off

    async def quick_save_checkpoint(self):
        """Quick save the game."""
        await self.execute(Action(ActionType.QUICK_SAVE))
        print("OK Game saved")

    def get_action_history(self, n: int = 10) -> List[Action]:
        """Get recent action history."""
        return self.action_history[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get action execution statistics."""
        return {
            **self.stats,
            'history_size': len(self.action_history),
            'using_game_api': self.use_game_api,
            'dry_run': self.dry_run,
            'pyautogui_available': PYAUTOGUI_AVAILABLE,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'actions_executed': 0,
            'total_duration': 0.0,
            'errors': 0,
        }


# Example usage
if __name__ == "__main__":
    print("Testing Skyrim Actions...")

    async def test():
        # Create action controller
        actions = SkyrimActions(dry_run=True)

        # Test basic actions
        print("\n1. Testing basic actions...")
        await actions.move_forward(2.0)
        await actions.execute(Action(ActionType.JUMP))
        await actions.execute(Action(ActionType.ATTACK))

        # Test high-level actions
        print("\n2. Testing high-level actions...")
        await actions.explore_area(duration=5.0)

        # Test sequences
        print("\n3. Testing action sequence...")
        await actions.combat_sequence("Bandit")

        # Stats
        print(f"\n4. Stats: {actions.get_stats()}")

        # History
        print(f"\n5. Recent actions:")
        for action in actions.get_action_history(5):
            print(f"   - {action.action_type.value} ({action.duration:.1f}s)")

    asyncio.run(test())

    print("\nOK Action tests complete")
