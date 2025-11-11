"""
Skyrim Action Control Layer

Handles all action execution in Skyrim:
1. Keyboard/mouse control (via pyautogui or pynput)
2. Game API actions (via SKSE mods)
3. Hierarchical action composition
4. Action sequences and macros

Philosophical grounding:
- ETHICA Part III: Actions express the mode's striving (conatus)
- Agency emerges from the ability to act autonomously
- Actions are chosen to maximize Δ𝒞 (coherence increase)
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

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
        dry_run: bool = False
    ):
        """
        Initialize action controller.

        Args:
            use_game_api: Use game API instead of keyboard/mouse
            custom_keys: Custom key bindings
            dry_run: Don't actually execute (for testing)
        """
        self.use_game_api = use_game_api
        self.dry_run = dry_run

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
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {action.action_type.value}")
            self.action_history.append(action)
            return True

        try:
            start_time = time.time()

            if self.use_game_api and self._game_api:
                success = await self._execute_via_api(action)
            else:
                success = await self._execute_via_input(action)

            # Update stats
            duration = time.time() - start_time
            self.stats['actions_executed'] += 1
            self.stats['total_duration'] += duration

            # Record in history
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

    async def look_around(self):
        """Look around (move camera)."""
        if not PYAUTOGUI_AVAILABLE:
            print("[DUMMY] Looking around")
            return

        # Move mouse to simulate looking
        import random
        dx = random.randint(-100, 100)
        dy = random.randint(-50, 50)
        pyautogui.moveRel(dx, dy, duration=0.5)

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
        print("✓ Game saved")

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

    print("\n✓ Action tests complete")
