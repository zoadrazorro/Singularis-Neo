"""
Virtual Xbox 360 Controller - Steam Input Style

Provides a virtual game controller interface modeling Steam Input for Xbox 360.
This allows the AGI to control games using native controller input instead of
keyboard/mouse simulation, which is more natural for many games including Skyrim.

Features:
- Full Xbox 360 controller emulation (buttons, triggers, sticks)
- Steam Input-style action layers and action sets
- Smooth analog input with deadzone handling
- Button combo support
- Input recording and playback
- Configurable sensitivity and response curves

Design principles:
- Controller input provides more fluid, analog control than keyboard
- Natural for Skyrim's movement and camera systems
- Enables smooth, game-appropriate actions
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    import vgamepad as vg
    VGAMEPAD_AVAILABLE = True
except ImportError:
    VGAMEPAD_AVAILABLE = False
    print("Warning: vgamepad not installed. Controller will use dummy mode.")
    print("Install with: pip install vgamepad")


class XboxButton(Enum):
    """Xbox 360 controller buttons."""
    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    
    LB = "LB"  # Left bumper
    RB = "RB"  # Right bumper
    
    BACK = "BACK"
    START = "START"
    GUIDE = "GUIDE"
    
    LS = "LS"  # Left stick click
    RS = "RS"  # Right stick click
    
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"


class XboxAxis(Enum):
    """Xbox 360 controller axes."""
    LEFT_STICK_X = "LEFT_STICK_X"
    LEFT_STICK_Y = "LEFT_STICK_Y"
    RIGHT_STICK_X = "RIGHT_STICK_X"
    RIGHT_STICK_Y = "RIGHT_STICK_Y"
    LEFT_TRIGGER = "LEFT_TRIGGER"
    RIGHT_TRIGGER = "RIGHT_TRIGGER"


@dataclass
class StickState:
    """State of an analog stick."""
    x: float = 0.0  # -1.0 to 1.0
    y: float = 0.0  # -1.0 to 1.0
    deadzone: float = 0.15
    
    def get_magnitude(self) -> float:
        """Get magnitude of stick deflection."""
        return math.sqrt(self.x**2 + self.y**2)
    
    def get_angle(self) -> float:
        """Get angle in radians."""
        return math.atan2(self.y, self.x)
    
    def apply_deadzone(self) -> Tuple[float, float]:
        """Apply circular deadzone."""
        magnitude = self.get_magnitude()
        if magnitude < self.deadzone:
            return 0.0, 0.0
        
        # Rescale to 0-1 range after deadzone
        scale = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        angle = self.get_angle()
        
        return scale * math.cos(angle), scale * math.sin(angle)


@dataclass
class TriggerState:
    """State of a trigger."""
    value: float = 0.0  # 0.0 to 1.0
    deadzone: float = 0.05
    
    def apply_deadzone(self) -> float:
        """Apply trigger deadzone."""
        if self.value < self.deadzone:
            return 0.0
        return (self.value - self.deadzone) / (1.0 - self.deadzone)


@dataclass
class ActionLayer:
    """
    Steam Input-style action layer.
    
    An action layer is a set of bindings that can be activated/deactivated.
    For example: "Combat" layer, "Menu Navigation" layer, "Exploration" layer.
    """
    name: str
    bindings: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    priority: int = 0  # Higher priority layers override lower ones


class VirtualXboxController:
    """
    Virtual Xbox 360 controller with Steam Input-style features.
    
    This class provides:
    1. Low-level controller emulation (buttons, sticks, triggers)
    2. High-level action layers (like Steam Input)
    3. Smooth analog input with proper deadzones
    4. Input recording and playback
    """
    
    def __init__(
        self,
        deadzone_stick: float = 0.15,
        deadzone_trigger: float = 0.05,
        sensitivity: float = 1.0,
        dry_run: bool = False
    ):
        """
        Initialize virtual controller.
        
        Args:
            deadzone_stick: Deadzone for analog sticks (0.0-1.0)
            deadzone_trigger: Deadzone for triggers (0.0-1.0)
            sensitivity: Global sensitivity multiplier
            dry_run: Don't actually send input (for testing)
        """
        self.dry_run = dry_run
        self.sensitivity = sensitivity
        
        # Initialize vgamepad controller
        self.gamepad = None
        if VGAMEPAD_AVAILABLE and not dry_run:
            try:
                self.gamepad = vg.VX360Gamepad()
                print("OK Virtual Xbox 360 controller initialized")
            except Exception as e:
                print(f"Warning: Could not initialize vgamepad: {e}")
                self.gamepad = None
        
        # Controller state
        self.left_stick = StickState(deadzone=deadzone_stick)
        self.right_stick = StickState(deadzone=deadzone_stick)
        self.left_trigger = TriggerState(deadzone=deadzone_trigger)
        self.right_trigger = TriggerState(deadzone=deadzone_trigger)
        self.buttons_pressed: set = set()
        
        # Action layers (Steam Input style)
        self.action_layers: Dict[str, ActionLayer] = {}
        self.active_layer: Optional[str] = None
        
        # Input recording
        self.recording: List[Dict[str, Any]] = []
        self.is_recording = False
        self.record_start_time = 0.0
        
        # Statistics
        self.stats = {
            'inputs_sent': 0,
            'buttons_pressed': 0,
            'stick_movements': 0,
            'trigger_pulls': 0,
            'total_duration': 0.0,
        }
    
    # === Low-level input methods ===
    
    def press_button(self, button: XboxButton):
        """Press a button."""
        if self.dry_run:
            print(f"[DRY RUN] Press {button.value}")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Press {button.value}")
            return
        
        button_map = {
            XboxButton.A: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            XboxButton.B: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            XboxButton.X: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            XboxButton.Y: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            XboxButton.LB: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
            XboxButton.RB: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            XboxButton.BACK: vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
            XboxButton.START: vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            XboxButton.GUIDE: vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
            XboxButton.LS: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
            XboxButton.RS: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
            XboxButton.DPAD_UP: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            XboxButton.DPAD_DOWN: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            XboxButton.DPAD_LEFT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            XboxButton.DPAD_RIGHT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        }
        
        vg_button = button_map.get(button)
        if vg_button:
            self.gamepad.press_button(button=vg_button)
            self.gamepad.update()
            self.buttons_pressed.add(button)
            self.stats['buttons_pressed'] += 1
            self._record_input('button_press', {'button': button.value})
    
    def release_button(self, button: XboxButton):
        """Release a button."""
        if self.dry_run:
            print(f"[DRY RUN] Release {button.value}")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Release {button.value}")
            return
        
        button_map = {
            XboxButton.A: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            XboxButton.B: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            XboxButton.X: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            XboxButton.Y: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            XboxButton.LB: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
            XboxButton.RB: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            XboxButton.BACK: vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
            XboxButton.START: vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            XboxButton.GUIDE: vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
            XboxButton.LS: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
            XboxButton.RS: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
            XboxButton.DPAD_UP: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            XboxButton.DPAD_DOWN: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            XboxButton.DPAD_LEFT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            XboxButton.DPAD_RIGHT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        }
        
        vg_button = button_map.get(button)
        if vg_button:
            self.gamepad.release_button(button=vg_button)
            self.gamepad.update()
            self.buttons_pressed.discard(button)
            self._record_input('button_release', {'button': button.value})
    
    async def tap_button(self, button: XboxButton, duration: float = 0.1):
        """Tap a button (press and release)."""
        self.press_button(button)
        await asyncio.sleep(duration)
        self.release_button(button)
    
    def set_left_stick(self, x: float, y: float):
        """
        Set left stick position.
        
        Args:
            x: Horizontal position (-1.0 to 1.0, left to right)
            y: Vertical position (-1.0 to 1.0, down to up)
        """
        self.left_stick.x = max(-1.0, min(1.0, x * self.sensitivity))
        self.left_stick.y = max(-1.0, min(1.0, y * self.sensitivity))
        
        if self.dry_run:
            print(f"[DRY RUN] Left stick: ({x:.2f}, {y:.2f})")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Left stick: ({x:.2f}, {y:.2f})")
            return
        
        # Apply deadzone
        dx, dy = self.left_stick.apply_deadzone()
        
        # Convert to vgamepad range (-32768 to 32767)
        vg_x = int(dx * 32767)
        vg_y = int(dy * 32767)
        
        self.gamepad.left_joystick(x_value=vg_x, y_value=vg_y)
        self.gamepad.update()
        self.stats['stick_movements'] += 1
        self._record_input('left_stick', {'x': x, 'y': y})
    
    def set_right_stick(self, x: float, y: float):
        """
        Set right stick position.
        
        Args:
            x: Horizontal position (-1.0 to 1.0, left to right)
            y: Vertical position (-1.0 to 1.0, down to up)
        """
        self.right_stick.x = max(-1.0, min(1.0, x * self.sensitivity))
        self.right_stick.y = max(-1.0, min(1.0, y * self.sensitivity))
        
        if self.dry_run:
            print(f"[DRY RUN] Right stick: ({x:.2f}, {y:.2f})")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Right stick: ({x:.2f}, {y:.2f})")
            return
        
        # Apply deadzone
        dx, dy = self.right_stick.apply_deadzone()
        
        # Convert to vgamepad range
        vg_x = int(dx * 32767)
        vg_y = int(dy * 32767)
        
        self.gamepad.right_joystick(x_value=vg_x, y_value=vg_y)
        self.gamepad.update()
        self.stats['stick_movements'] += 1
        self._record_input('right_stick', {'x': x, 'y': y})
    
    def set_left_trigger(self, value: float):
        """
        Set left trigger value.
        
        Args:
            value: Trigger value (0.0 to 1.0)
        """
        self.left_trigger.value = max(0.0, min(1.0, value))
        
        if self.dry_run:
            print(f"[DRY RUN] Left trigger: {value:.2f}")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Left trigger: {value:.2f}")
            return
        
        # Apply deadzone
        dv = self.left_trigger.apply_deadzone()
        
        # Convert to vgamepad range (0 to 255)
        vg_value = int(dv * 255)
        
        self.gamepad.left_trigger(value=vg_value)
        self.gamepad.update()
        self.stats['trigger_pulls'] += 1
        self._record_input('left_trigger', {'value': value})
    
    def set_right_trigger(self, value: float):
        """
        Set right trigger value.
        
        Args:
            value: Trigger value (0.0 to 1.0)
        """
        self.right_trigger.value = max(0.0, min(1.0, value))
        
        if self.dry_run:
            print(f"[DRY RUN] Right trigger: {value:.2f}")
            return
        
        if not self.gamepad:
            print(f"[DUMMY] Right trigger: {value:.2f}")
            return
        
        # Apply deadzone
        dv = self.right_trigger.apply_deadzone()
        
        # Convert to vgamepad range
        vg_value = int(dv * 255)
        
        self.gamepad.right_trigger(value=vg_value)
        self.gamepad.update()
        self.stats['trigger_pulls'] += 1
        self._record_input('right_trigger', {'value': value})
    
    def reset(self):
        """Reset all inputs to neutral."""
        # Release all buttons
        for button in list(self.buttons_pressed):
            self.release_button(button)
        
        # Reset sticks and triggers
        self.set_left_stick(0, 0)
        self.set_right_stick(0, 0)
        self.set_left_trigger(0)
        self.set_right_trigger(0)
    
    # === High-level convenience methods ===
    
    async def move(self, x: float, y: float, duration: float = 1.0):
        """
        Move using left stick for specified duration.
        
        Args:
            x: Horizontal movement (-1.0 to 1.0)
            y: Vertical movement (-1.0 to 1.0)
            duration: How long to hold the stick
        """
        self.set_left_stick(x, y)
        await asyncio.sleep(duration)
        self.set_left_stick(0, 0)
    
    async def look(self, x: float, y: float, duration: float = 0.5):
        """
        Look using right stick for specified duration.
        
        Args:
            x: Horizontal look (-1.0 to 1.0)
            y: Vertical look (-1.0 to 1.0)
            duration: How long to hold the stick
        """
        self.set_right_stick(x, y)
        await asyncio.sleep(duration)
        self.set_right_stick(0, 0)
    
    async def smooth_stick_movement(
        self,
        stick: str,
        target_x: float,
        target_y: float,
        duration: float = 0.5,
        steps: int = 20
    ):
        """
        Smoothly move a stick to target position.
        
        Args:
            stick: 'left' or 'right'
            target_x: Target X position
            target_y: Target Y position
            duration: Time to reach target
            steps: Number of interpolation steps
        """
        current_x = self.left_stick.x if stick == 'left' else self.right_stick.x
        current_y = self.left_stick.y if stick == 'left' else self.right_stick.y
        
        step_duration = duration / steps
        
        for i in range(steps + 1):
            t = i / steps
            # Ease-in-out interpolation
            t = t * t * (3 - 2 * t)
            
            x = current_x + (target_x - current_x) * t
            y = current_y + (target_y - current_y) * t
            
            if stick == 'left':
                self.set_left_stick(x, y)
            else:
                self.set_right_stick(x, y)
            
            await asyncio.sleep(step_duration)
    
    # === Steam Input-style Action Layers ===
    
    def create_action_layer(self, name: str, priority: int = 0) -> ActionLayer:
        """
        Create a new action layer.
        
        Args:
            name: Layer name (e.g., "Combat", "Menu", "Exploration")
            priority: Layer priority (higher overrides lower)
        
        Returns:
            Created action layer
        """
        layer = ActionLayer(name=name, priority=priority)
        self.action_layers[name] = layer
        return layer
    
    def activate_layer(self, name: str):
        """Activate an action layer."""
        if name in self.action_layers:
            self.action_layers[name].active = True
            self.active_layer = name
            print(f"* Activated layer: {name}")
    
    def deactivate_layer(self, name: str):
        """Deactivate an action layer."""
        if name in self.action_layers:
            self.action_layers[name].active = False
            if self.active_layer == name:
                self.active_layer = None
    
    def bind_action(self, layer_name: str, action_name: str, binding: Any):
        """
        Bind an action to a layer.
        
        Args:
            layer_name: Name of the layer
            action_name: Name of the action (e.g., "jump", "attack")
            binding: Button or combination to bind
        """
        if layer_name in self.action_layers:
            self.action_layers[layer_name].bindings[action_name] = binding
    
    async def execute_action(self, action_name: str) -> bool:
        """
        Execute a named action from the active layer.
        
        Args:
            action_name: Name of the action to execute
        
        Returns:
            Success status
        """
        # Find action in active layers (by priority)
        sorted_layers = sorted(
            [l for l in self.action_layers.values() if l.active],
            key=lambda l: l.priority,
            reverse=True
        )
        
        for layer in sorted_layers:
            if action_name in layer.bindings:
                binding = layer.bindings[action_name]
                
                # Execute binding
                if isinstance(binding, XboxButton):
                    await self.tap_button(binding)
                    return True
                elif callable(binding):
                    await binding(self)
                    return True
        
        return False
    
    # === Input Recording ===
    
    def start_recording(self):
        """Start recording inputs."""
        self.recording = []
        self.is_recording = True
        self.record_start_time = time.time()
        print("OK Started recording inputs")
    
    def stop_recording(self) -> List[Dict[str, Any]]:
        """
        Stop recording and return recorded inputs.
        
        Returns:
            List of recorded input events
        """
        self.is_recording = False
        print(f"OK Stopped recording ({len(self.recording)} events)")
        return self.recording
    
    def _record_input(self, input_type: str, data: Dict[str, Any]):
        """Record an input event."""
        if self.is_recording:
            timestamp = time.time() - self.record_start_time
            self.recording.append({
                'timestamp': timestamp,
                'type': input_type,
                'data': data
            })
    
    async def playback_recording(self, recording: Optional[List[Dict[str, Any]]] = None):
        """
        Play back recorded inputs.
        
        Args:
            recording: Recording to play back (uses last recording if None)
        """
        if recording is None:
            recording = self.recording
        
        if not recording:
            print("No recording to play back")
            return
        
        print(f"Playing back {len(recording)} events...")
        start_time = time.time()
        
        for event in recording:
            # Wait until event timestamp
            target_time = start_time + event['timestamp']
            wait_time = target_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Execute event
            event_type = event['type']
            data = event['data']
            
            if event_type == 'button_press':
                self.press_button(XboxButton(data['button']))
            elif event_type == 'button_release':
                self.release_button(XboxButton(data['button']))
            elif event_type == 'left_stick':
                self.set_left_stick(data['x'], data['y'])
            elif event_type == 'right_stick':
                self.set_right_stick(data['x'], data['y'])
            elif event_type == 'left_trigger':
                self.set_left_trigger(data['value'])
            elif event_type == 'right_trigger':
                self.set_right_trigger(data['value'])
        
        print("OK Playback complete")
    
    # === Statistics ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            **self.stats,
            'vgamepad_available': VGAMEPAD_AVAILABLE,
            'gamepad_connected': self.gamepad is not None,
            'dry_run': self.dry_run,
            'active_layer': self.active_layer,
            'num_layers': len(self.action_layers),
            'buttons_currently_pressed': len(self.buttons_pressed),
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.gamepad:
            self.reset()
