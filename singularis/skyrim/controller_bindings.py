"""
Skyrim Controller Bindings

Provides Steam Input-style action layers and bindings for Skyrim.
Maps high-level game actions to Xbox 360 controller inputs.

Action Layers:
- Exploration: Movement, camera, basic interaction
- Combat: Attacks, blocks, shouts
- Menu: Inventory, map, skills navigation
- Dialogue: NPC interaction, choice selection
- Stealth: Sneaking, lockpicking

This allows the AGI to switch between different control schemes
depending on the game context.
"""

from typing import Dict, Optional
from .controller import VirtualXboxController, XboxButton, ActionLayer
import asyncio


class SkyrimControllerBindings:
    """Manages Skyrim-specific controller bindings using a Steam Input-style action layer system.

    This class is responsible for creating and configuring the different action
    layers (e.g., "Exploration", "Combat", "Menu") and binding high-level game
    actions to specific controller inputs within each layer. This allows the agent
    to dynamically switch its control scheme to suit the current gameplay context.

    Attributes:
        controller: An instance of the VirtualXboxController to which the bindings
                    will be applied.
    """
    
    def __init__(self, controller: VirtualXboxController):
        """Initializes the Skyrim controller bindings.

        Args:
            controller: An instance of the VirtualXboxController.
        """
        self.controller = controller
        self._setup_layers()
    
    def _setup_layers(self):
        """Sets up all the action layers and their respective bindings."""
        self._setup_exploration_layer()
        self._setup_combat_layer()
        self._setup_menu_layer()
        self._setup_dialogue_layer()
        self._setup_stealth_layer()
    
    def _setup_exploration_layer(self):
        """Configures the bindings for the 'Exploration' action layer.

        This layer is intended for general gameplay, such as traversing the world,
        interacting with objects, and basic combat.
        """
        layer = self.controller.create_action_layer("Exploration", priority=0)
        
        # Movement actions
        async def move_forward(ctrl, duration=1.0):
            await ctrl.move(0, 1.0, duration=duration)
        
        async def move_backward(ctrl, duration=1.0):
            await ctrl.move(0, -1.0, duration=duration)
        
        async def move_left(ctrl, duration=1.0):
            await ctrl.move(-1.0, 0, duration=duration)
        
        async def move_right(ctrl, duration=1.0):
            await ctrl.move(1.0, 0, duration=duration)
        
        async def sprint(ctrl):
            await ctrl.tap_button(XboxButton.LB)  # FIXED: LB (Left Bumper) for sprint, not LS
        
        # Camera actions
        async def look_up(ctrl, duration=1.0):
            await ctrl.look(0, 1.0, duration=duration)
        
        async def look_down(ctrl, duration=1.0):
            await ctrl.look(0, -1.0, duration=duration)
        
        async def look_left(ctrl, duration=1.0):
            await ctrl.look(-1.0, 0, duration=duration)
        
        async def look_right(ctrl, duration=1.0):
            await ctrl.look(1.0, 0, duration=duration)
        
        # Basic actions
        async def jump(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.Y)  # Y for jump in Skyrim
        
        async def activate(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.A)
        
        async def sneak_toggle(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.LS)  # Left stick click for sneak
        
        async def sheath_weapon(ctrl, duration=0.0):
            ctrl.set_left_trigger(1.0)
            await asyncio.sleep(0.5)
            ctrl.set_left_trigger(0.0)
        
        async def attack(ctrl, duration=0.3):
            ctrl.set_right_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 0.3)
            ctrl.set_right_trigger(0.0)
        
        async def open_menu(ctrl, duration=0.0):
            print("[CONTROLLER] Opening menu with START button...")
            await ctrl.tap_button(XboxButton.START)
            await asyncio.sleep(0.5)  # Wait for menu to open
            print("[CONTROLLER] Menu should be open now")
        
        async def wait(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.BACK)
        
        async def block(ctrl, duration=1.0):
            ctrl.set_left_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 1.0)
            ctrl.set_left_trigger(0.0)
        
        async def power_attack(ctrl, duration=0.8):
            # Hold RT for power attack
            ctrl.set_right_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 0.8)
            ctrl.set_right_trigger(0.0)
        
        async def shout(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.RB)
        
        # Favorites/quick access
        async def favorite_up(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.DPAD_UP)
        
        async def favorite_down(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.DPAD_DOWN)
        
        async def favorite_left(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.DPAD_LEFT)
        
        async def favorite_right(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.DPAD_RIGHT)
        
        async def heal(ctrl, duration=1.0):
            # Use healing spell - press and hold RB (or use favorited healing)
            # In Skyrim, healing spells are typically cast with triggers
            # If healing spell is equipped in left hand, hold LT
            # If in right hand, hold RT
            # Simplified: use DPad to select healing from favorites, then cast
            await ctrl.tap_button(XboxButton.DPAD_UP)  # Select healing from favorites
            await asyncio.sleep(0.2)
            ctrl.set_right_trigger(0.5)  # Cast healing spell (partial hold)
            await asyncio.sleep(duration if duration > 0 else 1.0)
            ctrl.set_right_trigger(0.0)
        
        async def toggle_pov(ctrl, duration=0.0):
            # Toggle between 1st and 3rd person view
            await ctrl.tap_button(XboxButton.RS)  # Right stick click
        
        async def move_object(ctrl, duration=2.0):
            # Hold A to grab and move objects
            ctrl.press_button(XboxButton.A)
            await asyncio.sleep(duration if duration > 0 else 2.0)
            ctrl.release_button(XboxButton.A)
        
        # Bind actions
        self.controller.bind_action("Exploration", "move_forward", move_forward)
        self.controller.bind_action("Exploration", "move_backward", move_backward)
        self.controller.bind_action("Exploration", "move_left", move_left)
        self.controller.bind_action("Exploration", "move_right", move_right)
        self.controller.bind_action("Exploration", "sprint", sprint)
        
        self.controller.bind_action("Exploration", "look_up", look_up)
        self.controller.bind_action("Exploration", "look_down", look_down)
        self.controller.bind_action("Exploration", "look_left", look_left)
        self.controller.bind_action("Exploration", "look_right", look_right)
        
        self.controller.bind_action("Exploration", "jump", jump)
        self.controller.bind_action("Exploration", "activate", activate)
        self.controller.bind_action("Exploration", "sneak", sneak_toggle)
        self.controller.bind_action("Exploration", "sheath", sheath_weapon)
        self.controller.bind_action("Exploration", "attack", attack)
        self.controller.bind_action("Exploration", "block", block)
        self.controller.bind_action("Exploration", "power_attack", power_attack)
        self.controller.bind_action("Exploration", "shout", shout)
        self.controller.bind_action("Exploration", "menu", open_menu)
        self.controller.bind_action("Exploration", "wait", wait)
        
        self.controller.bind_action("Exploration", "favorite_up", favorite_up)
        self.controller.bind_action("Exploration", "favorite_down", favorite_down)
        self.controller.bind_action("Exploration", "favorite_left", favorite_left)
        self.controller.bind_action("Exploration", "favorite_right", favorite_right)
        self.controller.bind_action("Exploration", "heal", heal)
        self.controller.bind_action("Exploration", "toggle_pov", toggle_pov)
        self.controller.bind_action("Exploration", "move_object", move_object)
    
    def _setup_combat_layer(self):
        """Configures the bindings for the 'Combat' action layer.

        This layer provides a control scheme optimized for combat, with quick
        access to attacks, blocks, and special moves.
        """
        layer = self.controller.create_action_layer("Combat", priority=10)
        
        async def quick_attack(ctrl, duration=0.3):
            ctrl.set_right_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 0.3)
            ctrl.set_right_trigger(0.0)
        
        async def power_attack(ctrl, duration=0.8):
            await ctrl.tap_button(XboxButton.RB)
            ctrl.set_right_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 0.8)
            ctrl.set_right_trigger(0.0)
        
        async def block(ctrl, duration=1.0):
            ctrl.set_left_trigger(1.0)
            await asyncio.sleep(duration if duration > 0 else 1.0)
            ctrl.set_left_trigger(0.0)
        
        async def bash(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.LB)
        
        async def shout(ctrl, duration=0.0):
            await ctrl.tap_button(XboxButton.Y)
        
        async def dodge_roll(ctrl, duration=0.0):
            # Jump + direction
            await ctrl.tap_button(XboxButton.A)
        
        async def retreat(ctrl, duration=1.5):
            # Sprint backward
            await ctrl.tap_button(XboxButton.LB)  # FIXED: LB for sprint
            await ctrl.move(0, -1.0, duration=duration if duration > 0 else 1.5)
        
        async def quick_heal(ctrl, duration=0.0):
            # Use favorite (assuming healing potion on DPad)
            await ctrl.tap_button(XboxButton.DPAD_DOWN)
        
        # Movement during combat (reuse from exploration)
        async def move_forward(ctrl, duration=1.0):
            await ctrl.move(0, 1.0, duration=duration)
        
        async def move_backward(ctrl, duration=1.0):
            await ctrl.move(0, -1.0, duration=duration)
        
        async def move_left(ctrl, duration=1.0):
            await ctrl.move(-1.0, 0, duration=duration)
        
        async def move_right(ctrl, duration=1.0):
            await ctrl.move(1.0, 0, duration=duration)
        
        # Bind combat actions
        self.controller.bind_action("Combat", "attack", quick_attack)
        self.controller.bind_action("Combat", "power_attack", power_attack)
        self.controller.bind_action("Combat", "block", block)
        self.controller.bind_action("Combat", "bash", bash)
        self.controller.bind_action("Combat", "shout", shout)
        self.controller.bind_action("Combat", "dodge", dodge_roll)
        self.controller.bind_action("Combat", "retreat", retreat)
        self.controller.bind_action("Combat", "heal", quick_heal)
        
        # Bind movement actions for combat
        self.controller.bind_action("Combat", "move_forward", move_forward)
        self.controller.bind_action("Combat", "move_backward", move_backward)
        self.controller.bind_action("Combat", "move_left", move_left)
        self.controller.bind_action("Combat", "move_right", move_right)
    
    def _setup_menu_layer(self):
        """Configures the bindings for the 'Menu' action layer.

        This layer is designed for navigating through game menus, such as the
        inventory, map, and skills screens.
        """
        layer = self.controller.create_action_layer("Menu", priority=20)
        
        async def navigate_up(ctrl):
            ctrl.set_left_stick(0, 1.0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
        
        async def navigate_down(ctrl):
            ctrl.set_left_stick(0, -1.0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
        
        async def navigate_left(ctrl):
            ctrl.set_left_stick(-1.0, 0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
        
        async def navigate_right(ctrl):
            ctrl.set_left_stick(1.0, 0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
        
        async def select(ctrl):
            await ctrl.tap_button(XboxButton.A)
        
        async def back(ctrl):
            await ctrl.tap_button(XboxButton.B)
        
        async def drop_take(ctrl):
            await ctrl.tap_button(XboxButton.X)
        
        async def favorite(ctrl):
            await ctrl.tap_button(XboxButton.Y)
        
        async def tab_left(ctrl):
            await ctrl.tap_button(XboxButton.LB)
        
        async def tab_right(ctrl):
            await ctrl.tap_button(XboxButton.RB)
        
        async def close_menu(ctrl):
            await ctrl.tap_button(XboxButton.START)
        
        # Bind menu actions
        self.controller.bind_action("Menu", "up", navigate_up)
        self.controller.bind_action("Menu", "down", navigate_down)
        self.controller.bind_action("Menu", "left", navigate_left)
        self.controller.bind_action("Menu", "right", navigate_right)
        self.controller.bind_action("Menu", "select", select)
        self.controller.bind_action("Menu", "back", back)
        self.controller.bind_action("Menu", "drop", drop_take)
        self.controller.bind_action("Menu", "favorite", favorite)
        self.controller.bind_action("Menu", "tab_left", tab_left)
        self.controller.bind_action("Menu", "tab_right", tab_right)
        self.controller.bind_action("Menu", "close", close_menu)
    
    def _setup_dialogue_layer(self):
        """Configures the bindings for the 'Dialogue' action layer.

        This layer is used for navigating conversations with NPCs.
        """
        layer = self.controller.create_action_layer("Dialogue", priority=15)
        
        async def select_option_1(ctrl):
            ctrl.set_left_stick(0, 1.0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
            await ctrl.tap_button(XboxButton.A)
        
        async def select_option_2(ctrl):
            ctrl.set_left_stick(0, 0.5)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
            await ctrl.tap_button(XboxButton.A)
        
        async def select_option_3(ctrl):
            ctrl.set_left_stick(0, -0.5)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
            await ctrl.tap_button(XboxButton.A)
        
        async def select_option_4(ctrl):
            ctrl.set_left_stick(0, -1.0)
            await asyncio.sleep(0.1)
            ctrl.set_left_stick(0, 0)
            await ctrl.tap_button(XboxButton.A)
        
        async def exit_dialogue(ctrl):
            await ctrl.tap_button(XboxButton.B)
        
        # Bind dialogue actions
        self.controller.bind_action("Dialogue", "option_1", select_option_1)
        self.controller.bind_action("Dialogue", "option_2", select_option_2)
        self.controller.bind_action("Dialogue", "option_3", select_option_3)
        self.controller.bind_action("Dialogue", "option_4", select_option_4)
        self.controller.bind_action("Dialogue", "exit", exit_dialogue)
    
    def _setup_stealth_layer(self):
        """Configures the bindings for the 'Stealth' action layer.

        This layer provides a control scheme optimized for stealth-based gameplay,
        such as sneaking, pickpocketing, and performing backstabs.
        """
        layer = self.controller.create_action_layer("Stealth", priority=5)
        
        async def sneak_move_slow(ctrl):
            await ctrl.move(0, 0.3, duration=1.0)  # Slow movement
        
        async def backstab(ctrl):
            ctrl.set_right_trigger(1.0)
            await asyncio.sleep(0.3)
            ctrl.set_right_trigger(0.0)
        
        async def pickpocket(ctrl):
            await ctrl.tap_button(XboxButton.X)
        
        # Bind stealth actions
        self.controller.bind_action("Stealth", "sneak_move", sneak_move_slow)
        self.controller.bind_action("Stealth", "backstab", backstab)
        self.controller.bind_action("Stealth", "pickpocket", pickpocket)
    
    # === Context switching ===
    
    def switch_to_exploration(self):
        """Switches the controller to the 'Exploration' action layer."""
        self._deactivate_all()
        self.controller.activate_layer("Exploration")
    
    def switch_to_combat(self):
        """Switches the controller to the 'Combat' action layer."""
        self._deactivate_all()
        self.controller.activate_layer("Combat")
    
    def switch_to_menu(self):
        """Switches the controller to the 'Menu' action layer."""
        self._deactivate_all()
        self.controller.activate_layer("Menu")
    
    def switch_to_dialogue(self):
        """Switches the controller to the 'Dialogue' action layer."""
        self._deactivate_all()
        self.controller.activate_layer("Dialogue")
    
    def switch_to_stealth(self):
        """Switches the controller to the 'Stealth' action layer."""
        self._deactivate_all()
        self.controller.activate_layer("Stealth")
    
    def _deactivate_all(self):
        """Deactivates all action layers."""
        for layer_name in self.controller.action_layers:
            self.controller.deactivate_layer(layer_name)
    
    # === High-level action sequences ===
    
    async def combat_combo_light_heavy(self):
        """Executes a pre-defined light-heavy attack combo."""
        await self.controller.execute_action("attack")
        await asyncio.sleep(0.3)
        await self.controller.execute_action("attack")
        await asyncio.sleep(0.3)
        await self.controller.execute_action("power_attack")
    
    async def defensive_maneuver(self):
        """Executes a pre-defined defensive maneuver (block followed by a bash)."""
        await self.controller.execute_action("block")
        await asyncio.sleep(0.2)
        await self.controller.execute_action("bash")
    
    async def exploration_scan(self, duration: float = 3.0):
        """Performs a slow, 360-degree scan of the surroundings.

        Args:
            duration: The duration of the scan.
        """
        # Smooth 360-degree camera rotation
        steps = 8
        step_duration = duration / steps
        
        for i in range(steps):
            angle = (i / steps) * 2 * 3.14159  # Full circle
            x = 0.5 * (1 if i < steps/2 else -1)  # Look right then left
            
            self.controller.set_right_stick(x, 0)
            await asyncio.sleep(step_duration)
        
        self.controller.set_right_stick(0, 0)
