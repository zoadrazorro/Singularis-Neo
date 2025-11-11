"""
Test Virtual Xbox Controller

Demonstrates the Steam Input-style virtual controller for Skyrim.
Tests various input methods and action layers.

Usage:
    python examples/test_controller.py [--dry-run]
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.skyrim.controller import VirtualXboxController, XboxButton
from singularis.skyrim.controller_bindings import SkyrimControllerBindings


async def test_basic_inputs(controller: VirtualXboxController):
    """Test basic controller inputs."""
    print("\n=== Testing Basic Inputs ===")
    
    # Test buttons
    print("Testing buttons...")
    await controller.tap_button(XboxButton.A)
    await asyncio.sleep(0.5)
    await controller.tap_button(XboxButton.B)
    await asyncio.sleep(0.5)
    
    # Test analog sticks
    print("Testing left stick (movement)...")
    await controller.move(0, 1.0, duration=1.0)  # Forward
    await asyncio.sleep(0.5)
    await controller.move(1.0, 0, duration=1.0)  # Right
    await asyncio.sleep(0.5)
    
    print("Testing right stick (camera)...")
    await controller.look(1.0, 0, duration=0.5)  # Look right
    await asyncio.sleep(0.5)
    await controller.look(-1.0, 0, duration=0.5)  # Look left
    await asyncio.sleep(0.5)
    
    # Test triggers
    print("Testing triggers...")
    controller.set_left_trigger(1.0)
    await asyncio.sleep(0.5)
    controller.set_left_trigger(0.0)
    await asyncio.sleep(0.5)
    
    controller.set_right_trigger(1.0)
    await asyncio.sleep(0.5)
    controller.set_right_trigger(0.0)
    
    print("OK Basic inputs test complete")


async def test_smooth_movement(controller: VirtualXboxController):
    """Test smooth stick movements."""
    print("\n=== Testing Smooth Movement ===")
    
    # Smooth circular motion with left stick
    print("Smooth circular motion...")
    import math
    steps = 20
    duration = 2.0
    
    for i in range(steps):
        angle = (i / steps) * 2 * math.pi
        x = math.cos(angle) * 0.8
        y = math.sin(angle) * 0.8
        controller.set_left_stick(x, y)
        await asyncio.sleep(duration / steps)
    
    controller.set_left_stick(0, 0)
    
    # Smooth camera pan
    print("Smooth camera pan...")
    await controller.smooth_stick_movement('right', 1.0, 0, duration=1.0)
    await controller.smooth_stick_movement('right', 0, 0, duration=0.5)
    
    print("OK Smooth movement test complete")


async def test_action_layers(controller: VirtualXboxController, bindings: SkyrimControllerBindings):
    """Test Steam Input-style action layers."""
    print("\n=== Testing Action Layers ===")
    
    # Test exploration layer
    print("Testing Exploration layer...")
    bindings.switch_to_exploration()
    await controller.execute_action("jump")
    await asyncio.sleep(0.5)
    await controller.execute_action("activate")
    await asyncio.sleep(0.5)
    
    # Test combat layer
    print("Testing Combat layer...")
    bindings.switch_to_combat()
    await controller.execute_action("attack")
    await asyncio.sleep(0.5)
    await controller.execute_action("block")
    await asyncio.sleep(0.5)
    
    # Test menu layer
    print("Testing Menu layer...")
    bindings.switch_to_menu()
    await controller.execute_action("up")
    await asyncio.sleep(0.3)
    await controller.execute_action("select")
    await asyncio.sleep(0.5)
    
    print("OK Action layers test complete")


async def test_combat_sequence(controller: VirtualXboxController, bindings: SkyrimControllerBindings):
    """Test combat action sequence."""
    print("\n=== Testing Combat Sequence ===")
    
    bindings.switch_to_combat()
    
    print("Executing combat combo...")
    await bindings.combat_combo_light_heavy()
    await asyncio.sleep(1.0)
    
    print("Executing defensive maneuver...")
    await bindings.defensive_maneuver()
    await asyncio.sleep(1.0)
    
    print("OK Combat sequence test complete")


async def test_exploration_scan(controller: VirtualXboxController, bindings: SkyrimControllerBindings):
    """Test exploration scanning."""
    print("\n=== Testing Exploration Scan ===")
    
    bindings.switch_to_exploration()
    
    print("Scanning area (360-degree look)...")
    await bindings.exploration_scan(duration=3.0)
    
    print("OK Exploration scan test complete")


async def test_input_recording(controller: VirtualXboxController):
    """Test input recording and playback."""
    print("\n=== Testing Input Recording ===")
    
    print("Recording inputs...")
    controller.start_recording()
    
    # Perform some actions
    await controller.tap_button(XboxButton.A)
    await asyncio.sleep(0.5)
    await controller.move(0, 1.0, duration=1.0)
    await asyncio.sleep(0.5)
    await controller.tap_button(XboxButton.B)
    
    recording = controller.stop_recording()
    print(f"Recorded {len(recording)} events")
    
    # Play back
    print("Playing back recording...")
    await asyncio.sleep(1.0)
    await controller.playback_recording(recording)
    
    print("OK Input recording test complete")


async def demo_gameplay_scenario(controller: VirtualXboxController, bindings: SkyrimControllerBindings):
    """Demonstrate a realistic gameplay scenario."""
    print("\n=== Gameplay Scenario Demo ===")
    print("Simulating: Explore → Encounter enemy → Combat → Victory\n")
    
    # 1. Exploration
    print("Phase 1: Exploring...")
    bindings.switch_to_exploration()
    await controller.execute_action("move_forward")
    await asyncio.sleep(1.0)
    await bindings.exploration_scan(duration=2.0)
    
    # 2. Spot enemy
    print("\nPhase 2: Enemy spotted!")
    await controller.execute_action("sneak")
    await asyncio.sleep(0.5)
    
    # 3. Enter combat
    print("\nPhase 3: Engaging in combat...")
    bindings.switch_to_combat()
    
    # Attack sequence
    await controller.execute_action("attack")
    await asyncio.sleep(0.5)
    await controller.execute_action("attack")
    await asyncio.sleep(0.5)
    
    # Defensive
    await controller.execute_action("block")
    await asyncio.sleep(0.5)
    
    # Power attack
    await controller.execute_action("power_attack")
    await asyncio.sleep(1.0)
    
    # 4. Victory
    print("\nPhase 4: Victory! Looting...")
    bindings.switch_to_exploration()
    await controller.execute_action("activate")
    await asyncio.sleep(0.5)
    
    print("\nOK Gameplay scenario complete")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test virtual Xbox controller')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no actual input)')
    parser.add_argument('--test', choices=['all', 'basic', 'smooth', 'layers', 'combat', 'scan', 'record', 'demo'],
                       default='all', help='Which test to run')
    args = parser.parse_args()
    
    print("=" * 60)
    print("VIRTUAL XBOX 360 CONTROLLER TEST")
    print("Steam Input Style for Skyrim")
    print("=" * 60)
    
    # Initialize controller
    controller = VirtualXboxController(dry_run=args.dry_run)
    bindings = SkyrimControllerBindings(controller)
    
    # Default to exploration layer
    bindings.switch_to_exploration()
    
    try:
        # Run tests
        if args.test == 'all' or args.test == 'basic':
            await test_basic_inputs(controller)
        
        if args.test == 'all' or args.test == 'smooth':
            await test_smooth_movement(controller)
        
        if args.test == 'all' or args.test == 'layers':
            await test_action_layers(controller, bindings)
        
        if args.test == 'all' or args.test == 'combat':
            await test_combat_sequence(controller, bindings)
        
        if args.test == 'all' or args.test == 'scan':
            await test_exploration_scan(controller, bindings)
        
        if args.test == 'all' or args.test == 'record':
            await test_input_recording(controller)
        
        if args.test == 'demo':
            await demo_gameplay_scenario(controller, bindings)
        
        # Show statistics
        print("\n" + "=" * 60)
        print("CONTROLLER STATISTICS")
        print("=" * 60)
        stats = controller.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    finally:
        # Reset controller
        print("\nResetting controller...")
        controller.reset()
    
    print("\nOK All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
