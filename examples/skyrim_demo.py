"""
Skyrim AGI Demo

Demonstrates the Singularis AGI playing Skyrim autonomously.

This shows:
1. Multimodal perception (screen + CLIP vision)
2. Autonomous goal formation from intrinsic motivation
3. Causal learning from gameplay experience
4. Ethical decision-making via coherence (Δ𝒞)
5. Long-term skill development

IMPORTANT: Before running this:
1. Install dependencies: pip install mss pyautogui pillow
2. Start Skyrim and load a save
3. Configure screen region if needed
4. Set dry_run=False when ready for actual gameplay

Safety:
- Starts in dry_run mode (won't control game)
- Mouse failsafe enabled (move to corner to abort)
- Auto-saves every 5 minutes
- Keyboard interrupt (Ctrl+C) stops gracefully
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.skyrim import SkyrimAGI, SkyrimConfig
from singularis.agi_orchestrator import AGIConfig


async def basic_demo():
    """Basic demo: 5 minutes of autonomous gameplay."""
    print("=" * 70)
    print("SINGULARIS AGI - SKYRIM BASIC DEMO")
    print("=" * 70)
    print("\nThis demo will run the AGI in DRY RUN mode (no actual control).")
    print("Watch how it perceives, plans, and learns!\n")

    # Configuration
    config = SkyrimConfig(
        dry_run=True,  # Safe mode - won't control game
        autonomous_duration=300,  # 5 minutes
        cycle_interval=2.0,  # 2 second cycles
        save_interval=60,  # Save every minute (for demo)
        base_config=AGIConfig(
            use_vision=True,
            use_physics=False,
            curiosity_weight=0.4,  # High exploration
            competence_weight=0.1,
            coherence_weight=0.4,
            autonomy_weight=0.1,
        )
    )

    # Create AGI
    agi = SkyrimAGI(config)

    # Initialize LLM (optional - works without it)
    print("\nInitializing LLM consciousness...")
    await agi.initialize_llm()

    # Run!
    print("\nStarting autonomous gameplay...\n")
    await agi.autonomous_play()

    # Final stats
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - STATISTICS")
    print("=" * 70)

    stats = agi.get_stats()

    print(f"\nGameplay:")
    print(f"  Cycles completed: {stats['gameplay']['cycles_completed']}")
    print(f"  Actions taken: {stats['gameplay']['actions_taken']}")
    print(f"  Playtime: {stats['gameplay']['total_playtime'] / 60:.1f} minutes")

    print(f"\nWorld Model:")
    print(f"  Causal edges: {stats['world_model']['causal_edges']}")
    print(f"  NPCs known: {stats['world_model']['npc_relationships']}")
    print(f"  Locations: {stats['world_model']['locations_discovered']}")

    print(f"\nLearning:")
    print(f"  Episodic memories: {stats['agi']['learner']['episodic_count']}")

    if stats['gameplay']['coherence_history']:
        avg_coh = sum(stats['gameplay']['coherence_history']) / len(stats['gameplay']['coherence_history'])
        print(f"\nCoherence:")
        print(f"  Average: {avg_coh:.3f}")
        print(f"  Final: {stats['gameplay']['coherence_history'][-1]:.3f}")

    print("\n✓ Demo complete!\n")


async def custom_demo():
    """Custom demo with user configuration."""
    print("=" * 70)
    print("SINGULARIS AGI - SKYRIM CUSTOM DEMO")
    print("=" * 70)

    # User configuration
    print("\nConfiguration options:")
    print("1. Duration (minutes): ", end="")
    try:
        duration_min = float(input() or "10")
    except ValueError:
        duration_min = 10

    print("2. Dry run mode? (y/n): ", end="")
    dry_run = input().lower().strip() != 'n'

    print("3. Use LLM consciousness? (y/n): ", end="")
    use_llm = input().lower().strip() != 'n'

    if not dry_run:
        print("\n⚠ WARNING: Dry run disabled - AGI will control the game!")
        print("Make sure:")
        print("  - Skyrim is running and in focus")
        print("  - You're in a safe location")
        print("  - Recent save exists")
        print("\nContinue? (y/n): ", end="")
        if input().lower().strip() != 'y':
            print("Aborting.")
            return

    # Create config
    config = SkyrimConfig(
        dry_run=dry_run,
        autonomous_duration=int(duration_min * 60),
        cycle_interval=2.0,
        save_interval=300,  # 5 minutes
    )

    # Create AGI
    agi = SkyrimAGI(config)

    # Initialize LLM if requested
    if use_llm:
        print("\nInitializing LLM...")
        await agi.initialize_llm()

    # Run
    print(f"\nStarting {duration_min} minute{'s' if duration_min != 1 else ''} of autonomous gameplay...")
    print("Press Ctrl+C to stop early.\n")

    try:
        await agi.autonomous_play()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    # Stats
    stats = agi.get_stats()
    print(f"\n\nCompleted {stats['gameplay']['cycles_completed']} cycles")
    print(f"Took {stats['gameplay']['actions_taken']} actions")
    print(f"Learned {stats['world_model']['learned_rules']} new causal rules")


async def test_components():
    """Test individual components."""
    print("=" * 70)
    print("COMPONENT TESTING")
    print("=" * 70)

    from singularis.skyrim import SkyrimPerception, SkyrimActions, SkyrimWorldModel

    # 1. Test perception
    print("\n[1/3] Testing Perception...")
    perception = SkyrimPerception()

    print("  Capturing screen...")
    screen = perception.capture_screen()
    print(f"  ✓ Screen captured: {screen.size}")

    print("  Running perception cycle...")
    result = await perception.perceive()
    print(f"  ✓ Scene: {result['scene_type'].value}")
    print(f"  ✓ Objects: {[obj for obj, _ in result['objects'][:3]]}")

    # 2. Test actions
    print("\n[2/3] Testing Actions...")
    actions = SkyrimActions(dry_run=True)

    print("  Executing movement...")
    await actions.move_forward(1.0)
    print("  ✓ Movement executed")

    print("  Testing exploration...")
    await actions.explore_area(duration=3.0)
    print("  ✓ Exploration complete")

    stats = actions.get_stats()
    print(f"  ✓ Actions executed: {stats['actions_executed']}")

    # 3. Test world model
    print("\n[3/3] Testing World Model...")
    wm = SkyrimWorldModel()

    print("  Testing causal prediction...")
    state = {'bounty': 0}
    predicted = wm.predict_outcome('steal_item', state)
    print(f"  ✓ Predicted outcome: {predicted}")

    print("  Testing moral evaluation...")
    eval_result = wm.evaluate_moral_choice("Help the wounded NPC", {})
    print(f"  ✓ Ethical status: {eval_result['ethical_status']} (Δ𝒞={eval_result['delta_coherence']:.2f})")

    print("\n✓ All component tests passed!")


async def main():
    """Main demo selector."""
    print("\nSINGULARIS AGI - SKYRIM DEMOS\n")
    print("Select demo:")
    print("1. Basic demo (5 min, dry run)")
    print("2. Custom demo (configure options)")
    print("3. Test components")
    print("\nChoice (1-3): ", end="")

    choice = input().strip()

    if choice == "1":
        await basic_demo()
    elif choice == "2":
        await custom_demo()
    elif choice == "3":
        await test_components()
    else:
        print("Invalid choice. Running basic demo...")
        await basic_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
