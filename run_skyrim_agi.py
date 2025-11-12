"""
Run Singularis AGI in Skyrim

Complete autonomous gameplay with all features enabled.
"""

import asyncio
from singularis.skyrim import SkyrimAGI, SkyrimConfig

async def main():
    print("=" * 70)
    print("SINGULARIS AGI - SKYRIM AUTONOMOUS GAMEPLAY")
    print("=" * 70)
    print()
    print("This will run the AGI in Skyrim with full autonomy.")
    print()
    
    # Configuration
    dry_run = input("Run in DRY RUN mode (safe, no control)? [Y/n]: ").strip().lower()
    dry_run = dry_run != 'n'
    
    if not dry_run:
        print()
        print("⚠️  WARNING: AGI will control your keyboard and mouse!")
        print("⚠️  Make sure:")
        print("   1. Skyrim is running and loaded")
        print("   2. You're in a safe location")
        print("   3. You have a recent save")
        print()
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return
    
    duration = input("\nDuration in minutes [60]: ").strip()
    duration = int(duration) if duration else 60
    
    use_llm = input("Use LLM for smarter decisions? [Y/n]: ").strip().lower()
    use_llm = use_llm != 'n'
    
    print()
    print("Configuration:")
    print(f"  Dry run: {dry_run}")
    print(f"  Duration: {duration} minutes")
    print(f"  LLM: {use_llm}")
    print()
    
    # Create config
    config = SkyrimConfig(
        dry_run=dry_run,
        autonomous_duration=duration * 60,
        cycle_interval=2.0,
        save_interval=300,
        surprise_threshold=0.3,
        exploration_weight=0.5,
        phi4_main_model="microsoft/phi-4-mini-reasoning:1",
        phi4_rl_model="microsoft/phi-4-mini-reasoning:2",
        phi4_meta_model="microsoft/phi-4-mini-reasoning:3",
        phi4_action_model="microsoft/phi-4-mini-reasoning:4",
    )
    
    # Create AGI
    agi = SkyrimAGI(config)
    
    # Initialize LLM if requested
    if use_llm:
        print("Initializing LLM...")
        try:
            await agi.initialize_llm()
            print("✓ LLM initialized")
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            print("Continuing without LLM (will use heuristics)")
    
    print()
    
    if not dry_run:
        print("⚠️  AGI will start controlling Skyrim in 5 seconds...")
        print("⚠️  Move mouse to top-left corner to abort!")
        for i in range(5, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)
        print()
    
    # Run!
    print("Starting autonomous gameplay...")
    print("=" * 70)
    print()
    
    try:
        await agi.autonomous_play()
    except KeyboardInterrupt:
        print("\n\nGracefully shutting down...")
    
    # Show final stats
    print()
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    stats = agi.get_stats()
    
    print(f"\nGameplay:")
    print(f"  Cycles completed: {stats['gameplay']['cycles_completed']}")
    print(f"  Actions taken: {stats['gameplay']['actions_taken']}")
    print(f"  Playtime: {stats['gameplay']['total_playtime'] / 60:.1f} minutes")
    
    print(f"\nLearning:")
    print(f"  Causal rules learned: {stats['world_model']['causal_edges']}")
    print(f"  NPCs met: {stats['world_model']['npc_relationships']}")
    print(f"  Locations discovered: {stats['world_model']['locations_discovered']}")
    
    if 'coherence_history' in stats['gameplay'] and stats['gameplay']['coherence_history']:
        avg_coherence = sum(stats['gameplay']['coherence_history']) / len(stats['gameplay']['coherence_history'])
        print(f"\nCoherence:")
        print(f"  Average coherence: {avg_coherence:.3f}")
    
    if 'motivations' in stats and 'dominant_drive' in stats['motivations']:
        print(f"\nDominant Motivation:")
        print(f"  {stats['motivations']['dominant_drive']}")
    
    print()
    print("=" * 70)
    print("✓ AGI session complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
