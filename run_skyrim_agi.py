"""
Run Singularis AGI in Skyrim

Complete autonomous gameplay with all features enabled.
"""

import asyncio
from dotenv import load_dotenv
from singularis.skyrim import SkyrimAGI, SkyrimConfig

# Load API keys from .env file
load_dotenv()

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
    
    # LLM Architecture Options
    llm_mode = "hybrid"
    use_local_fallback = False
    
    if use_llm:
        print("\nLLM Architecture Options:")
        print("  1. Hybrid (Gemini vision + Claude Sonnet 4 reasoning) [Default]")
        print("  2. Hybrid with local fallback (adds optional local LLMs)")
        print("  3. MoE (6 Gemini + 3 Claude experts with rate limiting)")
        print("  4. PARALLEL (MoE + Hybrid simultaneously - MAXIMUM INTELLIGENCE)")
        print("  5. Local only (LM Studio models only)")
        
        llm_choice = input("\nSelect LLM mode [1]: ").strip()
        
        if llm_choice == "2":
            llm_mode = "hybrid"
            use_local_fallback = True
        elif llm_choice == "3":
            llm_mode = "moe"
            use_local_fallback = False
        elif llm_choice == "4":
            llm_mode = "parallel"
            use_local_fallback = False
        elif llm_choice == "5":
            llm_mode = "local"
            use_local_fallback = False
        else:
            llm_mode = "hybrid"
            use_local_fallback = False
    
    print()
    print("Configuration:")
    print(f"  Dry run: {dry_run}")
    print(f"  Duration: {duration} minutes")
    print(f"  LLM: {use_llm}")
    if use_llm:
        if llm_mode == "hybrid":
            print(f"  LLM Mode: Hybrid (Gemini + Claude Sonnet 4)")
            if use_local_fallback:
                print(f"  Local Fallback: Enabled")
            else:
                print(f"  Local Fallback: Disabled")
        elif llm_mode == "moe":
            print(f"  LLM Mode: MoE (6 Gemini + 3 Claude experts)")
            print(f"  Rate Limiting: Enabled (10 RPM Gemini, 50 RPM Claude)")
        elif llm_mode == "parallel":
            print(f"  LLM Mode: PARALLEL (MoE + Hybrid simultaneously)")
            print(f"  Total: 10 LLM instances running in parallel")
            print(f"  Consensus: MoE 60% + Hybrid 40%")
        else:
            print(f"  LLM Mode: Local only (LM Studio)")
    print()
    
    # Create config based on LLM mode selection
    if llm_mode == "parallel":
        # Parallel mode: MoE + Hybrid simultaneously
        config = SkyrimConfig(
            dry_run=dry_run,
            autonomous_duration=duration * 60,
            cycle_interval=2.0,
            save_interval=300,
            surprise_threshold=0.3,
            exploration_weight=0.5,
            
            # Parallel mode configuration
            use_parallel_mode=True,
            use_moe=False,  # Handled by parallel mode
            use_hybrid_llm=False,  # Handled by parallel mode
            
            # MoE settings
            num_gemini_experts=6,
            num_claude_experts=3,
            gemini_model="gemini-2.0-flash-exp",
            claude_model="claude-sonnet-4-20250514",
            gemini_rpm_limit=10,
            claude_rpm_limit=50,
            
            # Hybrid settings
            use_gemini_vision=True,
            use_claude_reasoning=True,
            use_local_fallback=False,
            
            # Consensus weights
            parallel_consensus_weight_moe=0.6,
            parallel_consensus_weight_hybrid=0.4,
            
            # Cloud RL enabled by default in parallel mode
            use_cloud_rl=True,
            rl_use_rag=True,
            rl_cloud_reward_shaping=True,
            rl_moe_evaluation=True,
            
            # Disable legacy
            enable_claude_meta=False,
            enable_gemini_vision=False,
        )
    elif llm_mode == "moe":
        # MoE mode: 6 Gemini + 3 Claude experts with rate limiting
        config = SkyrimConfig(
            dry_run=dry_run,
            autonomous_duration=duration * 60,
            cycle_interval=2.0,
            save_interval=300,
            surprise_threshold=0.3,
            exploration_weight=0.5,
            
            # MoE configuration
            use_moe=True,
            num_gemini_experts=6,
            num_claude_experts=3,
            gemini_model="gemini-2.0-flash-exp",
            claude_model="claude-sonnet-4-20250514",
            gemini_rpm_limit=10,  # Conservative rate limit
            claude_rpm_limit=50,  # Conservative rate limit
            
            # Disable other LLM modes
            use_hybrid_llm=False,
            use_gemini_vision=False,
            use_claude_reasoning=False,
            use_local_fallback=False,
            enable_claude_meta=False,
            enable_gemini_vision=False,
        )
    elif llm_mode == "hybrid":
        # Hybrid mode: Gemini + Claude with optional local fallback
        config = SkyrimConfig(
            dry_run=dry_run,
            autonomous_duration=duration * 60,
            cycle_interval=2.0,
            save_interval=300,
            surprise_threshold=0.3,
            exploration_weight=0.5,
            
            # Hybrid LLM configuration
            use_hybrid_llm=True,
            use_gemini_vision=True,
            gemini_model="gemini-2.0-flash-exp",
            use_claude_reasoning=True,
            claude_model="claude-sonnet-4-20250514",
            use_local_fallback=use_local_fallback,
            
            # MoE disabled
            use_moe=False,
            
            # Local models (used only if fallback enabled)
            phi4_action_model="mistralai/mistral-nemo-instruct-2407",
            huihui_cognition_model="huihui-moe-60b-a3b-abliterated-i1",
            qwen3_vl_perception_model="qwen/qwen3-vl-8b",
            
            # Disable legacy augmentation
            enable_claude_meta=False,
            enable_gemini_vision=False,
        )
    else:
        # Local only mode: LM Studio models
        config = SkyrimConfig(
            dry_run=dry_run,
            autonomous_duration=duration * 60,
            cycle_interval=2.0,
            save_interval=300,
            surprise_threshold=0.3,
            exploration_weight=0.5,
            
            # Disable hybrid system
            use_hybrid_llm=False,
            use_gemini_vision=False,
            use_claude_reasoning=False,
            use_local_fallback=False,
            
            # Local models only
            phi4_action_model="mistralai/mistral-nemo-instruct-2407",
            huihui_cognition_model="huihui-moe-60b-a3b-abliterated-i1",
            qwen3_vl_perception_model="qwen/qwen3-vl-8b",
            
            # Disable legacy augmentation
            enable_claude_meta=False,
            enable_gemini_vision=False,
        )
    
    # Create AGI
    agi = SkyrimAGI(config)
    
    # Initialize LLM if requested
    if use_llm:
        print("\n" + "=" * 70)
        print("INITIALIZING CLOUD LLM SYSTEMS")
        print("=" * 70)
        try:
            await agi.initialize_llm()
            print("\n✓ LLM initialization complete")
        except Exception as e:
            print(f"\n⚠️  LLM initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing without LLM (will use heuristics)")
    
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
