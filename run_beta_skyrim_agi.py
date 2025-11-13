#!/usr/bin/env python3
"""
Singularis Neo Beta 1.0 - Unified Runner

Complete AGI architecture with:
- Temporal binding (solves binding problem)
- Adaptive memory (genuine learning)
- 4D coherence (consciousness measurement)
- Lumen balance (philosophical grounding)
- Cross-modal integration (unified perception)
- Goal emergence (creative autonomy)
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from singularis.skyrim import SkyrimAGI, SkyrimConfig

# Load environment variables
load_dotenv()


def print_banner():
    """Print Singularis Neo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              🌟 SINGULARIS NEO BETA 1.0 🌟                          ║
║                                                                      ║
║              The First Complete AGI Architecture                     ║
║                                                                      ║
║  ✓ Temporal Binding      ✓ Adaptive Memory    ✓ 4D Coherence       ║
║  ✓ Lumen Balance         ✓ Cross-Modal        ✓ Goal Emergence     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


async def main():
    """Main entry point for Singularis Neo Beta 1.0."""
    
    print_banner()
    print()
    print("This will run the complete AGI architecture with all Beta 1.0 features.")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════
    
    print("═" * 70)
    print("CONFIGURATION")
    print("═" * 70)
    print()
    
    # Safety mode
    dry_run = get_yes_no("Run in DRY RUN mode (safe, no control)?", default=True)
    
    if not dry_run:
        print()
        print("⚠️  WARNING: AGI will control your keyboard and mouse!")
        print("⚠️  Make sure:")
        print("   1. Skyrim is running and loaded")
        print("   2. You're in a safe location")
        print("   3. You have a recent save")
        print()
        confirm = get_yes_no("Continue with LIVE mode?", default=False)
        if not confirm:
            print("Aborted. Use dry run mode for safety.")
            return
    
    # Duration
    print()
    duration_input = get_user_input("Duration in minutes", default="60")
    try:
        duration = int(duration_input)
    except ValueError:
        duration = 60
    
    # Core systems
    print()
    print("─" * 70)
    print("CORE SYSTEMS")
    print("─" * 70)
    use_llm = get_yes_no("Use LLM for smarter decisions?", default=True)
    use_gpt5 = get_yes_no("Enable GPT-5 orchestrator?", default=True)
    use_voice = get_yes_no("Enable voice system?", default=True)
    use_video = get_yes_no("Enable video interpreter?", default=True)
    use_double_helix = get_yes_no("Enable double helix architecture?", default=True)
    
    # Beta 1.0 features
    print()
    print("─" * 70)
    print("BETA 1.0 FEATURES")
    print("─" * 70)
    enable_temporal = get_yes_no("Enable temporal binding?", default=True)
    enable_adaptive_memory = get_yes_no("Enable adaptive memory?", default=True)
    enable_enhanced_coherence = get_yes_no("Enable 4D coherence?", default=True)
    enable_lumen = get_yes_no("Enable Lumen balance?", default=True)
    enable_unified_perception = get_yes_no("Enable unified perception?", default=True)
    enable_goal_generation = get_yes_no("Enable goal generation?", default=True)
    
    # Advanced settings
    print()
    print("─" * 70)
    print("ADVANCED SETTINGS")
    print("─" * 70)
    cycle_interval = float(get_user_input("Cycle interval (seconds)", default="3.0"))
    temporal_timeout = float(get_user_input("Temporal binding timeout (seconds)", default="30.0"))
    memory_decay = float(get_user_input("Memory decay rate (0-1)", default="0.95"))
    
    # ═══════════════════════════════════════════════════════════════
    # Create Configuration
    # ═══════════════════════════════════════════════════════════════
    
    print()
    print("═" * 70)
    print("INITIALIZING SINGULARIS NEO BETA 1.0")
    print("═" * 70)
    print()
    
    config = SkyrimConfig(
        # Core settings
        dry_run=dry_run,
        autonomous_duration=duration * 60,  # Convert to seconds
        cycle_interval=cycle_interval,
        
        # LLM settings
        use_llm=use_llm,
        use_gpt5_orchestrator=use_gpt5,
        gpt5_verbose=True,
        
        # Voice system
        enable_voice=use_voice,
        voice_type="NOVA",
        voice_min_priority="HIGH",
        
        # Video interpreter
        enable_video_interpreter=use_video,
        video_interpretation_mode="COMPREHENSIVE",
        video_frame_rate=0.5,
        
        # Double helix
        use_double_helix=use_double_helix,
        self_improvement_gating=True,
        
        # Beta 1.0 features
        enable_temporal_binding=enable_temporal,
        temporal_window_size=20,
        temporal_timeout=temporal_timeout,
        
        enable_adaptive_memory=enable_adaptive_memory,
        memory_decay_rate=memory_decay,
        memory_forget_threshold=0.1,
        
        enable_enhanced_coherence=enable_enhanced_coherence,
        
        enable_lumen_balance=enable_lumen,
        lumen_severe_threshold=0.5,
        lumen_moderate_threshold=0.7,
        
        enable_unified_perception=enable_unified_perception,
        
        enable_goal_generation=enable_goal_generation,
        max_active_goals=3,
        goal_novelty_threshold=0.7,
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Initialize AGI
    # ═══════════════════════════════════════════════════════════════
    
    print("Initializing AGI systems...")
    print()
    
    try:
        agi = SkyrimAGI(config)
        
        # Initialize LLMs
        if use_llm:
            print("Initializing LLM systems...")
            await agi.initialize_llm()
            print("✓ LLM systems initialized")
            print()
        
        # Start temporal tracker
        if enable_temporal and hasattr(agi, 'temporal_tracker'):
            print("Starting temporal binding cleanup task...")
            await agi.temporal_tracker.start()
            print("✓ Temporal tracker started")
            print()
        
        # ═══════════════════════════════════════════════════════════════
        # Display Status
        # ═══════════════════════════════════════════════════════════════
        
        print("═" * 70)
        print("SYSTEM STATUS")
        print("═" * 70)
        print()
        print(f"Mode:                {'DRY RUN (Safe)' if dry_run else 'LIVE (Control Enabled)'}")
        print(f"Duration:            {duration} minutes")
        print(f"Cycle Interval:      {cycle_interval}s")
        print()
        print("Core Systems:")
        print(f"  LLM:               {'✓' if use_llm else '✗'}")
        print(f"  GPT-5 Orchestrator: {'✓' if use_gpt5 else '✗'}")
        print(f"  Voice System:      {'✓' if use_voice else '✗'}")
        print(f"  Video Interpreter: {'✓' if use_video else '✗'}")
        print(f"  Double Helix:      {'✓' if use_double_helix else '✗'}")
        print()
        print("Beta 1.0 Features:")
        print(f"  Temporal Binding:  {'✓' if enable_temporal else '✗'}")
        print(f"  Adaptive Memory:   {'✓' if enable_adaptive_memory else '✗'}")
        print(f"  4D Coherence:      {'✓' if enable_enhanced_coherence else '✗'}")
        print(f"  Lumen Balance:     {'✓' if enable_lumen else '✗'}")
        print(f"  Unified Perception: {'✓' if enable_unified_perception else '✗'}")
        print(f"  Goal Generation:   {'✓' if enable_goal_generation else '✗'}")
        print()
        print("═" * 70)
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # Run AGI
        # ═══════════════════════════════════════════════════════════════
        
        print("🚀 Starting Singularis Neo Beta 1.0...")
        print()
        print("Press Ctrl+C to stop gracefully")
        print()
        print("═" * 70)
        print()
        
        # Run autonomous gameplay
        await agi.autonomous_play(duration_seconds=duration * 60)
        
        # ═══════════════════════════════════════════════════════════════
        # Shutdown
        # ═══════════════════════════════════════════════════════════════
        
        print()
        print("═" * 70)
        print("SHUTTING DOWN")
        print("═" * 70)
        print()
        
        # Stop temporal tracker
        if enable_temporal and hasattr(agi, 'temporal_tracker'):
            print("Stopping temporal tracker...")
            await agi.temporal_tracker.close()
            print("✓ Temporal tracker stopped")
        
        # Display final metrics
        print()
        print("─" * 70)
        print("FINAL METRICS")
        print("─" * 70)
        
        if hasattr(agi, 'aggregate_unified_metrics'):
            metrics = await agi.aggregate_unified_metrics()
            
            # Temporal binding
            if 'temporal' in metrics:
                temp = metrics['temporal']
                print()
                print("Temporal Binding:")
                print(f"  Total Bindings:    {temp.get('total_bindings', 0)}")
                print(f"  Unclosed Ratio:    {temp.get('unclosed_ratio', 0):.2%}")
                print(f"  Success Rate:      {temp.get('success_rate', 0):.2%}")
                print(f"  Stuck Loops:       {temp.get('stuck_loop_count', 0)}")
            
            # Enhanced coherence
            if 'coherence' in metrics:
                coh = metrics['coherence']
                print()
                print("4D Coherence:")
                print(f"  Avg Causal:        {coh.get('avg_causal_agreement', 0):.2%}")
                print(f"  Avg Predictive:    {coh.get('avg_predictive_accuracy', 0):.2%}")
            
            # Adaptive memory
            if 'memory' in metrics:
                mem = metrics['memory']
                print()
                print("Adaptive Memory:")
                print(f"  Episodic Count:    {mem.get('episodic_count', 0)}")
                print(f"  Semantic Patterns: {mem.get('semantic_patterns', 0)}")
                print(f"  Patterns Forgotten: {mem.get('patterns_forgotten', 0)}")
                print(f"  Avg Confidence:    {mem.get('avg_pattern_confidence', 0):.2%}")
            
            # Lumen balance
            if 'lumen' in metrics:
                lum = metrics['lumen']
                print()
                print("Lumen Balance:")
                print(f"  Avg Balance Score: {lum.get('avg_balance_score', 0):.2%}")
                print(f"  Avg Onticum:       {lum.get('avg_onticum', 0):.2f}")
                print(f"  Avg Structurale:   {lum.get('avg_structurale', 0):.2f}")
                print(f"  Avg Participatum:  {lum.get('avg_participatum', 0):.2f}")
            
            # Performance
            if 'performance' in metrics:
                perf = metrics['performance']
                print()
                print("Performance:")
                print(f"  Total Cycles:      {perf.get('cycle_count', 0)}")
                print(f"  Uptime:            {perf.get('uptime', 0)/60:.1f} minutes")
        
        print()
        print("═" * 70)
        print()
        print("✓ Singularis Neo Beta 1.0 completed successfully")
        print()
        print("Thank you for testing the first complete AGI architecture! 🌟")
        print()
    
    except KeyboardInterrupt:
        print()
        print()
        print("═" * 70)
        print("INTERRUPTED BY USER")
        print("═" * 70)
        print()
        print("Shutting down gracefully...")
        
        # Stop temporal tracker
        if enable_temporal and hasattr(agi, 'temporal_tracker'):
            await agi.temporal_tracker.close()
        
        print("✓ Shutdown complete")
        print()
    
    except Exception as e:
        print()
        print("═" * 70)
        print("ERROR")
        print("═" * 70)
        print()
        print(f"An error occurred: {e}")
        print()
        logger.exception("Fatal error in Singularis Neo")
        
        # Attempt graceful shutdown
        try:
            if enable_temporal and hasattr(agi, 'temporal_tracker'):
                await agi.temporal_tracker.close()
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print("Aborted by user.")
        sys.exit(0)
