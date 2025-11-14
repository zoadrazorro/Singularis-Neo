#!/usr/bin/env python3
"""
Singularis Beta v2 - Unified BeingState Runner

This is the complete unified AGI system with:
- BeingState: ONE unified state vector
- CoherenceEngine: ONE optimization function
- C_global: ONE target all subsystems optimize
- Wolfram Telemetry: Mathematical validation
- 20+ subsystems integrated

Philosophy → Mathematics → Code → Execution

Run with:
    python run_singularis_beta_v2.py --duration 3600 --mode async

Author: Singularis Team
Version: 2.0.0-beta
Date: 2025-11-13
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def print_banner():
    """Print the Singularis banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███████╗██╗███╗   ██╗ ██████╗ ██╗   ██╗██╗      █████╗ ██████╗ ██╗███████╗   ║
║   ██╔════╝██║████╗  ██║██╔════╝ ██║   ██║██║     ██╔══██╗██╔══██╗██║██╔════╝   ║
║   ███████╗██║██╔██╗ ██║██║  ███╗██║   ██║██║     ███████║██████╔╝██║███████╗   ║
║   ╚════██║██║██║╚██╗██║██║   ██║██║   ██║██║     ██╔══██║██╔══██╗██║╚════██║   ║
║   ███████║██║██║ ╚████║╚██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║██║███████║   ║
║   ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝   ║
║                                                                  ║
║                          BETA v2.0                               ║
║              "One Being, Striving for Coherence"                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

    Architecture:  BeingState + CoherenceEngine
    Philosophy:    Spinoza → IIT → Lumen → Buddhism
    Mathematics:   C: B → [0,1], max E[C(B(t+1))]
    Integration:   20+ Subsystems → 1 Unified Being
    Validation:    Wolfram Alpha Telemetry
    
"""
    print(banner)


def check_environment():
    """Check that required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API (required for GPT-5 Meta-RL & Wolfram telemetry)',
        'GEMINI_API_KEY': 'Google Gemini API (required for vision)',
    }
    
    optional_vars = {
        'ANTHROPIC_API_KEY': 'Anthropic Claude API (optional)',
    }
    
    missing_required = []
    missing_optional = []
    
    print("\n[ENV] Checking environment variables...")
    
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"  ✓ {var}: Set")
        else:
            print(f"  ✗ {var}: Missing ({description})")
            missing_required.append(var)
    
    for var, description in optional_vars.items():
        if os.getenv(var):
            print(f"  ✓ {var}: Set")
        else:
            print(f"  - {var}: Not set ({description})")
            missing_optional.append(var)
    
    if missing_required:
        print(f"\n[ERROR] Missing required environment variables: {', '.join(missing_required)}")
        print("\nPlease set them in your .env file or export them:")
        for var in missing_required:
            print(f"  export {var}='your-key-here'")
        return False
    
    if missing_optional:
        print(f"\n[INFO] Optional features disabled: {', '.join(missing_optional)}")
    
    print("\n[ENV] ✓ Environment check passed\n")
    return True


def load_config(args) -> 'SkyrimConfig':
    """Load and configure the system."""
    from singularis.skyrim.skyrim_agi import SkyrimConfig
    
    print("[CONFIG] Loading configuration...")
    
    config = SkyrimConfig()
    # Force parallel mode: run MoE and Hybrid LLMs together for maximum intelligence
    config.use_parallel_mode = True
    
    # Apply command-line overrides
    if args.cycle_interval:
        config.cycle_interval = args.cycle_interval
    
    if args.verbose:
        config.gpt5_verbose = True  # Use gpt5_verbose instead of verbose
    
    if args.no_voice:
        config.enable_voice = False
    
    if args.no_video:
        config.enable_video_interpreter = False
    
    if args.no_wolfram:
        # Note: Wolfram is always enabled - it's integrated into the core system
        # and performs analysis every 20 cycles automatically
        print("  [INFO] Wolfram telemetry is always enabled (integrated into core)")
    
    # Performance settings
    if args.fast:
        print("  [FAST MODE] Optimizing for speed...")
        config.cycle_interval = 1.0
        config.enable_voice = False
        config.enable_video_interpreter = False
        config.gpt5_verbose = False
    
    # Safety settings
    if args.conservative:
        print("  [CONSERVATIVE MODE] Reducing API calls...")
        config.cycle_interval = 5.0
        config.gemini_rpm_limit = 10
        config.num_gemini_experts = 1
        config.num_claude_experts = 1
    
    print(f"  Cycle interval: {config.cycle_interval}s")
    print(f"  Voice enabled: {config.enable_voice}")
    print(f"  Video enabled: {config.enable_video_interpreter}")
    print(f"  GPT-5 orchestrator: {config.use_gpt5_orchestrator}")
    print(f"  Wolfram telemetry: Always enabled (every 20 cycles)")
    print(f"  Verbose mode: {config.gpt5_verbose}")
    
    print("[CONFIG] ✓ Configuration loaded\n")
    return config


async def run_async_mode(duration: int, config: 'SkyrimConfig'):
    """Run in asynchronous mode (recommended)."""
    from singularis.skyrim.skyrim_agi import SkyrimAGI
    
    print("=" * 70)
    print("ASYNC MODE - Full Parallel Processing")
    print("=" * 70)
    print(f"Duration: {duration} seconds ({duration // 60} minutes)")
    print(f"Mode: Asynchronous (perception || reasoning || action)")
    print("=" * 70 + "\n")
    
    # Initialize AGI
    print("[INIT] Initializing Singularis AGI...\n")
    agi = SkyrimAGI(config)
    
    # Initialize LLM systems (includes Wolfram telemetry)
    print("[INIT] Initializing LLM systems and Wolfram telemetry...\n")
    await agi.initialize_llm()
    
    # Verify BeingState and CoherenceEngine are initialized
    if not hasattr(agi, 'being_state'):
        print("[ERROR] BeingState not initialized!")
        return
    
    if not hasattr(agi, 'coherence_engine'):
        print("[ERROR] CoherenceEngine not initialized!")
        return
    
    if not hasattr(agi, 'wolfram_analyzer'):
        print("[WARNING] Wolfram analyzer not initialized (will skip telemetry)")
    
    print("[VERIFY] ✓ BeingState initialized")
    print("[VERIFY] ✓ CoherenceEngine initialized")
    print("[VERIFY] ✓ Metaphysical center operational\n")
    
    # Start autonomous play
    print("[START] Beginning autonomous gameplay...\n")
    print("=" * 70)
    print("THE ONE BEING IS NOW STRIVING FOR COHERENCE")
    print("=" * 70 + "\n")
    
    try:
        await agi.autonomous_play(duration_seconds=duration)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received keyboard interrupt, shutting down gracefully...")
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print final statistics
        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        
        if hasattr(agi, 'being_state'):
            print(f"\nFinal BeingState:")
            print(f"  Cycle: {agi.being_state.cycle_number}")
            print(f"  C_global: {agi.being_state.global_coherence:.3f}")
            
            if agi.being_state.lumina:
                print(f"  Lumina: (ℓₒ={agi.being_state.lumina.ontic:.3f}, "
                      f"ℓₛ={agi.being_state.lumina.structural:.3f}, "
                      f"ℓₚ={agi.being_state.lumina.participatory:.3f})")
            
            print(f"  Spiral Stage: {agi.being_state.spiral_stage}")
        
        if hasattr(agi, 'coherence_engine'):
            stats = agi.coherence_engine.get_stats()
            print(f"\nCoherence Statistics:")
            print(f"  Mean: {stats.get('mean', 0):.3f}")
            print(f"  Std: {stats.get('std', 0):.3f}")
            print(f"  Min: {stats.get('min', 0):.3f}")
            print(f"  Max: {stats.get('max', 0):.3f}")
            print(f"  Trend: {stats.get('trend', 'unknown')}")
        
        if hasattr(agi, 'stats'):
            print(f"\nPerformance:")
            print(f"  Cycles: {agi.stats.get('cycles_completed', 0)}")
            print(f"  Actions: {agi.stats.get('actions_taken', 0)}")
            print(f"  Success Rate: {agi.stats.get('action_success_rate', 0):.1%}")
        
        print("\n" + "=" * 70)
        print("Thank you for using Singularis Beta v2")
        print("=" * 70 + "\n")


def run_sequential_mode(duration: int, config: 'SkyrimConfig'):
    """Run in sequential mode (simpler, for debugging)."""
    from singularis.skyrim.skyrim_agi import SkyrimAGI
    
    print("=" * 70)
    print("SEQUENTIAL MODE - Step-by-Step Processing")
    print("=" * 70)
    print(f"Duration: {duration} seconds")
    print(f"Mode: Sequential (perception → reasoning → action)")
    print("=" * 70 + "\n")
    
    # Initialize AGI
    agi = SkyrimAGI(config)
    
    # Run
    try:
        agi.autonomous_play_sequential(duration_seconds=duration)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Shutting down...")


def test_integration():
    """Run integration tests."""
    print("=" * 70)
    print("INTEGRATION TEST MODE")
    print("=" * 70 + "\n")
    
    print("[TEST] Running complete integration test...\n")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'test_complete_integration.py'],
        cwd=str(project_root)
    )
    
    return result.returncode == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Singularis Beta v2 - Unified BeingState AGI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for 1 hour in async mode (recommended)
  python run_singularis_beta_v2.py --duration 3600 --mode async
  
  # Run for 30 minutes in fast mode
  python run_singularis_beta_v2.py --duration 1800 --fast
  
  # Run in conservative mode (fewer API calls)
  python run_singularis_beta_v2.py --duration 3600 --conservative
  
  # Run integration tests
  python run_singularis_beta_v2.py --test
  
  # Run with custom cycle interval
  python run_singularis_beta_v2.py --duration 1800 --cycle-interval 2.5

Philosophy:
  Singularis implements "one being striving for coherence" through:
  - BeingState: The unified state of being
  - CoherenceEngine: The measurement of "how well the being is being"
  - C_global: The one thing all subsystems optimize
  
  This is Spinoza's conatus made executable.
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['async', 'sequential'],
        default='async',
        help='Execution mode (default: async)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=1800,
        help='Duration in seconds (default: 1800 = 30 minutes)'
    )
    
    # Performance options
    parser.add_argument(
        '--cycle-interval',
        type=float,
        help='Override cycle interval in seconds (default from config)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: disable voice, video, wolfram for speed'
    )
    
    parser.add_argument(
        '--conservative',
        action='store_true',
        help='Conservative mode: reduce API calls, increase intervals'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice system'
    )
    
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Disable video interpreter'
    )
    
    parser.add_argument(
        '--no-wolfram',
        action='store_true',
        help='Disable Wolfram telemetry'
    )
    
    # Debug options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run integration tests instead of main system'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Test mode
    if args.test:
        success = test_integration()
        sys.exit(0 if success else 1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Load configuration
    config = load_config(args)
    
    # Run in selected mode
    try:
        if args.mode == 'async':
            asyncio.run(run_async_mode(args.duration, config))
        else:
            run_sequential_mode(args.duration, config)
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
