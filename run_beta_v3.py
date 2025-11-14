"""
Singularis Beta v3 Runner

Complete runner for Beta v3 system with all Phase 1-3 improvements.

Features:
- Emergency stabilization (Phase 1)
- ActionArbiter with priority system (Phase 2)
- GPT-5 coordination, conflict prevention, temporal binding (Phase 3)
- BeingState unified state management
- Comprehensive monitoring and logging

Usage:
    python run_beta_v3.py                    # Run with default config
    python run_beta_v3.py --test-mode        # Run in test mode (no game)
    python run_beta_v3.py --no-gpt5          # Disable GPT-5 coordination
    python run_beta_v3.py --verbose          # Verbose logging
    python run_beta_v3.py --duration 3600    # Run for 1 hour
"""

import asyncio
import sys
import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

from singularis.core.being_state import BeingState
from singularis.core.temporal_binding import TemporalCoherenceTracker
from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
from singularis.llm.gpt5_orchestrator import GPT5Orchestrator, SystemType

# Import SkyrimAGI for production mode
try:
    from singularis.skyrim.skyrim_agi import SkyrimAGI, SkyrimConfig
    SKYRIM_AGI_AVAILABLE = True
except ImportError as e:
    SKYRIM_AGI_AVAILABLE = False
    logger.warning(f"SkyrimAGI not available: {e}")

from loguru import logger


@dataclass
class BetaV3Config:
    """Configuration for Beta v3 system."""
    
    # General
    test_mode: bool = False
    duration_seconds: Optional[int] = None
    verbose: bool = False
    
    # GPT-5 Coordination (Enabled with GPT-4.1 Nano)
    enable_gpt5: bool = True  # GPT-5 coordination enabled
    gpt5_model: str = "gpt-4.1-nano-2025-04-14"  # GPT-4.1 Nano for coordination (fastest, most efficient)
    openai_api_key: Optional[str] = None
    
    # Action Arbiter
    enable_conflict_prevention: bool = True
    enable_temporal_tracking: bool = True
    
    # Temporal Binding
    temporal_window_size: int = 20
    temporal_timeout: float = 30.0
    target_closure_rate: float = 0.95
    
    # Monitoring
    stats_interval: int = 60  # Print stats every 60 seconds
    checkpoint_interval: int = 300  # Save checkpoint every 5 minutes


class BetaV3System:
    """Main Beta v3 system with all Phase 1-3 improvements."""
    
    def __init__(self, config: BetaV3Config):
        self.config = config
        self.being_state = BeingState()
        self.temporal_tracker: Optional[TemporalCoherenceTracker] = None
        self.gpt5: Optional[GPT5Orchestrator] = None
        self.arbiter: Optional[ActionArbiter] = None
        self.skyrim_agi: Optional['SkyrimAGI'] = None  # Full SkyrimAGI instance for production
        
        self.running = False
        self.start_time = 0.0
        self.cycle_count = 0
        
        # Statistics
        self.stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'rejected_actions': 0,
            'conflicts_prevented': 0,
            'gpt5_coordinations': 0,
            'temporal_closure_rate': 0.0,
        }
    
    async def initialize(self):
        """Initialize all systems."""
        logger.info("Initializing Beta v3 system...")
        
        # Initialize BeingState
        self.being_state.cycle_number = 0
        self.being_state.global_coherence = 0.75
        self.being_state.session_id = f"beta_v3_{int(time.time())}"
        
        # Initialize Temporal Tracker
        if self.config.enable_temporal_tracking:
            self.temporal_tracker = TemporalCoherenceTracker(
                window_size=self.config.temporal_window_size,
                unclosed_timeout=self.config.temporal_timeout
            )
            await self.temporal_tracker.start()
            logger.info("✓ Temporal tracker initialized")
        
        # Initialize GPT-5 Orchestrator
        if self.config.enable_gpt5:
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.gpt5 = GPT5Orchestrator(
                    api_key=api_key,
                    model=self.config.gpt5_model,
                    verbose=self.config.verbose
                )
                self.gpt5.register_system("action_arbiter", SystemType.ACTION)
                self.gpt5.register_system("being_state", SystemType.CONSCIOUSNESS)
                logger.info("✓ GPT-5 orchestrator initialized")
            else:
                logger.warning("⚠ GPT-5 enabled but no API key found")
                logger.info("   Set OPENAI_API_KEY environment variable or run with --no-gpt5")
                logger.info("   Continuing without GPT-5 coordination...")
                self.config.enable_gpt5 = False
        
        # Initialize SkyrimAGI and ActionArbiter
        if self.config.test_mode:
            # Import controller for test mode
            from singularis.skyrim.controller import VirtualXboxController
            from singularis.skyrim.controller_bindings import SkyrimControllerBindings
            from singularis.skyrim.actions import SkyrimActions
            
            # Create inline mock for test mode (with gamepad)
            class MockSkyrimAGI:
                def __init__(self):
                    self.current_perception = {
                        'game_state': {'health': 100, 'in_combat': False, 'in_menu': False},
                        'scene_type': 'exploration',
                    }
                    self.action_history = []
                    self.actions_executed = []
                    self.being_state = None  # Will be set by BetaV3System
                    
                    # Initialize gamepad for test mode
                    logger.info("[TEST-MODE] Initializing virtual gamepad...")
                    self.controller = VirtualXboxController(dry_run=False)
                    self.bindings = SkyrimControllerBindings(self.controller)
                    self.bindings.switch_to_exploration()
                    self.actions = SkyrimActions(controller=self.controller, dry_run=False)
                    logger.info("[TEST-MODE] ✓ Virtual Xbox 360 controller ready")
                
                async def _execute_action(self, action, scene_type):
                    """Execute action with 1:1 mapping to SkyrimActions."""
                    self.actions_executed.append(action)
                    self.action_history.append(action)
                    
                    # Direct 1:1 execution - action names match SkyrimActions methods
                    if hasattr(self.actions, action):
                        try:
                            logger.info(f"[TEST-MODE] ⚡ Executing: {action}")
                            await getattr(self.actions, action)()
                        except Exception as e:
                            logger.error(f"[TEST-MODE] Action {action} failed: {e}")
                    else:
                        logger.warning(f"[TEST-MODE] Unknown action: {action}")
                        await asyncio.sleep(0.1)
            
            mock_agi = MockSkyrimAGI()
            mock_agi.being_state = self.being_state  # Share BeingState
            
            self.arbiter = ActionArbiter(
                skyrim_agi=mock_agi,
                gpt5_orchestrator=self.gpt5,
                enable_gpt5_coordination=self.config.enable_gpt5
            )
            logger.info("✓ ActionArbiter initialized (test mode with mock AGI)")
        else:
            # Production mode - use full SkyrimAGI
            if not SKYRIM_AGI_AVAILABLE:
                logger.error("❌ SkyrimAGI not available")
                logger.info("   Install required dependencies or run with --test-mode")
                raise RuntimeError("SkyrimAGI not available. Use --test-mode flag.")
            
            logger.info("Initializing full SkyrimAGI system...")
            
            # Create SkyrimConfig with Beta v3 features enabled
            skyrim_config = SkyrimConfig(
                # Enable Beta v3 features
                enable_temporal_binding=self.config.enable_temporal_tracking,
                temporal_window_size=self.config.temporal_window_size,
                temporal_timeout=self.config.temporal_timeout,
                
                # Disable competing loops (Phase 1 - Emergency Stabilization)
                enable_fast_loop=False,  # Disabled to prevent action conflicts
                
                # Enable GPT-5 if configured
                use_gpt5_orchestrator=self.config.enable_gpt5,
                gpt5_verbose=self.config.verbose,
                
                # Reduce cycle interval for better responsiveness
                cycle_interval=3.0,  # 3 seconds per cycle (Phase 1 rate limit fix)
            )
            
            # Initialize SkyrimAGI
            self.skyrim_agi = SkyrimAGI(skyrim_config)
            
            # Use SkyrimAGI's BeingState
            self.being_state = self.skyrim_agi.being_state
            
            # Use SkyrimAGI's temporal tracker if available
            if hasattr(self.skyrim_agi, 'temporal_tracker'):
                self.temporal_tracker = self.skyrim_agi.temporal_tracker
            
            # Create ActionArbiter with real SkyrimAGI
            self.arbiter = ActionArbiter(
                skyrim_agi=self.skyrim_agi,
                gpt5_orchestrator=self.gpt5,
                enable_gpt5_coordination=self.config.enable_gpt5
            )
            
            logger.info("✓ Full SkyrimAGI system initialized")
            logger.info("✓ ActionArbiter integrated with SkyrimAGI")
        
        logger.info("✅ Beta v3 system initialized")
    
    async def run_cycle(self):
        """Run a single system cycle."""
        self.cycle_count += 1
        self.being_state.cycle_number = self.cycle_count
        
        # Update subsystems (mock data for test mode)
        if self.config.test_mode:
            self.being_state.update_subsystem('sensorimotor', {
                'status': 'MOVING' if self.cycle_count % 3 != 0 else 'IDLE',
                'analysis': f'Cycle {self.cycle_count}',
                'visual_similarity': 0.3 + (self.cycle_count % 10) * 0.05
            })
            
            self.being_state.update_subsystem('action_plan', {
                'current': 'explore' if self.cycle_count % 2 == 0 else 'investigate',
                'confidence': 0.7 + (self.cycle_count % 5) * 0.05,
                'reasoning': 'Test mode action planning'
            })
            
            self.being_state.update_subsystem('memory', {
                'pattern_count': 10 + self.cycle_count // 10,
                'similar_situations': [],
                'recommendations': ['move_forward', 'explore']
            })
        
        # Gather candidate actions (using real low-level action names)
        # These map 1:1 with SkyrimActions methods
        import random
        available_actions = [
            # Movement (most common)
            'move_forward', 'move_forward', 'move_forward',  # 3x weight
            'move_backward', 'move_left', 'move_right',
            'turn_left', 'turn_right',
            # Actions
            'jump', 'sprint', 'attack', 'block', 'interact',
            # Exploration
            'look_around', 'look_up', 'look_down',
            # Advanced
            'evasive_maneuver', 'scan_for_targets', 'recenter_camera'
        ]
        
        # Generate 2-3 candidate actions with varying confidence
        num_candidates = random.randint(2, 3)
        candidate_actions = []
        
        for i in range(num_candidates):
            action = random.choice(available_actions)
            candidate_actions.append({
                'action': action,
                'priority': 'NORMAL',
                'source': 'reasoning' if i == 0 else 'action_plan',
                'confidence': 0.65 + random.random() * 0.25  # 0.65-0.90
            })
        
        # GPT-5 coordination (if enabled) - uses hybrid mode
        selected_action = None
        if self.config.enable_gpt5 and self.arbiter and len(candidate_actions) > 1:
            selected_action = await self.arbiter.coordinate_action_decision(
                being_state=self.being_state,
                candidate_actions=candidate_actions
            )
            # Track coordination method (hybrid system)
            if selected_action:
                if selected_action.get('coordination_method') == 'local_fast':
                    self.stats['local_coordinations'] = self.stats.get('local_coordinations', 0) + 1
                else:
                    self.stats['gpt5_coordinations'] += 1
            else:
                # GPT-5 was called but returned no selection
                self.stats['gpt5_coordinations'] += 1
        
        if not selected_action:
            # Fallback: select highest confidence
            selected_action = max(candidate_actions, key=lambda x: x['confidence'])
        
        # Conflict prevention
        if self.config.enable_conflict_prevention and self.arbiter:
            priority = ActionPriority[selected_action['priority']]
            is_allowed, reason = self.arbiter.prevent_conflicting_action(
                action=selected_action['action'],
                being_state=self.being_state,
                priority=priority
            )
            
            if not is_allowed:
                self.stats['conflicts_prevented'] += 1
                self.stats['rejected_actions'] += 1
                logger.warning(f"Action blocked: {reason}")
                return
        
        # Execute action
        if self.arbiter:
            result = await self.arbiter.request_action(
                action=selected_action['action'],
                priority=ActionPriority[selected_action['priority']],
                source=selected_action['source'],
                context={
                    'perception_timestamp': time.time(),
                    'scene_type': 'exploration',
                    'game_state': self.being_state.game_state
                }
            )
            
            self.stats['total_actions'] += 1
            if result.success:
                self.stats['successful_actions'] += 1
        
        # Temporal binding
        if self.temporal_tracker:
            # Bind perception to action
            binding_id = self.temporal_tracker.bind_perception_to_action(
                perception={'cycle': self.cycle_count},
                action=selected_action['action']
            )
            
            # Close loop (simulate outcome)
            await asyncio.sleep(0.1)
            self.temporal_tracker.close_loop(
                binding_id=binding_id,
                outcome='success',
                coherence_delta=0.05,
                success=True
            )
            
            # Update BeingState
            stats = self.temporal_tracker.get_statistics()
            self.being_state.temporal_coherence = 1.0 - stats['unclosed_ratio']
            self.being_state.unclosed_bindings = stats['unclosed_loops']
            self.being_state.stuck_loop_count = stats['stuck_loop_count']
            self.stats['temporal_closure_rate'] = 1.0 - stats['unclosed_ratio']
        
        # Update global coherence
        self.being_state.global_coherence = (
            self.being_state.temporal_coherence * 0.4 +
            0.75 * 0.6  # Base coherence
        )
    
    async def monitoring_loop(self):
        """Background monitoring and stats reporting."""
        last_stats_time = time.time()
        last_checkpoint_time = time.time()
        
        while self.running:
            await asyncio.sleep(1.0)
            
            current_time = time.time()
            
            # Print stats
            if current_time - last_stats_time >= self.config.stats_interval:
                self.print_stats()
                last_stats_time = current_time
            
            # Checkpoint
            if current_time - last_checkpoint_time >= self.config.checkpoint_interval:
                self.save_checkpoint()
                last_checkpoint_time = current_time
    
    def print_stats(self):
        """Print current statistics."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(f"BETA V3 STATISTICS - Cycle {self.cycle_count} ({elapsed:.0f}s)")
        print("="*80)
        
        print(f"\nActions:")
        print(f"  Total: {self.stats['total_actions']}")
        print(f"  Successful: {self.stats['successful_actions']}")
        print(f"  Rejected: {self.stats['rejected_actions']}")
        print(f"  Conflicts Prevented: {self.stats['conflicts_prevented']}")
        
        if self.config.enable_gpt5:
            local_count = self.stats.get('local_coordinations', 0)
            gpt5_count = self.stats['gpt5_coordinations']
            total_coord = local_count + gpt5_count
            
            print(f"\nHybrid Coordination (Speed Optimized):")
            if total_coord > 0:
                print(f"  Total decisions: {total_coord}")
                print(f"  ⚡ Fast local: {local_count} ({local_count/total_coord*100:.1f}%)")
                print(f"  🧠 GPT-5 Mini: {gpt5_count} ({gpt5_count/total_coord*100:.1f}%)")
                speed_improvement = (local_count / total_coord) * 100
                print(f"  Speed improvement: ~{speed_improvement:.0f}% instant")
            else:
                print(f"  No coordinations yet")
        
        if self.temporal_tracker:
            print(f"\nTemporal Binding:")
            print(f"  Closure Rate: {self.stats['temporal_closure_rate']:.1%}")
            print(f"  Unclosed Bindings: {self.being_state.unclosed_bindings}")
            print(f"  Stuck Loop Count: {self.being_state.stuck_loop_count}")
        
        print(f"\nBeingState:")
        print(f"  Global Coherence: {self.being_state.global_coherence:.3f}")
        print(f"  Temporal Coherence: {self.being_state.temporal_coherence:.3f}")
        
        if self.arbiter:
            arbiter_stats = self.arbiter.get_stats()
            print(f"\nArbiter:")
            print(f"  Rejection Rate: {arbiter_stats['rejection_rate']:.1%}")
            if arbiter_stats.get('override_rate'):
                print(f"  Override Rate: {arbiter_stats['override_rate']:.1%}")
        
        print("="*80 + "\n")
    
    def save_checkpoint(self):
        """Save system checkpoint."""
        checkpoint_file = project_root / f"checkpoints/beta_v3_{self.being_state.session_id}.json"
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        import json
        checkpoint = {
            'cycle': self.cycle_count,
            'elapsed': time.time() - self.start_time,
            'stats': self.stats,
            'being_state': self.being_state.export_snapshot()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file.name}")
    
    async def run(self):
        """Main run loop."""
        self.running = True
        self.start_time = time.time()
        
        logger.info("Starting Beta v3 system...")
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.monitoring_loop())
        
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(0.5)  # Cycle interval
                
                # Check duration limit
                if self.config.duration_seconds:
                    elapsed = time.time() - self.start_time
                    if elapsed >= self.config.duration_seconds:
                        logger.info(f"Duration limit reached ({self.config.duration_seconds}s)")
                        break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.running = False
            monitor_task.cancel()
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all systems."""
        logger.info("Shutting down Beta v3 system...")
        
        # Print final stats
        self.print_stats()
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Close temporal tracker
        if self.temporal_tracker:
            await self.temporal_tracker.close()
        
        # Close GPT-5
        if self.gpt5:
            await self.gpt5.close()
        
        logger.info("✅ Beta v3 system shutdown complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Singularis Beta v3 Runner")
    parser.add_argument("--test-mode", action="store_true", default=False, help="Run in test mode")
    parser.add_argument("--production", action="store_true", default=True, help="Run in production mode with full SkyrimAGI (default)")
    parser.add_argument("--no-gpt5", action="store_true", help="Disable GPT-5 coordination")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--duration", type=int, help="Run duration in seconds")
    parser.add_argument("--stats-interval", type=int, default=60, help="Stats print interval")
    
    args = parser.parse_args()
    
    # If --production is specified, disable test mode
    if args.production:
        args.test_mode = False
    
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/beta_v3_{time}.log", rotation="1 day", level="DEBUG")
    
    # Create config
    config = BetaV3Config(
        test_mode=args.test_mode,
        duration_seconds=args.duration,
        verbose=args.verbose,
        enable_gpt5=not args.no_gpt5,
        stats_interval=args.stats_interval
    )
    
    # Create and run system
    system = BetaV3System(config)
    await system.initialize()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
