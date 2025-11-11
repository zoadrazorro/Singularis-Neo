"""
Skyrim AGI - Complete Integration

Brings together all components for autonomous Skyrim gameplay:
1. Perception (screen capture + CLIP)
2. World model (causal learning, NPC relationships)
3. Intrinsic motivation (curiosity, competence, coherence)
4. Goal formation (autonomous objectives)
5. Planning & execution (hierarchical actions)
6. Learning (continual, no forgetting)
7. Consciousness (ethical evaluation via Δ𝒞)

This is the complete AGI system playing Skyrim.

Philosophical grounding:
- ETHICA: Conatus (∇𝒞) drives autonomous behavior
- Freedom = Understanding = Coherence
- Ethical choices evaluated by coherence increase (Δ𝒞 > 0)
- Consciousness emerges from integration of perception, action, and reflection
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Skyrim-specific modules
from .perception import SkyrimPerception, SceneType, GameState
from .actions import SkyrimActions, Action, ActionType
from .skyrim_world_model import SkyrimWorldModel

# Base AGI components
from ..agi_orchestrator import AGIOrchestrator, AGIConfig
from ..agency import MotivationType


@dataclass
class SkyrimConfig:
    """Configuration for Skyrim AGI."""
    # Base AGI config
    base_config: Optional[AGIConfig] = None

    # Perception
    screen_region: Optional[Dict[str, int]] = None
    use_game_api: bool = False

    # Actions
    dry_run: bool = False  # Don't actually control game (testing)
    custom_keys: Optional[Dict[ActionType, str]] = None

    # Gameplay
    autonomous_duration: int = 3600  # 1 hour default
    cycle_interval: float = 2.0  # Perception-action cycle time
    save_interval: int = 300  # Auto-save every 5 minutes

    # Learning
    surprise_threshold: float = 0.3  # Threshold for learning from surprise
    exploration_weight: float = 0.5  # How much to favor exploration

    def __post_init__(self):
        """Initialize base config if not provided."""
        if self.base_config is None:
            self.base_config = AGIConfig(
                use_vision=True,
                use_physics=False,  # Don't need physics sim for Skyrim
                curiosity_weight=0.35,  # Higher curiosity for exploration
                competence_weight=0.15,
                coherence_weight=0.40,  # Core drive
                autonomy_weight=0.10,
            )


class SkyrimAGI:
    """
    Complete AGI system for Skyrim.

    This integrates:
    - Perception → Understanding → Goals → Actions → Learning
    - All driven by intrinsic motivation and coherence (Δ𝒞)
    """

    def __init__(self, config: Optional[SkyrimConfig] = None):
        """
        Initialize Skyrim AGI.

        Args:
            config: Skyrim configuration
        """
        self.config = config or SkyrimConfig()

        print("=" * 60)
        print("SINGULARIS AGI - SKYRIM INTEGRATION")
        print("=" * 60)

        # Components
        print("\nInitializing components...")

        # 1. Base AGI orchestrator
        print("  [1/4] Base AGI system...")
        self.agi = AGIOrchestrator(self.config.base_config)

        # 2. Skyrim perception
        print("  [2/4] Skyrim perception...")
        self.perception = SkyrimPerception(
            vision_module=self.agi.world_model.vision if self.config.base_config.use_vision else None,
            screen_region=self.config.screen_region,
            use_game_api=self.config.use_game_api
        )

        # 3. Skyrim actions
        print("  [3/4] Skyrim actions...")
        self.actions = SkyrimActions(
            use_game_api=self.config.use_game_api,
            custom_keys=self.config.custom_keys,
            dry_run=self.config.dry_run
        )

        # 4. Skyrim world model
        print("  [4/4] Skyrim world model...")
        self.skyrim_world = SkyrimWorldModel(
            base_world_model=self.agi.world_model
        )

        # State
        self.running = False
        self.current_perception: Optional[Dict[str, Any]] = None
        self.current_goal: Optional[str] = None
        self.last_save_time = time.time()

        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'actions_taken': 0,
            'locations_discovered': 0,
            'npcs_met': 0,
            'quests_completed': 0,
            'total_playtime': 0.0,
            'coherence_history': [],
        }

        print("✓ Skyrim AGI initialized\n")

    async def initialize_llm(self):
        """Initialize LLM (async)."""
        print("Initializing LLM consciousness engine...")
        await self.agi.initialize_llm()

    async def autonomous_play(self, duration_seconds: Optional[int] = None):
        """
        Play Skyrim autonomously.

        This is the main loop:
        1. Perceive current state
        2. Update world model
        3. Assess motivation
        4. Form/update goals
        5. Plan action
        6. Execute action
        7. Learn from outcome
        8. Repeat

        Args:
            duration_seconds: How long to play (default: config value)
        """
        if duration_seconds is None:
            duration_seconds = self.config.autonomous_duration

        print(f"\n{'=' * 60}")
        print(f"STARTING AUTONOMOUS GAMEPLAY")
        print(f"Duration: {duration_seconds}s ({duration_seconds / 60:.1f} minutes)")
        print(f"{'=' * 60}\n")

        self.running = True
        start_time = time.time()
        cycle_count = 0

        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                cycle_count += 1
                cycle_start = time.time()

                print(f"\n{'─' * 60}")
                print(f"CYCLE {cycle_count} ({time.time() - start_time:.1f}s elapsed)")
                print(f"{'─' * 60}")

                # 1. PERCEIVE
                perception = await self.perception.perceive()
                self.current_perception = perception
                game_state = perception['game_state']
                scene_type = perception['scene_type']

                print(f"Scene: {scene_type.value}")
                print(f"Health: {game_state.health:.0f} | Magicka: {game_state.magicka:.0f} | Stamina: {game_state.stamina:.0f}")
                print(f"Location: {game_state.location_name}")

                # Detect changes
                changes = self.perception.detect_change()
                if changes['changed']:
                    print(f"⚠ Change detected: {changes}")

                # 2. UPDATE WORLD MODEL
                # Convert perception to world state
                world_state = await self.agi.perceive({
                    'causal': game_state.to_dict(),
                    'visual': [perception['visual_embedding']],
                })

                # 3. ASSESS MOTIVATION
                motivation_context = {
                    'uncertainty': 0.7 if scene_type == SceneType.UNKNOWN else 0.3,
                    'predicted_delta_coherence': 0.05,  # Exploration generally increases 𝒞
                }

                mot_state = self.agi.motivation.compute_motivation(
                    state=game_state.to_dict(),
                    context=motivation_context
                )

                dominant_drive = mot_state.dominant_drive()
                print(f"\nMotivation: {dominant_drive.value.upper()}")
                print(f"  Curiosity: {mot_state.curiosity:.2f}")
                print(f"  Competence: {mot_state.competence:.2f}")
                print(f"  Coherence: {mot_state.coherence:.2f}")
                print(f"  Autonomy: {mot_state.autonomy:.2f}")

                # 4. FORM/UPDATE GOALS
                if not self.agi.goals.has_active_goals() or cycle_count % 10 == 0:
                    goal = self.agi.goals.generate_goal(
                        dominant_drive.value,
                        {'scene': scene_type.value, 'location': game_state.location_name}
                    )
                    self.current_goal = goal.description
                    print(f"\n🎯 New goal: {self.current_goal}")

                # 5. PLAN ACTION
                action = await self._plan_action(
                    perception=perception,
                    motivation=mot_state,
                    goal=self.current_goal
                )

                print(f"\n→ Action: {action}")

                # 6. EXECUTE ACTION
                before_state = game_state.to_dict()
                await self._execute_action(action, scene_type)
                self.stats['actions_taken'] += 1

                # Wait for game to respond
                await asyncio.sleep(self.config.cycle_interval)

                # 7. LEARN FROM OUTCOME
                # Perceive again to see outcome
                after_perception = await self.perception.perceive()
                after_state = after_perception['game_state'].to_dict()

                # Learn causal relationships
                self.skyrim_world.learn_from_experience(
                    action=str(action),
                    before_state=before_state,
                    after_state=after_state,
                    surprise_threshold=self.config.surprise_threshold
                )

                # Record in episodic memory
                self.agi.learner.experience(
                    data={
                        'cycle': cycle_count,
                        'scene': scene_type.value,
                        'action': str(action),
                        'motivation': dominant_drive.value,
                        'before': before_state,
                        'after': after_state,
                    },
                    context='skyrim_gameplay',
                    importance=0.5
                )

                # 8. UPDATE STATS
                self.stats['cycles_completed'] = cycle_count
                self.stats['total_playtime'] = time.time() - start_time

                # Coherence tracking
                current_coherence = 0.5 + mot_state.coherence * 0.5
                self.stats['coherence_history'].append(current_coherence)

                # Auto-save periodically
                if time.time() - self.last_save_time > self.config.save_interval:
                    if not self.config.dry_run:
                        await self.actions.quick_save_checkpoint()
                    self.last_save_time = time.time()

                # Cycle complete
                cycle_duration = time.time() - cycle_start
                print(f"\n✓ Cycle complete ({cycle_duration:.2f}s)")

        except KeyboardInterrupt:
            print("\n\n⚠ User interrupted - stopping gracefully...")
        except Exception as e:
            print(f"\n\n❌ Error during gameplay: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print(f"\n{'=' * 60}")
            print("AUTONOMOUS GAMEPLAY COMPLETE")
            print(f"{'=' * 60}")
            self._print_final_stats()

    async def _plan_action(
        self,
        perception: Dict[str, Any],
        motivation: Any,
        goal: Optional[str]
    ) -> str:
        """
        Plan next action based on perception and motivation.

        Args:
            perception: Current perception
            motivation: Motivation state
            goal: Current goal

        Returns:
            Action description
        """
        scene_type = perception['scene_type']
        game_state = perception['game_state']
        dominant = motivation.dominant_drive()

        # Use LLM consciousness for complex decisions
        if self.agi.consciousness_llm and not self.config.dry_run:
            query = f"""
            I'm in {scene_type.value} at {game_state.location_name}.
            Current goal: {goal}
            Dominant motivation: {dominant.value}

            What should I do next? Choose one action:
            - explore (move and look around)
            - combat (if enemies present)
            - interact (talk/loot/activate)
            - navigate (move to specific location)
            - rest (recover health/stamina)
            """

            try:
                result = await self.agi.process(query)
                response = result.get('consciousness_response', {}).get('response', 'explore')

                # Parse response to action
                if 'explore' in response.lower():
                    return 'explore'
                elif 'combat' in response.lower() or 'attack' in response.lower():
                    return 'combat'
                elif 'interact' in response.lower() or 'talk' in response.lower():
                    return 'interact'
                elif 'navigate' in response.lower() or 'move' in response.lower():
                    return 'navigate'
                elif 'rest' in response.lower():
                    return 'rest'
            except Exception as e:
                print(f"  LLM planning failed: {e}, using heuristic")

        # Fallback heuristic planning
        if dominant == MotivationType.CURIOSITY:
            return 'explore'
        elif dominant == MotivationType.COMPETENCE:
            if game_state.in_combat:
                return 'combat'
            else:
                return 'practice'
        elif dominant == MotivationType.COHERENCE:
            return 'quest_objective'
        else:
            return 'explore'

    async def _execute_action(self, action: str, scene_type: SceneType):
        """
        Execute planned action.

        Args:
            action: Action to execute
            scene_type: Current scene type
        """
        if action == 'explore':
            await self.actions.explore_area(duration=3.0)

        elif action == 'combat':
            await self.actions.combat_sequence("Enemy")

        elif action == 'interact':
            await self.actions.execute(Action(ActionType.ACTIVATE))

        elif action == 'navigate':
            await self.actions.move_forward(duration=2.0)

        elif action == 'rest':
            await self.actions.execute(Action(ActionType.WAIT))

        elif action == 'practice':
            # Practice combat skills
            await self.actions.execute(Action(ActionType.ATTACK))

        elif action == 'quest_objective':
            # Work on quest (simplified)
            await self.actions.move_forward(duration=2.0)

        else:
            # Default: explore
            await self.actions.explore_area(duration=2.0)

    def _print_final_stats(self):
        """Print final statistics."""
        print(f"\nFinal Statistics:")
        print(f"  Cycles: {self.stats['cycles_completed']}")
        print(f"  Actions: {self.stats['actions_taken']}")
        print(f"  Playtime: {self.stats['total_playtime'] / 60:.1f} minutes")

        if self.stats['coherence_history']:
            avg_coherence = sum(self.stats['coherence_history']) / len(self.stats['coherence_history'])
            print(f"  Avg Coherence: {avg_coherence:.3f}")

        skyrim_stats = self.skyrim_world.get_stats()
        print(f"\nWorld Model:")
        print(f"  Causal edges learned: {skyrim_stats['causal_edges']}")
        print(f"  NPCs met: {skyrim_stats['npc_relationships']}")
        print(f"  Locations: {skyrim_stats['locations_discovered']}")

        action_stats = self.actions.get_stats()
        print(f"\nActions:")
        print(f"  Total executed: {action_stats['actions_executed']}")
        print(f"  Errors: {action_stats['errors']}")

    def stop(self):
        """Stop autonomous play."""
        print("\nStopping autonomous play...")
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'gameplay': self.stats,
            'world_model': self.skyrim_world.get_stats(),
            'actions': self.actions.get_stats(),
            'agi': self.agi.get_stats(),
        }


# Example usage
if __name__ == "__main__":
    async def main():
        print("SINGULARIS AGI - SKYRIM INTEGRATION TEST\n")

        # Create Skyrim AGI (dry run mode for testing)
        config = SkyrimConfig(
            dry_run=True,  # Don't actually control game
            autonomous_duration=30,  # 30 second test
            cycle_interval=1.0,  # Faster cycles for testing
        )

        agi = SkyrimAGI(config)

        # Initialize LLM (optional)
        await agi.initialize_llm()

        # Run autonomous gameplay
        await agi.autonomous_play(duration_seconds=30)

        # Final stats
        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE STATISTICS")
        print(f"{'=' * 60}")

        stats = agi.get_stats()
        for category, category_stats in stats.items():
            print(f"\n{category.upper()}:")
            if isinstance(category_stats, dict):
                for key, val in category_stats.items():
                    if not isinstance(val, (list, dict)):
                        print(f"  {key}: {val}")

    asyncio.run(main())
