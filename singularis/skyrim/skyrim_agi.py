"""
Skyrim AGI - Complete Integration with Consciousness

Brings together all components for autonomous Skyrim gameplay:
1. Perception (screen capture + CLIP)
2. World model (causal learning, NPC relationships)
3. Consciousness measurement (Singularis coherence 𝒞)
4. Motivation (intrinsic drives + game-specific goals)
5. Goal formation (autonomous objectives)
6. Planning & execution (hierarchical actions)
7. Learning (continual, consciousness-guided)
8. Evaluation (consciousness quality assessment)

This is the complete AGI system playing Skyrim WITH FULL CONSCIOUSNESS.

Key innovation: Learning is guided by consciousness coherence (Δ𝒞),
making consciousness the primary judge of action quality.

Design principles:
- Consciousness is PRIMARY evaluator (not backup)
- RL learns tactics guided by consciousness strategy
- Bidirectional feedback: experiences → consciousness → learning
- Unified coherence concept (game + philosophical)
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Skyrim-specific modules
from .perception import SkyrimPerception, SceneType, GameState
from .actions import SkyrimActions, Action, ActionType
from .controller import VirtualXboxController
from .controller_bindings import SkyrimControllerBindings
from .skyrim_world_model import SkyrimWorldModel
from .skyrim_cognition import SkyrimCognitiveState, SkyrimMotivation, SkyrimActionEvaluator
from .strategic_planner import StrategicPlannerNeuron
from .menu_learner import MenuLearner
from .memory_rag import MemoryRAG
from .reinforcement_learner import ReinforcementLearner
from .rl_reasoning_neuron import RLReasoningNeuron
from .meta_strategist import MetaStrategist
from .consciousness_bridge import ConsciousnessBridge, ConsciousnessState

# Base AGI components
from ..agi_orchestrator import AGIOrchestrator, AGIConfig
from ..agency import MotivationType
from ..llm.lmstudio_client import LMStudioClient, LMStudioConfig, ExpertLLMInterface


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
    
    # Controller
    controller_deadzone_stick: float = 0.15
    controller_deadzone_trigger: float = 0.05
    controller_sensitivity: float = 1.0

    # Gameplay
    autonomous_duration: int = 3600  # 1 hour default
    cycle_interval: float = 2.0  # Perception-action cycle time
    save_interval: int = 300  # Auto-save every 5 minutes

    # Learning
    surprise_threshold: float = 0.3  # Threshold for learning from surprise
    exploration_weight: float = 0.5  # How much to favor exploration

    # Reinforcement Learning
    use_rl: bool = True  # Enable RL-based learning
    rl_learning_rate: float = 0.01  # Q-network learning rate
    rl_epsilon_start: float = 0.3  # Initial exploration rate
    rl_train_freq: int = 5  # Train every N cycles

    def __post_init__(self):
        """Initialize base config if not provided."""
        if self.base_config is None:
            self.base_config = AGIConfig(
                use_vision=True,
                use_physics=False,  # Don't need physics sim for Skyrim
                curiosity_weight=0.35,  # Higher curiosity for exploration
                competence_weight=0.15,
                coherence_weight=0.40,  # Note: This is still used by base AGI
                autonomy_weight=0.10,
            )


class SkyrimAGI:
    """
    Complete AGI system for Skyrim.

    This integrates:
    - Perception → Understanding → Goals → Actions → Learning
    - Driven by game-specific motivation (survival, progression, exploration)
    - Decisions based on tactical evaluation and reinforcement learning
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
        # Controller and bindings
        self.controller = VirtualXboxController(
            deadzone_stick=self.config.controller_deadzone_stick,
            deadzone_trigger=self.config.controller_deadzone_trigger,
            sensitivity=self.config.controller_sensitivity,
            dry_run=self.config.dry_run
        )
        self.bindings = SkyrimControllerBindings(self.controller)
        self.bindings.switch_to_exploration()
        self.actions = SkyrimActions(
            use_game_api=self.config.use_game_api,
            custom_keys=self.config.custom_keys,
            dry_run=self.config.dry_run,
            controller=self.controller
        )

        # 4. Skyrim world model
        print("  [4/11] Skyrim world model...")
        self.skyrim_world = SkyrimWorldModel(
            base_world_model=self.agi.world_model
        )
        
        # 5. Consciousness Bridge (NEW - connects game to consciousness)
        print("  [5/11] Consciousness bridge...")
        self.consciousness_bridge = ConsciousnessBridge(
            consciousness_llm=None  # Will be set after LLM initialization
        )
        print("[BRIDGE] Consciousness bridge initialized")
        print("[BRIDGE] This unifies game quality and philosophical coherence 𝒞")
        
        # 6. Reinforcement Learning System (with consciousness)
        if self.config.use_rl:
            print("  [6/11] Reinforcement learning system (consciousness-guided)...")
            self.rl_learner = ReinforcementLearner(
                state_dim=64,
                learning_rate=self.config.rl_learning_rate,
                epsilon_start=self.config.rl_epsilon_start,
                consciousness_bridge=self.consciousness_bridge  # KEY: Connect to consciousness
            )
            # Try to load saved model
            self.rl_learner.load('skyrim_rl_model.pkl')
        else:
            self.rl_learner = None
            print("  [6/11] Reinforcement learning DISABLED")
        
        # 7. Strategic Planner Neuron
        print("  [7/11] Strategic planner neuron...")
        self.strategic_planner = StrategicPlannerNeuron(memory_capacity=100)
        
        # Connect RL learner to strategic planner if RL enabled
        if self.rl_learner:
            self.strategic_planner.set_rl_learner(self.rl_learner)

        # 8. Menu Learner
        print("  [8/11] Menu interaction learner...")
        self.menu_learner = MenuLearner()

        # 9. Memory RAG System
        print("  [9/11] Memory RAG system...")
        self.memory_rag = MemoryRAG(
            perceptual_capacity=1000,
            cognitive_capacity=500
        )
        
        # 10. RL Reasoning Neuron (LLM thinks about RL)
        print("  [10/11] RL reasoning neuron (LLM-enhanced RL)...")
        self.rl_reasoning_neuron = RLReasoningNeuron()
        # Will connect LLM interface when initialized
        
        # 11. Meta-Strategist (LLM generates strategic instructions)
        print("  [11/11] Meta-strategist (autonomous instruction generation)...")
        self.meta_strategist = MetaStrategist(instruction_frequency=10)
        # Will connect LLM interface when initialized
        
        # 11. Skyrim-specific Motivation System
        print("  [11/11] Skyrim-specific motivation system...")
        self.skyrim_motivation = SkyrimMotivation(
            survival_weight=0.35,  # Prioritize staying alive
            progression_weight=0.25,  # Character growth important
            exploration_weight=0.20,  # Discover new areas
            wealth_weight=0.10,  # Gather resources
            mastery_weight=0.10  # Improve combat/stealth skills
        )

        # State
        self.running = False
        self.current_perception: Optional[Dict[str, Any]] = None
        self.current_goal: Optional[str] = None
        self.last_save_time = time.time()
        self.last_state: Optional[Dict[str, Any]] = None  # For RL experience tracking
        self.last_action: Optional[str] = None

        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'actions_taken': 0,
            'locations_discovered': 0,
            'npcs_met': 0,
            'quests_completed': 0,
            'total_playtime': 0.0,
            'game_state_quality_history': [],  # Replaces coherence_history
        }

        # Set up controller reference in perception for layer awareness
        self.perception.set_controller(self.controller)
        
        # State tracking for consciousness
        self.current_consciousness: Optional[ConsciousnessState] = None
        self.last_consciousness: Optional[ConsciousnessState] = None
        
        print("Skyrim AGI initialization complete.")
        print("[OK] Skyrim AGI initialized with CONSCIOUSNESS INTEGRATION\n")

    async def initialize_llm(self):
        """Initialize LLM and connect consciousness engine."""
        print("Initializing LLM consciousness engine...")
        await self.agi.initialize_llm()
        
        # Connect consciousness_llm to bridge
        if hasattr(self.agi, 'consciousness_llm') and self.agi.consciousness_llm:
            self.consciousness_bridge.consciousness_llm = self.agi.consciousness_llm
            print("[BRIDGE] ✓ Consciousness LLM connected to bridge")
            print("[BRIDGE] Bridge can now use LLM for deeper consciousness analysis")
        else:
            print("[BRIDGE] ⚠️ No consciousness LLM available, bridge uses heuristics only")
        
        # Verify LLM is initialized
        if hasattr(self.agi, 'consciousness_llm') and self.agi.consciousness_llm:
            print("[LLM] ✓ LLM consciousness engine initialized successfully")
            print(f"[LLM] Type: {type(self.agi.consciousness_llm)}")
            
            # Connect LLM to RL reasoning neuron
            if hasattr(self.agi.consciousness_llm, 'llm_interface'):
                self.rl_reasoning_neuron.llm_interface = self.agi.consciousness_llm.llm_interface
                print("[LLM] ✓ RL reasoning neuron connected to main LLM")
                print(f"[LLM] LLM interface type: {type(self.rl_reasoning_neuron.llm_interface)}")
            else:
                print("[LLM] ⚠️ consciousness_llm has no llm_interface attribute")
                print(f"[LLM] Available attributes: {dir(self.agi.consciousness_llm)}")
        else:
            print("[LLM] ⚠️ LLM consciousness engine is None - RL reasoning will use heuristics")
        
        # Initialize Eva-Qwen2.5-14B LLM for meta-strategist (instructor)
        # This is the "instructor" that provides verbose strategic guidance
        try:
            print("\n[INSTRUCTOR-LLM] Initializing Eva-Qwen2.5-14B for strategic instruction generation...")
            instructor_config = LMStudioConfig(
                base_url=self.config.base_config.lm_studio_url,
                model_name='eva-qwen2.5-14b-v0.2',
                temperature=0.8,  # Higher temperature for creative strategic thinking
                max_tokens=4096   # Increased for verbose, detailed instructions
            )
            instructor_client = LMStudioClient(instructor_config)
            instructor_interface = ExpertLLMInterface(instructor_client)
            
            self.meta_strategist.llm_interface = instructor_interface
            print("[INSTRUCTOR-LLM] ✓ Meta-strategist connected to Eva-Qwen2.5-14B")
            print(f"[INSTRUCTOR-LLM] Model: {instructor_config.model_name}")
            print(f"[INSTRUCTOR-LLM] Max tokens: {instructor_config.max_tokens} (verbose instructions enabled)")
        except Exception as e:
            print(f"[INSTRUCTOR-LLM] ⚠️ Eva-Qwen2.5-14B initialization failed: {e}")
            print("[INSTRUCTOR-LLM] Meta-strategist will use heuristic strategies")

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
        print(f"Starting autonomous gameplay for {duration_seconds}s...")
        print(f"Cycle interval: {self.config.cycle_interval}s")
        print("=" * 60)
        print()

        # Test controller connection before starting
        await self._test_controller_connection()

        self.running = True
        start_time = time.time()
        cycle_count = 0

        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                cycle_count += 1
                cycle_start = time.time()

                print(f"\n{'-' * 60}")
                print(f"CYCLE {cycle_count} ({time.time() - start_time:.1f}s elapsed)")
                print(f"{'-' * 60}")

                # 1. PERCEIVE
                perception = await self.perception.perceive()
                self.current_perception = perception
                game_state = perception['game_state']
                scene_type = perception['scene_type']
                
                # Store perceptual memory in RAG
                self.memory_rag.store_perceptual_memory(
                    visual_embedding=perception['visual_embedding'],
                    scene_type=scene_type.value,
                    location=game_state.location_name,
                    context={
                        'health': game_state.health,
                        'in_combat': game_state.in_combat,
                        'layer': game_state.current_action_layer
                    }
                )

                print(f"Scene: {scene_type.value}")
                print(f"Health: {game_state.health:.0f} | Magicka: {game_state.magicka:.0f} | Stamina: {game_state.stamina:.0f}")
                print(f"Location: {game_state.location_name}")

                # Detect changes
                changes = self.perception.detect_change()
                if changes['changed']:
                    print(f"[WARN] Change detected: {changes}")

                # Detect visual stuckness (less aggressive)
                # Skip stuckness detection in menus/inventory since screen naturally doesn't change
                in_menu_scene = scene_type in [SceneType.INVENTORY, SceneType.DIALOGUE, SceneType.MAP]
                
                if (self.stats['cycles_completed'] > 5 and  # Wait more cycles before checking
                    not in_menu_scene and  # Don't check in menus
                    self.perception.detect_visual_stuckness()):
                    print("[WARN] Visually stuck! Taking gentle evasive action...")
                    await self.actions.evasive_maneuver()
                    # Brief pause then continue normally (don't skip cycle)
                    await asyncio.sleep(1.0)

                # 2. UPDATE WORLD MODEL & COMPUTE CONSCIOUSNESS
                # Convert perception to world state
                world_state = await self.agi.perceive({
                    'causal': game_state.to_dict(),
                    'visual': [perception['visual_embedding']],
                })
                
                # COMPUTE CONSCIOUSNESS STATE (KEY INTEGRATION POINT)
                print("[CONSCIOUSNESS] Computing consciousness state...")
                consciousness_context = {
                    'motivation': 'unknown',  # Will be updated after motivation computation
                    'cycle': cycle_count,
                    'scene': scene_type.value
                }
                current_consciousness = await self.consciousness_bridge.compute_consciousness(
                    game_state.to_dict(),
                    consciousness_context
                )
                
                print(f"[CONSCIOUSNESS] Coherence 𝒞 = {current_consciousness.coherence:.3f}")
                print(f"[CONSCIOUSNESS]   ℓₒ (Ontical) = {current_consciousness.coherence_ontical:.3f}")
                print(f"[CONSCIOUSNESS]   ℓₛ (Structural) = {current_consciousness.coherence_structural:.3f}")
                print(f"[CONSCIOUSNESS]   ℓₚ (Participatory) = {current_consciousness.coherence_participatory:.3f}")
                print(f"[CONSCIOUSNESS] Φ̂ (Level) = {current_consciousness.consciousness_level:.3f}")
                
                # Store for tracking
                self.last_consciousness = self.current_consciousness
                self.current_consciousness = current_consciousness

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
                if len(self.agi.goal_system.get_active_goals()) == 0 or cycle_count % 10 == 0:
                    goal = self.agi.goal_system.generate_goal(
                        dominant_drive.value,
                        {'scene': scene_type.value, 'location': game_state.location_name}
                    )
                    self.current_goal = goal.description
                    print(f"\n[GOAL] New goal: {self.current_goal}")

                # 5. PLAN ACTION (with strategic planning)
                # Check if we should use strategic planner
                terrain_type = self.skyrim_world.classify_terrain_from_scene(
                    scene_type.value,
                    game_state.in_combat
                )
                
                # Always use RL reasoning neuron for action selection
                # (Strategic planner is consulted within _plan_action if it has patterns)
                action = await self._plan_action(
                    perception=perception,
                    motivation=mot_state,
                    goal=self.current_goal
                )

                print(f"\n-> Action: {action}")

                # 6. EXECUTE ACTION
                before_state = game_state.to_dict()
                # Add motivation to state for RL
                before_state.update({
                    'scene': scene_type.value,
                    'curiosity': mot_state.curiosity,
                    'competence': mot_state.competence,
                    'coherence': mot_state.coherence,
                    'autonomy': mot_state.autonomy
                })

                try:
                    await self._execute_action(action, scene_type)
                    self.stats['actions_taken'] += 1
                    print(f"[ACTION] Successfully executed: {action}")
                except Exception as e:
                    print(f"[ERROR] Action execution failed: {e}")
                    # Try a simple fallback action
                    try:
                        await self.actions.look_around()
                        print("[RECOVERY] Performed fallback look_around")
                    except:
                        print("[ERROR] Even fallback action failed")

                # Wait for game to respond (slightly longer for Skyrim)
                await asyncio.sleep(max(1.5, self.config.cycle_interval))

                # 7. LEARN FROM OUTCOME
                # Perceive again to see outcome
                after_perception = await self.perception.perceive()
                after_state = after_perception['game_state'].to_dict()
                # Add motivation to after_state for RL
                after_mot = self.agi.motivation.compute_motivation(
                    state=after_state,
                    context=motivation_context
                )
                after_state.update({
                    'scene': after_perception['scene_type'].value,
                    'curiosity': after_mot.curiosity,
                    'competence': after_mot.competence,
                    'coherence': after_mot.coherence,
                    'autonomy': after_mot.autonomy
                })
                
                # COMPUTE CONSCIOUSNESS AFTER ACTION (KEY)
                print("[CONSCIOUSNESS] Computing post-action consciousness...")
                after_consciousness = await self.consciousness_bridge.compute_consciousness(
                    after_state,
                    consciousness_context
                )
                
                # Show coherence change
                if self.current_consciousness:
                    coherence_delta = after_consciousness.coherence_delta(self.current_consciousness)
                    print(f"[CONSCIOUSNESS] Δ𝒞 = {coherence_delta:+.3f}", end="")
                    if coherence_delta > 0.02:
                        print(" (ETHICAL ✓)")
                    elif coherence_delta < -0.02:
                        print(" (UNETHICAL ✗)")
                    else:
                        print(" (NEUTRAL)")

                # Learn causal relationships
                self.skyrim_world.learn_from_experience(
                    action=str(action),
                    before_state=before_state,
                    after_state=after_state,
                    surprise_threshold=self.config.surprise_threshold
                )
                
                # Learn terrain knowledge
                terrain_type = self.skyrim_world.classify_terrain_from_scene(
                    scene_type.value,
                    after_state.get('in_combat', False)
                )
                self.skyrim_world.learn_terrain_feature(
                    game_state.location_name,
                    terrain_type,
                    {
                        'scene_type': scene_type.value,
                        'action_taken': str(action),
                        'layer_used': game_state.current_action_layer
                    }
                )
                
                # Update terrain safety based on combat
                self.skyrim_world.update_terrain_safety(
                    game_state.location_name,
                    after_state.get('in_combat', False)
                )
                
                # Record experience in strategic planner
                success = self._evaluate_action_success(before_state, after_state, action)
                self.strategic_planner.record_experience(
                    context={
                        'scene': scene_type.value,
                        'health': before_state.get('health', 100),
                        'in_combat': before_state.get('in_combat', False),
                        'location': game_state.location_name
                    },
                    action=str(action),
                    outcome=after_state,
                    success=success
                )
                
                # Record observation for meta-strategist
                if self.rl_learner:
                    reward = self.rl_learner.compute_reward(before_state, str(action), after_state)
                    self.meta_strategist.observe(
                        state=before_state,
                        action=str(action),
                        reward=reward
                    )
                
                # Store cognitive memory in RAG
                self.memory_rag.store_cognitive_memory(
                    situation=before_state,
                    action_taken=str(action),
                    outcome=after_state,
                    success=success,
                    reasoning=f"Motivation: {dominant_drive.value}, Layer: {game_state.current_action_layer}"
                )
                
                # Record menu action if in menu
                if scene_type in [SceneType.INVENTORY, SceneType.MAP]:
                    after_scene = after_perception['scene_type']
                    self.menu_learner.record_action(
                        action=str(action),
                        success=success,
                        resulted_in_menu=after_scene.value if after_scene != scene_type else None
                    )
                elif self.menu_learner.current_menu:
                    # Exited menu
                    self.menu_learner.exit_menu()

                # Record in episodic memory (now with consciousness)
                self.agi.learner.experience(
                    data={
                        'cycle': cycle_count,
                        'scene': scene_type.value,
                        'action': str(action),
                        'motivation': dominant_drive.value,
                        'before': before_state,
                        'after': after_state,
                        'consciousness_before': self.current_consciousness,
                        'consciousness_after': after_consciousness,
                        'coherence_delta': after_consciousness.coherence_delta(self.current_consciousness) if self.current_consciousness else 0.0
                    },
                    context='skyrim_gameplay',
                    importance=0.5
                )

                # RL: Store experience with consciousness states (KEY INTEGRATION)
                if self.rl_learner is not None:
                    # Store experience for RL WITH CONSCIOUSNESS
                    self.rl_learner.store_experience(
                        state_before=before_state,
                        action=str(action),
                        state_after=after_state,
                        done=False,
                        consciousness_before=self.current_consciousness,  # NEW
                        consciousness_after=after_consciousness  # NEW
                    )
                    
                    print(f"[RL] Experience stored with consciousness (Δ𝒞 = {after_consciousness.coherence_delta(self.current_consciousness) if self.current_consciousness else 0.0:+.3f})")

                    # Train periodically
                    if cycle_count % self.config.rl_train_freq == 0:
                        print(f"[RL] Training at cycle {cycle_count}...")
                        self.rl_learner.train_step()

                    # Save RL model periodically
                    if cycle_count % 50 == 0:
                        self.rl_learner.save('skyrim_rl_model.pkl')

                # 8. UPDATE STATS (with consciousness tracking)
                self.stats['cycles_completed'] = cycle_count
                self.stats['total_playtime'] = time.time() - start_time

                # Track consciousness coherence (primary metric)
                if after_consciousness:
                    if 'consciousness_coherence_history' not in self.stats:
                        self.stats['consciousness_coherence_history'] = []
                    self.stats['consciousness_coherence_history'].append(after_consciousness.coherence)
                    
                    # Also track by Lumina
                    if 'coherence_by_lumina' not in self.stats:
                        self.stats['coherence_by_lumina'] = {
                            'ontical': [],
                            'structural': [],
                            'participatory': []
                        }
                    self.stats['coherence_by_lumina']['ontical'].append(after_consciousness.coherence_ontical)
                    self.stats['coherence_by_lumina']['structural'].append(after_consciousness.coherence_structural)
                    self.stats['coherence_by_lumina']['participatory'].append(after_consciousness.coherence_participatory)
                
                # Track game state quality (secondary metric)
                try:
                    cognitive_state = SkyrimCognitiveState.from_game_state(after_state)
                    self.stats['game_state_quality_history'].append(cognitive_state.overall_quality)
                except Exception:
                    # Fallback if cognitive state computation fails
                    pass

                # Auto-save periodically
                if time.time() - self.last_save_time > self.config.save_interval:
                    if not self.config.dry_run:
                        await self.actions.quick_save_checkpoint()
                    self.last_save_time = time.time()

                # Cycle complete
                cycle_duration = time.time() - cycle_start
                print(f"\n[OK] Cycle complete ({cycle_duration:.2f}s)")

        except KeyboardInterrupt:
            print("\n\n[WARN] User interrupted - stopping gracefully...")
        except Exception as e:
            print(f"\n\n[ERROR] Error during gameplay: {e}")
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
        goal: str
    ) -> str:
        """
        Plan next action based on perception, motivation, and goal.
        Now includes RL-based learning and layer-aware strategic reasoning.

        Args:
            perception: Current perception data
            motivation: Motivation state
            goal: Current goal

        Returns:
            Action to take
        """
        game_state = perception['game_state']
        scene_type = perception['scene_type']
        current_layer = game_state.current_action_layer
        available_actions = game_state.available_actions

        print(f"[PLANNING] Current layer: {current_layer}")
        print(f"[PLANNING] Available actions: {available_actions}")

        # Prepare state dict for RL
        state_dict = game_state.to_dict()
        state_dict.update({
            'scene': scene_type.value,
            'curiosity': motivation.curiosity,
            'competence': motivation.competence,
            'coherence': motivation.coherence,
            'autonomy': motivation.autonomy
        })

        # Increment meta-strategist cycle before planning
        self.meta_strategist.tick_cycle()
        # Use RL-based action selection if enabled
        if self.rl_learner is not None:
            print("[PLANNING] Using RL-based action selection with LLM reasoning...")
            print(f"[PLANNING] RL reasoning neuron LLM status: {'Connected' if self.rl_reasoning_neuron.llm_interface else 'Using heuristics'}")
            
            # Check if meta-strategist should generate new instruction
            if await self.meta_strategist.should_generate_instruction():
                q_values = self.rl_learner.get_q_values(state_dict)
                instruction = await self.meta_strategist.generate_instruction(
                    current_state=state_dict,
                    q_values=q_values,
                    motivation=motivation.dominant_drive().value
                )
            
            # Get Q-values from RL
            q_values = self.rl_learner.get_q_values(state_dict)
            print(f"[RL] Q-values: {', '.join([f'{k}={v:.2f}' for k, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]])}")
            
            # Get meta-strategic context
            meta_context = self.meta_strategist.get_active_instruction_context()
            
            # Use RL reasoning neuron to think about Q-values (with meta-strategic guidance)
            rl_reasoning = await self.rl_reasoning_neuron.reason_about_q_values(
                state=state_dict,
                q_values=q_values,
                available_actions=available_actions,
                context={
                    'motivation': motivation.dominant_drive().value,
                    'terrain_type': self.skyrim_world.classify_terrain_from_scene(
                        scene_type.value,
                        game_state.in_combat
                    ),
                    'meta_strategy': meta_context  # Add strategic guidance
                }
            )
            
            action = rl_reasoning.recommended_action
            print(f"[RL-NEURON] Action: {action} (tactical score: {rl_reasoning.tactical_score:.2f})")
            print(f"[RL-NEURON] Reasoning: {rl_reasoning.reasoning}")
            if rl_reasoning.strategic_insight:
                print(f"[RL-NEURON] Insight: {rl_reasoning.strategic_insight}")
            return action

        # Get strategic analysis from world model (layer effectiveness)
        strategic_analysis = self.skyrim_world.get_strategic_layer_analysis(
            game_state.to_dict()
        )
        
        # Get terrain-aware recommendations
        terrain_recommendations = self.skyrim_world.get_terrain_recommendations(
            game_state.location_name,
            scene_type.value,
            game_state.in_combat
        )
        strategic_analysis['terrain_recommendations'] = terrain_recommendations
        
        print(f"[STRATEGIC] Layer effectiveness: {strategic_analysis['layer_effectiveness']}")
        if strategic_analysis['recommendations']:
            print(f"[STRATEGIC] Recommendations: {strategic_analysis['recommendations']}")

        # Meta-strategic reasoning: Should we switch layers?
        optimal_layer = None
        
        # Combat situations - prioritize Combat layer if effective
        if game_state.in_combat or scene_type == SceneType.COMBAT:
            if current_layer != "Combat":
                combat_effectiveness = strategic_analysis['layer_effectiveness'].get('Combat', 0.5)
                if combat_effectiveness > 0.6:
                    optimal_layer = "Combat"
                    print(f"[META-STRATEGY] Switching to Combat layer (effectiveness: {combat_effectiveness:.2f})")
            
            # Choose combat action based on context
            if game_state.enemies_nearby > 2:
                return 'power_attack' if 'power_attack' in available_actions else 'combat'
            elif game_state.health < 50:
                return 'block' if 'block' in available_actions else 'combat'
            else:
                return 'combat'

        # Low health - consider Menu layer for healing
        if game_state.health < 30:
            if current_layer != "Menu":
                menu_effectiveness = strategic_analysis['layer_effectiveness'].get('Menu', 0.5)
                if menu_effectiveness > 0.5:
                    optimal_layer = "Menu"
                    print(f"[META-STRATEGY] Switching to Menu layer for healing (effectiveness: {menu_effectiveness:.2f})")
            
            if 'consume_item' in available_actions:
                return 'consume_item'
            else:
                return 'rest'

        # Stealth opportunities
        if (not game_state.in_combat and 
            len(game_state.nearby_npcs) > 0 and 
            motivation.dominant_drive().value == 'competence'):
            stealth_effectiveness = strategic_analysis['layer_effectiveness'].get('Stealth', 0.5)
            if stealth_effectiveness > 0.6 and current_layer != "Stealth":
                optimal_layer = "Stealth"
                print(f"[META-STRATEGY] Switching to Stealth layer (effectiveness: {stealth_effectiveness:.2f})")

        # If we determined an optimal layer, suggest layer transition
        if optimal_layer and optimal_layer != current_layer:
            # Return a meta-action that will trigger layer switch
            return f'switch_to_{optimal_layer.lower()}'

        # Try LLM-based planning if available, otherwise use heuristics
        has_attr = hasattr(self.agi, 'consciousness_llm')
        has_llm = has_attr and self.agi.consciousness_llm is not None
        
        print(f"[DEBUG] LLM Check: hasattr={has_attr}, consciousness_llm={self.agi.consciousness_llm if has_attr else 'N/A'}")
        
        if has_llm:
            print("[PLANNING] Using LLM-based strategic planning...")
            try:
                llm_action = await self._plan_action_with_llm(
                    perception, game_state, scene_type, current_layer, available_actions, 
                    strategic_analysis, motivation
                )
                if llm_action:
                    print(f"[LLM] Selected action: {llm_action}")
                    return llm_action
                else:
                    print("[LLM] LLM returned None, falling back to heuristics")
            except Exception as e:
                print(f"[LLM] Planning failed: {e}, using heuristics")
                import traceback
                traceback.print_exc()
        else:
            print("[PLANNING] LLM not available, using heuristic planning...")
            if has_attr:
                print(f"[PLANNING] consciousness_llm value: {self.agi.consciousness_llm}")

        # Fallback: Action selection within current layer based on motivation
        # Default to exploration with forward bias for most motivations
        if motivation.dominant_drive().value == 'curiosity':
            if 'activate' in available_actions:
                return 'activate'  # Interact with world
            return 'explore'  # Forward-biased exploration
        elif motivation.dominant_drive().value == 'competence':
            if 'power_attack' in available_actions and current_layer == "Combat":
                return 'power_attack'  # Practice advanced combat
            elif 'backstab' in available_actions and current_layer == "Stealth":
                return 'backstab'  # Practice stealth
            return 'explore'  # Practice by exploring (forward-biased)
        elif motivation.dominant_drive().value == 'coherence':
            # Even for coherence, prefer gentle exploration over rest
            if game_state.health < 30:
                return 'rest'  # Only rest if low health
            return 'explore'  # Gentle forward exploration
        else:  # autonomy or default
            return 'explore'  # Exercise autonomy through forward exploration

    async def _plan_action_with_llm(
        self,
        perception: Dict[str, Any],
        game_state,
        scene_type,
        current_layer: str,
        available_actions: list,
        strategic_analysis: dict,
        motivation
    ) -> Optional[str]:
        """
        Use LLM for terrain-aware, non-narrative action planning.
        
        Returns:
            Action string if LLM planning succeeds, None otherwise
        """
        # Build context for LLM focused on environment and terrain
        context = f"""
SKYRIM AGENT - TERRAIN-AWARE PLANNING

PHYSICAL STATE:
- Health: {game_state.health:.0f}/100 | Magicka: {game_state.magicka:.0f}/100 | Stamina: {game_state.stamina:.0f}/100
- Scene Type: {scene_type.value} (visual classification of environment)
- In Combat: {game_state.in_combat}

SPATIAL CONTEXT:
- Current Location: {game_state.location_name}
- Action Layer: {current_layer} (determines available movement/interaction options)
- Available Actions: {', '.join(available_actions)}

TERRAIN KNOWLEDGE:
You are an autonomous agent exploring a medieval fantasy world. Your goal is to navigate terrain intelligently:
- INDOOR spaces (inventory/menu scenes): Confined areas, look for exits, interact with objects
- OUTDOOR spaces (exploration scenes): Open terrain, prioritize forward movement, scan horizon
- COMBAT spaces: Immediate threats, use terrain for advantage (cover, elevation, retreat paths)
- VERTICAL terrain: Cliffs, stairs, elevated positions - consider climbing/jumping
- OBSTACLES: Walls, rocks, water - navigate around or find alternate paths

STRATEGIC ANALYSIS:
- Layer Effectiveness: {strategic_analysis['layer_effectiveness']}
- Recommendations: {strategic_analysis.get('recommendations', [])}

BEHAVIORAL DRIVE: {motivation.dominant_drive().value}

AVAILABLE ACTIONS (terrain-focused):
- explore: Forward-biased waypoint navigation with camera scanning
- navigate: Direct forward movement to cover distance
- combat: Engage threats using terrain advantages
- interact: Activate objects, open doors, loot containers
- rest: Recover resources when safe
- switch_to_combat/menu/stealth: Change action layer for different terrain interactions

PLANNING CONSTRAINTS:
1. Prioritize FORWARD movement in open terrain
2. Use CAMERA scanning to assess environment
3. Consider VERTICAL space (look up for paths, down for items)
4. Adapt to TERRAIN type (indoor vs outdoor vs combat)
5. NO story/quest assumptions - pure environmental reasoning
6. Focus on SPATIAL navigation, not narrative goals

Based on the terrain type and physical state, select the most appropriate action for navigating this environment:"""

        # Augment context with RAG memories
        memory_context = self.memory_rag.augment_context_with_memories(
            current_visual=perception['visual_embedding'],
            current_situation={
                'scene': scene_type.value,
                'health': game_state.health,
                'in_combat': game_state.in_combat,
                'location': game_state.location_name
            },
            max_memories=3
        )
        
        if memory_context:
            context += "\n" + memory_context
            print("[RAG] Augmented context with relevant memories")

        try:
            print("[LLM] Calling LM Studio for layer-aware action planning...")
            print(f"[LLM] Context length: {len(context)} characters")
            
            result = await self.agi.process(context)
            print(f"[LLM] Result type: {type(result)}")
            print(f"[LLM] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            response = result.get('consciousness_response', {}).get('response', '')
            print(f"[LLM] Raw response: {response}")
            print(f"[LLM] Response length: {len(response)} characters")

            # Parse LLM response to extract action
            response_lower = response.lower()
            
            # Check for layer transition actions first
            if 'switch_to_combat' in response_lower:
                return 'switch_to_combat'
            elif 'switch_to_menu' in response_lower:
                return 'switch_to_menu'
            elif 'switch_to_stealth' in response_lower:
                return 'switch_to_stealth'
            elif 'switch_to_exploration' in response_lower:
                return 'switch_to_exploration'
            
            # Check for specific actions
            elif 'power_attack' in response_lower and 'power_attack' in available_actions:
                return 'power_attack'
            elif 'backstab' in response_lower and 'backstab' in available_actions:
                return 'backstab'
            elif 'block' in response_lower and 'block' in available_actions:
                return 'block'
            elif 'activate' in response_lower and 'activate' in available_actions:
                return 'activate'
            elif 'move_forward' in response_lower and 'move_forward' in available_actions:
                return 'move_forward'
            
            # Check for general action categories
            elif 'combat' in response_lower or 'attack' in response_lower:
                return 'combat'
            elif 'explore' in response_lower:
                return 'explore'
            elif 'interact' in response_lower:
                return 'interact'
            elif 'stealth' in response_lower or 'sneak' in response_lower:
                return 'stealth'
            elif 'rest' in response_lower or 'heal' in response_lower:
                return 'rest'
            elif 'navigate' in response_lower or 'move' in response_lower:
                return 'navigate'
            
            print(f"[LLM] Could not parse action from response, using fallback")
            return None
            
        except Exception as e:
            print(f"[LLM] Error during planning: {e}")
            return None

    def _evaluate_action_success(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        action: str
    ) -> bool:
        """
        Evaluate if an action was successful.
        
        Args:
            before_state: State before action
            after_state: State after action
            action: Action taken
            
        Returns:
            True if action was successful
        """
        # Simple heuristics for success
        # Success if health didn't decrease significantly
        health_before = before_state.get('health', 100)
        health_after = after_state.get('health', 100)
        
        if health_after < health_before - 20:
            return False  # Took significant damage
        
        # Success if we're not stuck (scene changed or position changed)
        scene_before = before_state.get('scene', '')
        scene_after = after_state.get('scene', '')
        
        if scene_before != scene_after:
            return True  # Scene changed - progress made
        
        # Default to success if no obvious failure
        return True

    async def _test_controller_connection(self):
        """Test controller connection and basic functionality."""
        try:
            print("[CONTROLLER] Testing controller connection...")
            
            if self.controller and hasattr(self.controller, 'active_layer'):
                print(f"[CONTROLLER] Active layer: {self.controller.active_layer}")
                print("[CONTROLLER] ✓ Controller connection OK")
            else:
                print("[CONTROLLER] ⚠️ Controller not properly initialized")
                
            # Test basic action
            if not self.config.dry_run:
                print("[CONTROLLER] Testing basic look action...")
                await self.actions.look_horizontal(5.0)  # Small test movement
                await asyncio.sleep(0.2)
                await self.actions.look_horizontal(-5.0)  # Return to center
                print("[CONTROLLER] ✓ Basic actions working")
            else:
                print("[CONTROLLER] Dry run mode - skipping action test")
                
        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Controller test failed: {e}")
            print("[CONTROLLER] Continuing anyway...")

    def _print_final_stats(self):
        """Print final gameplay statistics."""
        print(f"\n{'=' * 60}")
        print("FINAL STATISTICS")
        print(f"{'=' * 60}")
        
        # Determine skill level
        if self.stats['cycles_completed'] < 10:
            skill_level = 'novice'
        elif self.stats['cycles_completed'] < 50:
            skill_level = 'apprentice'
        elif self.stats['cycles_completed'] < 100:
            skill_level = 'adept'
        elif self.stats['cycles_completed'] < 200:
            skill_level = 'expert'
        else:
            skill_level = 'master'
        
        print(f"\nGameplay:")
        print(f"  Cycles: {self.stats['cycles_completed']}")
        print(f"  Actions: {self.stats['actions_taken']}")
        print(f"  Skill Level: {skill_level.title()}")
        
        # Check if this is practice mode
        if self.stats['cycles_completed'] < 20:
            if self.stats['actions_taken'] > 0:
                return 'practice'
        print(f"  Playtime: {self.stats['total_playtime'] / 60:.1f} minutes")

        if self.stats['game_state_quality_history']:
            avg_quality = sum(self.stats['game_state_quality_history']) / len(self.stats['game_state_quality_history'])
            print(f"  Avg Game State Quality: {avg_quality:.3f}")

        skyrim_stats = self.skyrim_world.get_stats()
        print(f"\nWorld Model:")
        print(f"  Causal edges learned: {skyrim_stats['causal_edges']}")
        print(f"  NPCs met: {skyrim_stats['npc_relationships']}")
        print(f"  Locations: {skyrim_stats['locations_discovered']}")

        action_stats = self.actions.get_stats()
        print(f"\nActions:")
        print(f"  Total executed: {action_stats['actions_executed']}")
        print(f"  Errors: {action_stats['errors']}")

    async def _execute_action(self, action: str, scene_type: SceneType):
        """
        Execute planned action.

        Args:
            action: Action to execute
            scene_type: Current scene type
        """
        print(f"[DEBUG] Executing action: {action} | Scene: {scene_type} | Active layer: {self.controller.active_layer}")
        
        # Handle menu interactions with learning
        if scene_type in [SceneType.INVENTORY, SceneType.MAP]:
            # We're in a menu - use menu learner
            if not self.menu_learner.current_menu:
                # Entering menu
                menu_type = scene_type.value
                available_actions = ['activate', 'navigate', 'exit', 'select', 'back']
                self.menu_learner.enter_menu(menu_type, available_actions)
            
            # Get recommended action from menu learner
            suggested_action = self.menu_learner.suggest_menu_action(
                self.menu_learner.current_menu or scene_type.value,
                goal='explore' if action == 'explore' else 'exit'
            )
            
            if suggested_action:
                print(f"[MENU] Using learned action: {suggested_action}")
                action = suggested_action
        
        # Sync action layer to context
        if action in ('explore', 'navigate', 'quest_objective', 'practice'):
            self.bindings.switch_to_exploration()
        elif action == 'combat':
            self.bindings.switch_to_combat()
        elif action == 'interact':
            self.bindings.switch_to_exploration()
        elif action == 'rest':
            self.bindings.switch_to_exploration()
        # Extend with menu/dialogue/stealth as needed
        print(f"[DEBUG] After layer switch: {self.controller.active_layer}")

        if action == 'explore':
            # Use waypoint-based exploration instead of random movement
            await self.actions.explore_with_waypoints(duration=3.0)
        elif action == 'combat':
            await self.actions.combat_sequence("Enemy")
        elif action == 'interact':
            await self.actions.execute(Action(ActionType.ACTIVATE))
        elif action == 'navigate':
            await self.actions.move_forward(duration=2.0)
        elif action == 'rest':
            await self.actions.execute(Action(ActionType.WAIT))
        elif action == 'practice':
            await self.actions.execute(Action(ActionType.ATTACK))
        elif action == 'quest_objective':
            await self.actions.move_forward(duration=2.0)
        elif action.startswith('switch_to_'):
            # Handle layer transition actions
            target_layer = action.replace('switch_to_', '').title()
            print(f"[META-STRATEGY] Executing layer transition to {target_layer}")
            
            if target_layer == 'Combat':
                self.bindings.switch_to_combat()
                # Perform a combat-ready action
                await self.actions.execute(Action(ActionType.ATTACK))
            elif target_layer == 'Menu':
                self.bindings.switch_to_menu()
                # Open menu (would be handled by controller bindings)
                print("[LAYER] Switched to Menu layer - ready for inventory management")
            elif target_layer == 'Stealth':
                self.bindings.switch_to_stealth()
                # Enter sneak mode
                await self.actions.execute(Action(ActionType.SNEAK))
            elif target_layer == 'Exploration':
                self.bindings.switch_to_exploration()
                # Continue exploration
                await self.actions.explore_with_waypoints(duration=2.0)
        else:
            # Fallback to waypoint exploration
            await self.actions.explore_with_waypoints(duration=2.0)

    def _print_final_stats(self):
        """Print final statistics."""
        print(f"\nFinal Statistics:")
        print(f"  Cycles: {self.stats['cycles_completed']}")
        print(f"  Actions: {self.stats['actions_taken']}")
        print(f"  Playtime: {self.stats['total_playtime'] / 60:.1f} minutes")
        if self.stats['game_state_quality_history']:
            avg_quality = sum(self.stats['game_state_quality_history']) / len(self.stats['game_state_quality_history'])
            print(f"  Avg Game State Quality: {avg_quality:.3f}")
        skyrim_stats = self.skyrim_world.get_stats()
        print(f"\nWorld Model:")
        print(f"  Causal edges learned: {skyrim_stats['causal_edges']}")
        print(f"  NPCs met: {skyrim_stats['npc_relationships']}")
        print(f"  Locations: {skyrim_stats['locations_discovered']}")
        action_stats = self.actions.get_stats()
        print(f"\nActions:")
        print(f"  Total executed: {action_stats['actions_executed']}")
        print(f"  Errors: {action_stats['errors']}")
        
        # Strategic planner stats
        planner_stats = self.strategic_planner.get_stats()
        print(f"\nStrategic Planner:")
        print(f"  Patterns learned: {planner_stats['patterns_learned']}")
        print(f"  Experiences: {planner_stats['experiences_recorded']}")
        print(f"  Plans executed: {planner_stats['plans_executed']}")
        print(f"  Success rate: {planner_stats['success_rate']:.1%}")
        
        # Menu learner stats
        menu_stats = self.menu_learner.get_stats()
        print(f"\nMenu Learning:")
        print(f"  Menus explored: {menu_stats['menus_explored']}")
        print(f"  Menu actions: {menu_stats['total_menu_actions']}")
        print(f"  Transitions learned: {menu_stats['transitions_learned']}")
        
        # Memory RAG stats
        rag_stats = self.memory_rag.get_stats()
        print(f"\nMemory RAG:")
        print(f"  Perceptual memories: {rag_stats['perceptual_memories']}")
        print(f"  Cognitive memories: {rag_stats['cognitive_memories']}")
        print(f"  Total memories: {rag_stats['total_memories']}")

        # RL stats
        if self.rl_learner is not None:
            rl_stats = self.rl_learner.get_stats()
            print(f"\nReinforcement Learning:")
            print(f"  Total experiences: {rl_stats['total_experiences']}")
            print(f"  Training steps: {rl_stats['training_steps']}")
            print(f"  Avg reward: {rl_stats['avg_reward']:.3f}")
            print(f"  Exploration rate (ε): {rl_stats['epsilon']:.3f}")
            print(f"  Buffer size: {rl_stats['buffer_size']}")
            print(f"  Avg Q-value: {rl_stats['avg_q_value']:.3f}")
        
        # RL reasoning neuron stats
        rl_neuron_stats = self.rl_reasoning_neuron.get_stats()
        print(f"\nRL Reasoning Neuron (LLM-Enhanced):")
        print(f"  Total reasonings: {rl_neuron_stats['total_reasonings']}")
        print(f"  Avg confidence: {rl_neuron_stats['avg_confidence']:.3f}")
        print(f"  Avg tactical score: {rl_neuron_stats['avg_tactical_score']:.3f}")
        print(f"  Patterns learned: {rl_neuron_stats['patterns_learned']}")
        
        # Meta-strategist stats
        meta_stats = self.meta_strategist.get_stats()
        print(f"\nMeta-Strategist (Mistral-7B):")
        print(f"  Active instructions: {meta_stats['active_instructions']}")
        print(f"  Total generated: {meta_stats['total_generated']}")
        print(f"  Current cycle: {meta_stats['current_cycle']}")
        print(f"  Cycles since last: {meta_stats['cycles_since_last']}")
        
        # Consciousness bridge stats (NEW)
        consciousness_stats = self.consciousness_bridge.get_stats()
        print(f"\n🧠 Consciousness Bridge (Singularis Integration):")
        print(f"  Total measurements: {consciousness_stats['total_measurements']}")
        print(f"  Avg coherence 𝒞: {consciousness_stats['avg_coherence']:.3f}")
        print(f"  Avg consciousness Φ̂: {consciousness_stats['avg_consciousness']:.3f}")
        print(f"  Coherence trend: {consciousness_stats['trend']}")
        if 'coherence_by_lumina' in consciousness_stats and consciousness_stats['coherence_by_lumina']:
            lumina = consciousness_stats['coherence_by_lumina']
            print(f"  Three Lumina:")
            print(f"    ℓₒ (Ontical): {lumina['ontical']:.3f}")
            print(f"    ℓₛ (Structural): {lumina['structural']:.3f}")
            print(f"    ℓₚ (Participatory): {lumina['participatory']:.3f}")
        
        # Show consciousness vs game quality correlation
        if 'consciousness_coherence_history' in self.stats and self.stats['consciousness_coherence_history']:
            avg_consciousness_coherence = sum(self.stats['consciousness_coherence_history']) / len(self.stats['consciousness_coherence_history'])
            avg_game_quality = sum(self.stats['game_state_quality_history']) / len(self.stats['game_state_quality_history']) if self.stats['game_state_quality_history'] else 0
            print(f"\n  Consciousness 𝒞: {avg_consciousness_coherence:.3f}")
            print(f"  Game Quality: {avg_game_quality:.3f}")
            print(f"  Combined Value: {0.6 * avg_consciousness_coherence + 0.4 * avg_game_quality:.3f}")
            print(f"  (60% consciousness + 40% game = unified evaluation)")

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
