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
import random
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
from .combat_tactics import SkyrimCombatTactics
from .quest_tracker import QuestTracker
from .smart_navigation import SmartNavigator
from .inventory_manager import InventoryManager
from .dialogue_intelligence import DialogueIntelligence
from .character_progression import CharacterProgression
from .enhanced_vision import EnhancedVision
from .crafting_system import CraftingSystem
from .hierarchical_goal_planner import HierarchicalGoalPlanner
from .adaptive_loop_manager import AdaptiveLoopManager, LoopSettings
from .gameplay_analytics import GameplayAnalytics
from .meta_learning import MetaLearner

# Base AGI components
from ..agi_orchestrator import AGIOrchestrator, AGIConfig
from ..agency import MotivationType
from ..llm import (
    LMStudioClient,
    LMStudioConfig,
    ExpertLLMInterface,
    ClaudeClient,
    GeminiClient,
    HybridLLMClient,
    HybridConfig,
    TaskType,
)


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
    
    # Async execution
    enable_async_reasoning: bool = True  # Run reasoning in parallel with actions
    action_queue_size: int = 3  # Max queued actions
    perception_interval: float = 0.25  # How often to perceive (seconds) - faster for responsiveness
    max_concurrent_llm_calls: int = 4  # With 4 models (2 mistral + 2 big), can handle 4 concurrent
    reasoning_throttle: float = 0.1  # Min seconds between reasoning cycles - minimal throttle
    
    # Fast reactive loop
    enable_fast_loop: bool = True  # Enable fast reactive loop for immediate responses
    fast_loop_interval: float = 0.5  # Fast loop runs every half second - twice as fast
    fast_health_threshold: float = 30.0  # Health % to trigger emergency healing
    fast_danger_threshold: int = 3  # Number of enemies to trigger defensive actions

    # Core models
    phi4_action_model: str = "mistralai/mistral-nemo-instruct-2407"  # Action planning
    huihui_cognition_model: str = "huihui-moe-60b-a3b-abliterated-i1"  # Main cognition, reasoning, strategy
    qwen3_vl_perception_model: str = "qwen/qwen3-vl-8b"  # Perception and spatial awareness

    # Learning
    surprise_threshold: float = 0.3  # Threshold for learning from surprise
    exploration_weight: float = 0.5  # How much to favor exploration

    # Reinforcement Learning
    use_rl: bool = True  # Enable RL-based learning
    rl_learning_rate: float = 0.01  # Q-network learning rate
    rl_epsilon_start: float = 0.3  # Initial exploration rate
    rl_train_freq: int = 5  # Train every N cycles

    # Hybrid LLM Architecture (Primary: Gemini + Claude, Optional Fallback: Local)
    use_hybrid_llm: bool = True
    use_gemini_vision: bool = True
    gemini_model: str = "gemini-2.0-flash-exp"
    use_claude_reasoning: bool = True
    claude_model: str = "claude-sonnet-4-20250514"
    use_local_fallback: bool = False  # Optional local LLMs as fallback
    
    # Legacy external augmentation (deprecated in favor of hybrid)
    enable_claude_meta: bool = False
    enable_gemini_vision: bool = False
    gemini_max_output_tokens: int = 768

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
        
        # Hybrid LLM system (Gemini + Claude + optional local fallback)
        self.hybrid_llm: Optional[HybridLLMClient] = None
        
        # Legacy LLM references (for backward compatibility)
        self.huihui_llm = None  # Main cognition
        self.perception_llm = None  # Visual perception
        self.action_planning_llm = None  # Action planning
        
        # 11. Meta-Strategist (coordinates tactical & strategic thinking)
        print("  [11/12] Meta-strategist coordinator...")
        # Lower instruction frequency for faster API testing (default: 10)
        self.meta_strategist = MetaStrategist(instruction_frequency=3)
        # Will connect LLM interface when initialized
        
        # 12. Skyrim-specific Motivation System
        print("  [11/11] Skyrim-specific motivation system...")
        self.skyrim_motivation = SkyrimMotivation(
            survival_weight=0.35,  # Prioritize staying alive
            progression_weight=0.25,  # Character growth important
            exploration_weight=0.20,  # Discover new areas
            wealth_weight=0.10,  # Gather resources
            mastery_weight=0.10  # Improve combat/stealth skills
        )

        # Skyrim-specific enhancement modules
        self.combat_tactics = SkyrimCombatTactics()
        self.quest_tracker = QuestTracker()
        self.smart_navigator = SmartNavigator()
        self.inventory_manager = InventoryManager()
        self.dialogue_intelligence = DialogueIntelligence()
        self.character_progression = CharacterProgression()
        self.crafting_system = CraftingSystem()
        self.goal_planner = HierarchicalGoalPlanner()
        self.analytics = GameplayAnalytics()
        self.meta_learner = MetaLearner()

        # External augmentation clients (initialized later)
        self.claude_meta_client = None
        self.gemini_vision_client = None

        # Enhanced perception utilities
        self.enhanced_vision = EnhancedVision()
        self.perception.set_enhanced_vision(self.enhanced_vision)

        # Adaptive loop scheduling
        self.loop_manager = AdaptiveLoopManager(
            LoopSettings(
                perception_interval=self.config.perception_interval,
                reasoning_throttle=self.config.reasoning_throttle,
                fast_loop_interval=self.config.fast_loop_interval
            )
        )

        # State
        self.running = False
        self.current_perception: Optional[Dict[str, Any]] = None
        self.current_goal: Optional[str] = None
        self.last_save_time = time.time()
        self.last_state: Optional[Dict[str, Any]] = None  # For RL experience tracking
        self.last_action: Optional[str] = None
        
        # Async queues for parallel execution
        self.perception_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self.action_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.action_queue_size)
        self.learning_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.fast_action_queue: asyncio.Queue = asyncio.Queue(maxsize=5)  # Fast reactive actions
        
        # Resource management for async execution
        self.llm_semaphore: Optional[asyncio.Semaphore] = None  # Limit concurrent LLM calls
        self.last_reasoning_time: float = 0.0  # Track last reasoning to throttle
        
        # Multi-LLM architecture (initialized in initialize_llm)
        # Mistral Nemo for fast consciousness/tactical reasoning
        self.action_planning_llm: Optional[Any] = None  # mistral-nemo: Fast action planning
        # 2x big models for high-level strategic planning
        self.strategic_planning_llm: Optional[Any] = None  # Big model: Long-term strategy
        self.world_understanding_llm: Optional[Any] = None  # Big model: Deep world understanding

        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'actions_taken': 0,
            'locations_discovered': 0,
            'npcs_met': 0,
            'quests_completed': 0,
            'total_playtime': 0.0,
            'game_state_quality_history': [],  # Replaces coherence_history
            # Performance metrics
            'action_success_count': 0,
            'action_failure_count': 0,
            'llm_action_count': 0,
            'heuristic_action_count': 0,
            'rl_action_count': 0,
            'fast_action_count': 0,  # Fast reactive actions
            'planning_times': [],  # Track planning duration
            'execution_times': [],  # Track execution duration
            'fast_action_times': []  # Track fast action execution
        }

        # Set up controller reference in perception for layer awareness
        self.perception.set_controller(self.controller)
        
        # State tracking for consciousness
        self.current_consciousness: Optional[ConsciousnessState] = None
        self.last_consciousness: Optional[ConsciousnessState] = None
        
        # Stuck detection and recovery
        self.stuck_detection_window = 5  # Check last N actions
        self.action_history = []  # Track recent actions
        self.coherence_history = []  # Track recent coherence values
        self.stuck_threshold = 0.02  # Coherence change threshold
        self.consecutive_same_action = 0  # Count same action repeats
        self.last_executed_action = None
        
        print("Skyrim AGI initialization complete.")
        print("[OK] Skyrim AGI initialized with CONSCIOUSNESS INTEGRATION\n")


    async def initialize_llm(self):
        """
        Initialize hybrid LLM architecture: Gemini (vision) + Claude Sonnet 4 (reasoning).
        
        Primary Architecture:
        - Gemini 2.0 Flash: Vision and visual perception
          * Fast, efficient image analysis
          * Scene understanding and spatial awareness
          * Real-time visual feedback
        
        - Claude Sonnet 4: Strategic reasoning and planning
          * High-level strategic thinking
          * Complex decision making
          * World understanding and causal reasoning
          * Action planning and tactical decisions
        
        Optional Fallback (if enabled):
        - Local LLMs via LM Studio
          * Vision: qwen3-vl-8b
          * Reasoning: huihui-moe-60b
          * Action: mistral-nemo
        
        This hybrid approach leverages cloud AI for primary intelligence with
        optional local fallback for reliability.
        """
        print("=" * 70)
        print("INITIALIZING HYBRID LLM ARCHITECTURE")
        print("=" * 70)
        print("Primary: Gemini 2.0 Flash (vision) + Claude Sonnet 4 (reasoning)")
        if self.config.use_local_fallback:
            print("Fallback: Local LLMs (optional)")
        else:
            print("Fallback: Disabled")
        print("=" * 70)
        print()
        
        # ===== HYBRID LLM SYSTEM =====
        if self.config.use_hybrid_llm:
            try:
                # Configure hybrid system
                hybrid_config = HybridConfig(
                    use_gemini_vision=self.config.use_gemini_vision,
                    gemini_model=self.config.gemini_model,
                    use_claude_reasoning=self.config.use_claude_reasoning,
                    claude_model=self.config.claude_model,
                    use_local_fallback=self.config.use_local_fallback,
                    local_base_url=self.config.base_config.lm_studio_url if self.config.use_local_fallback else "http://localhost:1234/v1",
                    local_vision_model=self.config.qwen3_vl_perception_model if self.config.use_local_fallback else "qwen/qwen3-vl-8b",
                    local_reasoning_model=self.config.huihui_cognition_model if self.config.use_local_fallback else "huihui-moe-60b-a3b-abliterated-i1",
                    local_action_model=self.config.phi4_action_model if self.config.use_local_fallback else "mistralai/mistral-nemo-instruct-2407",
                    timeout=30,
                    max_concurrent_requests=self.config.max_concurrent_llm_calls,
                )
                
                # Initialize hybrid client
                self.hybrid_llm = HybridLLMClient(hybrid_config)
                await self.hybrid_llm.initialize()
                
                print("\n[HYBRID] ✓ Hybrid LLM system initialized successfully")
                
                # Connect hybrid system to all components
                await self._connect_hybrid_llm()
                
            except Exception as e:
                print(f"[HYBRID] ⚠️ Failed to initialize hybrid system: {e}")
                import traceback
                traceback.print_exc()
                self.hybrid_llm = None
        else:
            print("[HYBRID] Hybrid LLM system disabled, using legacy architecture")
            # Fall back to legacy initialization if needed
            await self._initialize_legacy_llms()
        
        print("\n" + "=" * 70)
        print("LLM ARCHITECTURE READY")
        print("=" * 70)
        if self.hybrid_llm:
            print("✓ Hybrid system active: Gemini (vision) + Claude Sonnet 4 (reasoning)")
            if self.config.use_local_fallback:
                print("✓ Local fallback enabled")
        print("Async execution for parallel processing")
        print("=" * 70)
        print()

    async def _connect_hybrid_llm(self):
        """Connect hybrid LLM system to all AGI components."""
        if not self.hybrid_llm:
            return
        
        print("\n[HYBRID] Connecting to AGI components...")
        
        # Connect to perception for vision tasks
        if hasattr(self.perception, 'set_hybrid_llm'):
            self.perception.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to perception system")
        
        # Connect to strategic planner for reasoning
        if self.strategic_planner and hasattr(self.strategic_planner, 'set_hybrid_llm'):
            self.strategic_planner.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to strategic planner")
        
        # Connect to meta-strategist
        if hasattr(self.meta_strategist, 'set_hybrid_llm'):
            self.meta_strategist.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to meta-strategist")
        
        # Connect to RL reasoning neuron
        if hasattr(self.rl_reasoning_neuron, 'set_hybrid_llm'):
            self.rl_reasoning_neuron.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to RL reasoning neuron")
        
        # Connect to world model
        if hasattr(self.skyrim_world, 'set_hybrid_llm'):
            self.skyrim_world.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to world model")
        
        # Connect to consciousness bridge
        if hasattr(self.consciousness_bridge, 'set_hybrid_llm'):
            self.consciousness_bridge.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to consciousness bridge")
        
        # Connect to quest tracker and dialogue
        if hasattr(self.quest_tracker, 'set_hybrid_llm'):
            self.quest_tracker.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to quest tracker")
        
        if hasattr(self.dialogue_intelligence, 'set_hybrid_llm'):
            self.dialogue_intelligence.set_hybrid_llm(self.hybrid_llm)
            print("[HYBRID] ✓ Connected to dialogue intelligence")
        
        print("[HYBRID] Component connection complete\n")
    
    async def _initialize_legacy_llms(self):
        """Initialize legacy local LLM architecture (fallback if hybrid disabled)."""
        print("\n[LEGACY] Initializing legacy local LLM architecture...")
        
        try:
            # Initialize huihui as the main LLM
            huihui_config = LMStudioConfig(
                base_url=self.config.base_config.lm_studio_url,
                model_name=self.config.huihui_cognition_model,
                temperature=0.7,
                max_tokens=2048
            )
            huihui_client = LMStudioClient(huihui_config)
            self.huihui_llm = ExpertLLMInterface(huihui_client)
            print("[LEGACY] ✓ Main cognition LLM initialized")
            
            # Initialize base Singularis AGI
            self.agi.config.lm_studio_url = self.config.base_config.lm_studio_url
            self.agi.config.model_name = self.config.huihui_cognition_model
            await self.agi.initialize_llm()
            print("[LEGACY] ✓ Base Singularis AGI initialized")
            
            # Connect to components
            self.rl_reasoning_neuron.llm_interface = self.huihui_llm
            self.meta_strategist.llm_interface = self.huihui_llm
            self.strategic_planner.llm_interface = self.huihui_llm
            self.quest_tracker.set_llm_interface(self.huihui_llm)
            self.dialogue_intelligence.set_llm_interface(self.huihui_llm)
            
            if hasattr(self.agi, 'consciousness_llm') and self.agi.consciousness_llm:
                self.consciousness_bridge.consciousness_llm = self.agi.consciousness_llm
            else:
                self.consciousness_bridge.consciousness_llm = self.huihui_llm
            
            print("[LEGACY] ✓ Legacy LLM system ready")
            
        except Exception as e:
            print(f"[LEGACY] ⚠️ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()

    async def _initialize_claude_meta(self) -> None:
        """Bring up optional Claude client for auxiliary strategic reasoning."""

        if not self.config.enable_claude_meta:
            print("[CLAUDE] Auxiliary meta reasoning disabled via config")
            self.claude_meta_client = None
            return

        try:
            self.claude_meta_client = ClaudeClient(model=self.config.claude_model)
            if not self.claude_meta_client.is_available():
                print("[CLAUDE] API key missing (ANTHROPIC_API_KEY); skipping auxiliary meta reasoning")
                self.claude_meta_client = None
                return

            self.meta_strategist.add_auxiliary_interface(self.claude_meta_client, self.config.claude_model)
            print("[CLAUDE] ✓ Auxiliary meta reasoning client ready (runs alongside Huihui)")

        except Exception as exc:
            print(f"[CLAUDE] ⚠️ Failed to initialize auxiliary meta reasoning: {exc}")
            import traceback
            traceback.print_exc()
            self.claude_meta_client = None

    async def _initialize_gemini_vision(self) -> None:
        """Attach Gemini client as an optional vision augment."""

        if not self.config.enable_gemini_vision:
            print("[GEMINI] Vision augmentation disabled via config")
            self.gemini_vision_client = None
            self.perception.set_gemini_analyzer(None)
            return

        try:
            self.gemini_vision_client = GeminiClient(model=self.config.gemini_model)
            if not self.gemini_vision_client.is_available():
                print("[GEMINI] API key missing (GEMINI_API_KEY); keeping local vision only")
                self.gemini_vision_client = None
                self.perception.set_gemini_analyzer(None)
                return

            self.perception.set_gemini_analyzer(self.gemini_vision_client)
            print("[GEMINI] ✓ Vision augmentation ready (complements Qwen/CLIP)")

        except Exception as exc:
            print(f"[GEMINI] ⚠️ Failed to initialize vision augmentation: {exc}")
            import traceback
            traceback.print_exc()
            self.gemini_vision_client = None
            self.perception.set_gemini_analyzer(None)

    async def _augment_with_gemini(self, perception: Dict[str, Any], cycle_count: int) -> Optional[str]:
        """Run optional Gemini analysis and attach its summary to the perception payload."""

        if not self.config.enable_gemini_vision or self.gemini_vision_client is None:
            return None

        # Throttle Gemini usage so it complements rather than overloads the loop.
        # Changed from every 3rd to every 2nd cycle for faster testing
        if cycle_count % 2 != 0:
            return None

        screenshot = perception.get('screenshot')
        if screenshot is None:
            return None

        scene = perception.get('scene_type', SceneType.UNKNOWN)
        scene_label = scene.value if hasattr(scene, 'value') else str(scene)
        game_state: Optional[GameState] = perception.get('game_state')

        location = getattr(game_state, 'location_name', 'Unknown') if game_state else 'Unknown'
        health_value = getattr(game_state, 'health', None) if game_state else None
        health_str = f"{health_value:.0f}" if isinstance(health_value, (int, float)) else "unknown"
        in_combat = getattr(game_state, 'in_combat', False) if game_state else False
        enemies_value = getattr(game_state, 'enemies_nearby', None) if game_state else None
        enemies_str = str(enemies_value) if enemies_value is not None else "unknown"

        prompt = (
            "You supplement a local Skyrim perception module."
            f"\nScene type: {scene_label}"
            f"\nLocation: {location}"
            f"\nIn combat: {in_combat}"
            f"\nPlayer health: {health_str}/100"
            f"\nEnemies nearby: {enemies_str}"
            "\n\nProvide a concise tactical snapshot with bullets covering:"
            "\n1. Immediate threats or hazards"
            "\n2. Valuable interactables or loot"
            "\n3. Recommended tactical focus"
            "\nKeep the response under 90 words."
        )

        image_for_gemini = screenshot if screenshot.mode == "RGB" else screenshot.convert("RGB")

        try:
            analysis = await asyncio.wait_for(
                self.gemini_vision_client.analyze_image(
                    prompt=prompt,
                    image=image_for_gemini,
                    max_output_tokens=self.config.gemini_max_output_tokens,
                    temperature=0.4,
                ),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            if cycle_count % 20 == 0:
                print("[GEMINI] Analysis timed out (12s) - continuing with local perception")
            return None
        except Exception as exc:
            if cycle_count % 20 == 0:
                print(f"[GEMINI] Analysis error: {exc}")
            return None

        if not analysis:
            return None

        analysis = analysis.strip()
        if not analysis:
            return None

        perception['gemini_analysis'] = analysis
        return analysis

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
        
        Now supports async mode where reasoning and actions run in parallel.

        Args:
            duration_seconds: How long to play (default: config value)
        """
        if duration_seconds is None:
            duration_seconds = self.config.autonomous_duration

        print(f"\n{'=' * 60}")
        print(f"STARTING AUTONOMOUS GAMEPLAY")
        print(f"Starting autonomous gameplay for {duration_seconds}s...")
        print(f"Cycle interval: {self.config.cycle_interval}s")
        if self.config.enable_async_reasoning:
            print(f"Async mode: ENABLED (reasoning runs in parallel with actions)")
            print(f"Max concurrent LLM calls: {self.config.max_concurrent_llm_calls}")
            print(f"Reasoning throttle: {self.config.reasoning_throttle}s")
        else:
            print(f"Async mode: DISABLED (sequential execution)")
        if self.config.enable_fast_loop:
            print(f"Fast reactive loop: ENABLED ({self.config.fast_loop_interval}s interval)")
            print(f"  Health threshold: {self.config.fast_health_threshold}%")
            print(f"  Danger threshold: {self.config.fast_danger_threshold} enemies")
        else:
            print(f"Fast reactive loop: DISABLED")
        print("=" * 60)
        print()

        # Test controller connection before starting
        await self._test_controller_connection()

        self.running = True
        start_time = time.time()
        
        # Initialize LLM semaphore for resource management
        self.llm_semaphore = asyncio.Semaphore(self.config.max_concurrent_llm_calls)

        try:
            if self.config.enable_async_reasoning:
                # Run async mode with parallel reasoning and actions
                await self._autonomous_play_async(duration_seconds, start_time)
            else:
                # Run traditional sequential mode
                await self._autonomous_play_sequential(duration_seconds, start_time)
                
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

    async def _autonomous_play_async(self, duration_seconds: int, start_time: float):
        """
        Async gameplay mode where reasoning, actions, and perception run in parallel.
        
        This prevents blocking during LLM reasoning - the agent continues to act.
        Now includes fast reactive loop for immediate survival responses.
        """
        print("[ASYNC] Starting parallel execution loops...")
        
        # Start all async loops concurrently
        perception_task = asyncio.create_task(self._perception_loop(duration_seconds, start_time))
        reasoning_task = asyncio.create_task(self._reasoning_loop(duration_seconds, start_time))
        action_task = asyncio.create_task(self._action_loop(duration_seconds, start_time))
        learning_task = asyncio.create_task(self._learning_loop(duration_seconds, start_time))
        
        # Start fast reactive loop if enabled
        tasks = [perception_task, reasoning_task, action_task, learning_task]
        if self.config.enable_fast_loop:
            fast_loop_task = asyncio.create_task(self._fast_reactive_loop(duration_seconds, start_time))
            tasks.append(fast_loop_task)
            print("[ASYNC] Fast reactive loop ENABLED")
        else:
            print("[ASYNC] Fast reactive loop DISABLED")
        
        # Wait for all tasks to complete (or any to fail)
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print("[ASYNC] All parallel loops completed")

    async def _perception_loop(self, duration_seconds: int, start_time: float):
        """
        Continuously perceive the game state and queue perceptions for reasoning.
        Uses adaptive throttling to prevent queue overflow.
        """
        print("[PERCEPTION] Loop started")
        print(f"[PERCEPTION] Qwen3-VL status: {'ENABLED' if self.perception_llm else 'DISABLED (None)'}")
        cycle_count = 0
        skip_count = 0
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                cycle_count += 1
                
                # Check queue status BEFORE perceiving to avoid wasted work
                queue_size = self.perception_queue.qsize()
                max_queue_size = self.perception_queue.maxsize
                
                # If queue is full, skip perception entirely and wait
                if queue_size >= max_queue_size:
                    skip_count += 1
                    if skip_count % 20 == 0:  # Only log every 20 skips
                        print(f"[PERCEPTION] Queue full, skipped {skip_count} cycles")
                    await asyncio.sleep(1.5)  # Shorter wait when queue is full
                    continue
                
                perception = await self.perception.perceive()
                game_state = perception.get('game_state')
                scene_type = perception.get('scene_type', SceneType.UNKNOWN)

                state_dict = game_state.to_dict() if game_state else {}
                scene_name = scene_type.value if hasattr(scene_type, 'value') else str(scene_type)

                # Update support subsystems with latest state
                if game_state:
                    self.smart_navigator.learn_location(
                        game_state.location_name,
                        {
                            'in_combat': game_state.in_combat,
                            'scene': scene_name,
                        }
                    )
                    self.character_progression.update_from_state(state_dict)
                    self.inventory_manager.update_from_state(state_dict)
                    self.analytics.update_state(state_dict)

                loop_settings = self.loop_manager.update_for_state(scene_name, state_dict)

                # Adaptive throttling based on queue fullness
                base_interval = loop_settings.perception_interval
                if queue_size >= max_queue_size * 0.8:
                    throttle_delay = max(1.0, base_interval * 3)
                    if cycle_count % 5 == 0:
                        print(f"[PERCEPTION] Queue {queue_size}/{max_queue_size} - heavy throttling ({throttle_delay:.2f}s)")
                elif queue_size >= max_queue_size * 0.6:
                    throttle_delay = max(0.5, base_interval * 2)
                    if cycle_count % 10 == 0:
                        print(f"[PERCEPTION] Queue {queue_size}/{max_queue_size} - moderate throttling ({throttle_delay:.2f}s)")
                elif queue_size >= max_queue_size * 0.4:
                    throttle_delay = max(0.3, base_interval * 1.5)
                else:
                    throttle_delay = base_interval
                
                # Enhance perception with Qwen3-VL using CLIP data (not raw images)
                if self.perception_llm and cycle_count % 2 == 0:  # Every 2nd cycle for faster analysis
                    print(f"[QWEN3-VL] Cycle {cycle_count}: Starting CLIP-based analysis...")
                    try:
                        print(f"[QWEN3-VL] DEBUG: Extracting perception data...")
                        print(f"[QWEN3-VL] DEBUG: game_state type: {type(game_state)}")
                        print(f"[QWEN3-VL] DEBUG: scene_type: {scene_type}")
                        objects = perception.get('objects', [])
                        print(f"[QWEN3-VL] DEBUG: objects count: {len(objects)}")
                        scene_probs = perception.get('scene_probs', {})
                        print(f"[QWEN3-VL] DEBUG: scene_probs: {scene_probs}")
                        
                        # Build rich context from CLIP analysis
                        print(f"[QWEN3-VL] DEBUG: Building context string...")
                        # Objects are tuples (label, confidence), not dicts
                        try:
                            objects_list = ', '.join([f"{obj[0]} ({obj[1]:.2f})" for obj in objects[:5]])
                        except Exception as e:
                            print(f"[QWEN3-VL] DEBUG: Failed to build objects_list: {e}")
                            objects_list = ''
                        print(f"[QWEN3-VL] DEBUG: objects_list: {objects_list}")
                        scene_confidence = max(scene_probs.values()) if scene_probs else 0.0
                        print(f"[QWEN3-VL] DEBUG: scene_confidence: {scene_confidence}")
                        
                        # Convert scene_type enum to string
                        scene_type_str = scene_type.value if hasattr(scene_type, 'value') else str(scene_type)
                        print(f"[QWEN3-VL] DEBUG: scene_type_str: {scene_type_str}")
                        
                        clip_context = f"""Analyze Skyrim gameplay based on CLIP visual perception:

CLIP Scene Classification:
- Scene type: {scene_type_str} (confidence: {scene_confidence:.2f})
- Detected objects: {objects_list if objects_list else 'none detected'}

Game State:
- Location: {game_state.location_name if game_state else 'unknown'}
- Health: {game_state.health if game_state else 100:.0f}/100
- In combat: {game_state.in_combat if game_state else False}
- Enemies nearby: {game_state.enemies_nearby if game_state else 0}
- NPCs nearby: {len(game_state.nearby_npcs) if game_state and game_state.nearby_npcs else 0}

Based on this visual and contextual data, provide:
1. Environment description and spatial awareness
2. Potential threats or opportunities
3. Recommended actions or focus areas
4. Strategic considerations"""
                        
                        print(f"[QWEN3-VL] Analyzing CLIP perception (cycle {cycle_count})...")
                        try:
                            # Add timeout to prevent hanging
                            visual_analysis = await asyncio.wait_for(
                                self.perception_llm.generate(
                                    prompt=clip_context,
                                    max_tokens=256
                                ),
                                timeout=20.0  # 20 second timeout (Qwen3-VL can be slow)
                            )
                            perception['visual_analysis'] = visual_analysis.get('content', '')
                            # Log every Qwen3-VL analysis (since it only runs every 2nd cycle anyway)
                            print(f"[QWEN3-VL] Analysis: {visual_analysis.get('content', '')[:150]}...")
                        except asyncio.TimeoutError:
                            print(f"[QWEN3-VL] Analysis timed out after 10s")
                            perception['visual_analysis'] = "[TIMEOUT] Visual analysis timed out"
                        
                    except Exception as e:
                        # Log errors but don't break the loop
                        if cycle_count % 30 == 0:
                            print(f"[QWEN3-VL] Analysis failed: {e}")
                            import traceback
                            traceback.print_exc()

                # Optional Gemini augmentation after CLIP/Qwen analysis
                gemini_summary = await self._augment_with_gemini(perception, cycle_count)
                if gemini_summary:
                    print(f"[GEMINI] Vision augment: {gemini_summary[:140]}...")
                
                self.current_perception = perception
                
                # Queue perception for reasoning (should succeed since we checked above)
                try:
                    self.perception_queue.put_nowait({
                        'perception': perception,
                        'cycle': cycle_count,
                        'timestamp': time.time()
                    })
                except asyncio.QueueFull:
                    # This shouldn't happen often since we check above
                    skip_count += 1
                
                # Wait before next perception (adaptive)
                await asyncio.sleep(throttle_delay)
                
            except Exception as e:
                print(f"[PERCEPTION] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print(f"[PERCEPTION] Loop ended (skipped {skip_count} cycles)")

    async def _reasoning_loop(self, duration_seconds: int, start_time: float):
        """
        Continuously process perceptions, compute consciousness, plan actions.
        Uses semaphore to limit concurrent LLM calls and prevent system overload.
        """
        print("[REASONING] Loop started")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Get next perception (wait if none available)
                perception_data = await asyncio.wait_for(
                    self.perception_queue.get(),
                    timeout=5.0
                )
                
                # Throttle reasoning to prevent overload
                time_since_last = time.time() - self.last_reasoning_time
                reasoning_throttle = self.loop_manager.get_interval('reasoning')
                if time_since_last < reasoning_throttle:
                    await asyncio.sleep(reasoning_throttle - time_since_last)
                
                self.last_reasoning_time = time.time()
                
                perception = perception_data['perception']
                cycle_count = perception_data['cycle']
                
                print(f"\n[REASONING] Processing cycle {cycle_count}")
                
                game_state = perception['game_state']
                scene_type = perception['scene_type']

                if game_state:
                    state_dict = game_state.to_dict()
                    scene_name = scene_type.value if hasattr(scene_type, 'value') else str(scene_type)
                    self.goal_planner.update_state(state_dict, scene_name)
                
                # Store perceptual memory
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
                
                # Compute world state and consciousness (consciousness runs in parallel, no semaphore)
                world_state = await self.agi.perceive({
                    'causal': game_state.to_dict(),
                    'visual': [perception['visual_embedding']],
                })
                
                consciousness_context = {
                    'motivation': 'unknown',
                    'cycle': cycle_count,
                    'scene': scene_type.value
                }
                # Consciousness bridge calls 2 big models in parallel - don't throttle it
                current_consciousness = await self.consciousness_bridge.compute_consciousness(
                    game_state.to_dict(),
                    consciousness_context
                )
                
                print(f"[REASONING] Coherence 𝒞 = {current_consciousness.coherence:.3f}")
                
                # Store consciousness
                self.last_consciousness = self.current_consciousness
                self.current_consciousness = current_consciousness
                
                # Assess motivation
                motivation_context = {
                    'uncertainty': 0.7 if scene_type == SceneType.UNKNOWN else 0.3,
                    'predicted_delta_coherence': 0.05,
                }
                mot_state = self.agi.motivation.compute_motivation(
                    state=game_state.to_dict(),
                    context=motivation_context
                )
                
                # Form/update goals
                if len(self.agi.goal_system.get_active_goals()) == 0 or cycle_count % 10 == 0:
                    goal = self.agi.goal_system.generate_goal(
                        mot_state.dominant_drive().value,
                        {'scene': scene_type.value, 'location': game_state.location_name}
                    )
                    self.current_goal = goal.description
                    print(f"[REASONING] New goal: {self.current_goal}")
                
                # Plan action (with LLM throttling)
                planning_start = time.time()
                async with self.llm_semaphore:
                    action = await self._plan_action(
                        perception=perception,
                        motivation=mot_state,
                        goal=self.current_goal
                    )
                planning_duration = time.time() - planning_start
                self.stats['planning_times'].append(planning_duration)
                
                # Handle None action with fallback
                if action is None:
                    print("[REASONING] WARNING: No action returned by _plan_action, using fallback")
                    action = 'explore'  # Safe default fallback
                    self.stats['heuristic_action_count'] += 1
                
                print(f"[REASONING] Planned action: {action} ({planning_duration:.3f}s)")
                
                # Queue action for execution (non-blocking)
                try:
                    self.action_queue.put_nowait({
                        'action': action,
                        'scene_type': scene_type,
                        'game_state': game_state,
                        'motivation': mot_state,
                        'cycle': cycle_count,
                        'consciousness': current_consciousness
                    })
                except asyncio.QueueFull:
                    print(f"[REASONING] Action queue full, action {action} dropped")
                
            except asyncio.TimeoutError:
                # No perception available, wait a bit
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[REASONING] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print("[REASONING] Loop ended")

    async def _action_loop(self, duration_seconds: int, start_time: float):
        """
        Continuously execute queued actions.
        """
        print("[ACTION] Loop started")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Get next action (wait if none available)
                action_data = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=2.0
                )
                
                action = action_data['action']
                scene_type = action_data['scene_type']
                game_state = action_data['game_state']
                
                print(f"\n[ACTION] Executing: {action}")
                
                # Execute action with timing
                execution_start = time.time()
                try:
                    await self._execute_action(action, scene_type)
                    execution_duration = time.time() - execution_start
                    self.stats['execution_times'].append(execution_duration)
                    self.stats['actions_taken'] += 1
                    self.stats['action_success_count'] += 1
                    print(f"[ACTION] Successfully executed: {action} ({execution_duration:.3f}s)")
                    
                    # Update stuck detection tracking
                    if self.current_consciousness:
                        coherence = self.current_consciousness.coherence
                        self._update_stuck_tracking(action, coherence)
                except Exception as e:
                    execution_duration = time.time() - execution_start
                    self.stats['execution_times'].append(execution_duration)
                    self.stats['action_failure_count'] += 1
                    print(f"[ACTION] Execution failed: {e}")
                    try:
                        await self.actions.look_around()
                        print("[ACTION] Performed fallback look_around")
                    except:
                        print("[ACTION] Even fallback action failed")
                
                # Queue for learning
                try:
                    self.learning_queue.put_nowait({
                        'action_data': action_data,
                        'execution_time': time.time()
                    })
                except asyncio.QueueFull:
                    print(f"[ACTION] Learning queue full")
                
                # Minimal pause to let action complete
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                # No action available, continue monitoring
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[ACTION] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print("[ACTION] Loop ended")

    async def _learning_loop(self, duration_seconds: int, start_time: float):
        """
        Continuously process completed actions and learn from outcomes.
        """
        print("[LEARNING] Loop started")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Get completed action data
                learning_data = await asyncio.wait_for(
                    self.learning_queue.get(),
                    timeout=5.0
                )
                
                action_data = learning_data['action_data']
                action = action_data['action']
                cycle_count = action_data['cycle']
                
                print(f"[LEARNING] Processing cycle {cycle_count}")
                
                # Perceive outcome
                after_perception = await self.perception.perceive()
                after_state = after_perception['game_state'].to_dict()
                
                # Use fast heuristic consciousness (skip slow LLM calls in learning loop)
                # The reasoning loop already computed consciousness, we just need a quick estimate
                from .skyrim_cognition import SkyrimCognitiveState
                cognitive_after = SkyrimCognitiveState.from_game_state(after_state)
                
                # Create simple consciousness state without LLM calls
                after_consciousness = ConsciousnessState(
                    coherence=cognitive_after.overall_quality * 0.5 + 0.1,  # Quick estimate
                    coherence_ontical=cognitive_after.survival * 0.4,
                    coherence_structural=cognitive_after.progression * 0.3,
                    coherence_participatory=cognitive_after.effectiveness * 0.5,
                    game_quality=cognitive_after.overall_quality,
                    consciousness_level=0.1,  # Minimal estimate
                    self_awareness=0.3  # Default
                )
                
                # Build before state
                before_state = action_data['game_state'].to_dict()
                before_state.update({
                    'scene': action_data['scene_type'].value,
                    'curiosity': action_data['motivation'].curiosity,
                    'competence': action_data['motivation'].competence,
                    'coherence': action_data['motivation'].coherence,
                    'autonomy': action_data['motivation'].autonomy
                })
                
                # Learn from experience
                self.skyrim_world.learn_from_experience(
                    action=str(action),
                    before_state=before_state,
                    after_state=after_state,
                    surprise_threshold=self.config.surprise_threshold
                )
                
                # Store RL experience (with consciousness)
                if self.rl_learner is not None:
                    self.rl_learner.store_experience(
                        state_before=before_state,
                        action=str(action),
                        state_after=after_state,
                        done=False,
                        consciousness_before=action_data['consciousness'],
                        consciousness_after=after_consciousness
                    )
                    
                    # Train periodically
                    if cycle_count % self.config.rl_train_freq == 0:
                        print(f"[LEARNING] Training RL at cycle {cycle_count}...")
                        self.rl_learner.train_step()
                
                # Update stats
                self.stats['cycles_completed'] = cycle_count
                self.stats['total_playtime'] = time.time() - start_time
                
                if after_consciousness:
                    if 'consciousness_coherence_history' not in self.stats:
                        self.stats['consciousness_coherence_history'] = []
                    self.stats['consciousness_coherence_history'].append(after_consciousness.coherence)
                
            except asyncio.TimeoutError:
                # No learning data available
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[LEARNING] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print("[LEARNING] Loop ended")

    async def _fast_reactive_loop(self, duration_seconds: int, start_time: float):
        """
        Fast reactive loop that executes every second with basic heuristics.
        
        This loop provides immediate responses to critical situations without
        waiting for the slower deliberative reasoning loop. It handles:
        - Emergency health situations (healing)
        - Immediate combat threats (blocking, dodging)
        - Quick environmental responses (falling, fire, traps)
        
        The fast loop uses simple heuristics and doesn't involve LLM calls,
        making it extremely responsive for survival-critical actions.
        """
        print("[FAST-LOOP] Fast reactive loop started")
        print(f"[FAST-LOOP] Interval: {self.config.fast_loop_interval}s")
        print(f"[FAST-LOOP] Health threshold: {self.config.fast_health_threshold}%")
        print(f"[FAST-LOOP] Danger threshold: {self.config.fast_danger_threshold} enemies")
        
        cycle_count = 0
        last_fast_action_time = 0
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                cycle_count += 1
                cycle_start = time.time()
                
                # Get current perception (non-blocking, use cached if available)
                if self.current_perception is None:
                    await asyncio.sleep(self.config.fast_loop_interval)
                    continue
                
                game_state = self.current_perception.get('game_state')
                if game_state is None:
                    await asyncio.sleep(self.config.fast_loop_interval)
                    continue
                
                scene_type = self.current_perception.get('scene_type', SceneType.UNKNOWN)
                
                # Skip fast actions in menus (they don't need reactive responses)
                if scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                    await asyncio.sleep(self.config.fast_loop_interval)
                    continue
                
                # === FAST HEURISTICS ===
                fast_action = None
                fast_reason = None
                priority = 0  # Higher = more urgent
                
                # 1. CRITICAL HEALTH - Highest priority
                if game_state.health < self.config.fast_health_threshold:
                    if game_state.health < 15:
                        # Extremely low health - immediate retreat
                        fast_action = 'retreat'
                        fast_reason = f"CRITICAL health {game_state.health:.0f}% - retreating"
                        priority = 100
                    elif game_state.magicka > 30:
                        # Try healing spell
                        fast_action = 'heal'
                        fast_reason = f"Low health {game_state.health:.0f}% - healing"
                        priority = 90
                    else:
                        # Block and back away
                        fast_action = 'block'
                        fast_reason = f"Low health {game_state.health:.0f}%, no magicka - blocking"
                        priority = 85
                
                # 2. OVERWHELMING COMBAT - High priority
                elif game_state.in_combat and game_state.enemies_nearby >= self.config.fast_danger_threshold:
                    if game_state.stamina > 40:
                        # Power attack to clear space
                        fast_action = 'power_attack'
                        fast_reason = f"Surrounded by {game_state.enemies_nearby} enemies - power attack"
                        priority = 70
                    else:
                        # Defensive stance
                        fast_action = 'block'
                        fast_reason = f"Surrounded by {game_state.enemies_nearby} enemies, low stamina - blocking"
                        priority = 65
                
                # 3. ACTIVE COMBAT - Medium priority
                elif game_state.in_combat and game_state.enemies_nearby > 0:
                    # Basic combat response
                    if game_state.stamina > 50:
                        fast_action = 'attack'
                        fast_reason = f"In combat with {game_state.enemies_nearby} enemies - attacking"
                        priority = 50
                    else:
                        fast_action = 'block'
                        fast_reason = f"In combat, low stamina - blocking to recover"
                        priority = 45
                
                # 4. LOW STAMINA OUT OF COMBAT - Low priority
                elif game_state.stamina < 20 and not game_state.in_combat:
                    fast_action = 'wait'
                    fast_reason = f"Low stamina {game_state.stamina:.0f}% - resting"
                    priority = 20
                
                # 5. RESOURCE MANAGEMENT - Very low priority
                elif game_state.health < 60 and game_state.magicka > 50 and not game_state.in_combat:
                    # Preventive healing when safe
                    fast_action = 'heal'
                    fast_reason = f"Safe healing opportunity - health {game_state.health:.0f}%"
                    priority = 10
                
                # Execute fast action if one was determined
                if fast_action and priority > 0:
                    # Throttle fast actions to avoid spam (minimum 2 seconds between actions)
                    time_since_last = time.time() - last_fast_action_time
                    if time_since_last < 2.0:
                        # Too soon, skip this action
                        await asyncio.sleep(self.config.fast_loop_interval)
                        continue
                    
                    # Log fast action
                    if cycle_count % 5 == 0 or priority >= 70:  # Log high priority always
                        print(f"\n[FAST-LOOP] Cycle {cycle_count} | Priority {priority}")
                        print(f"[FAST-LOOP] Action: {fast_action} | Reason: {fast_reason}")
                    
                    # Execute immediately (bypass normal action queue for critical actions)
                    execution_start = time.time()
                    try:
                        await self._execute_action(fast_action, scene_type)
                        execution_duration = time.time() - execution_start
                        
                        # Update statistics
                        self.stats['fast_action_count'] += 1
                        self.stats['fast_action_times'].append(execution_duration)
                        self.stats['actions_taken'] += 1
                        
                        last_fast_action_time = time.time()
                        
                        if cycle_count % 5 == 0 or priority >= 70:
                            print(f"[FAST-LOOP] Executed in {execution_duration:.3f}s")
                    
                    except Exception as e:
                        print(f"[FAST-LOOP] Execution failed: {e}")
                        # Don't crash the fast loop on errors
                
                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0.1, self.config.fast_loop_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"[FAST-LOOP] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print(f"[FAST-LOOP] Loop ended ({cycle_count} cycles, {self.stats['fast_action_count']} actions)")

    async def _autonomous_play_sequential(self, duration_seconds: int, start_time: float):
        """
        Sequential gameplay mode (original behavior).
        Kept for backwards compatibility and debugging.
        """
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
        motivation,
        goal: Optional[str] = None
    ) -> Optional[str]:
        try:
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
            
            # Check if we should force variety (prevent RL dominance)
            force_variety = self.consecutive_same_action >= 6  # Force variety after 6 same actions
            
            # Inject variety: Sometimes use heuristics even with RL to explore
            use_rl = random.random() > 0.1  # 90% RL, 10% heuristics for variety
            if not use_rl:
                print(f"[VARIETY] Random variety injection - using heuristics instead of RL")
            
            if force_variety and self.consecutive_same_action >= 6:
                print(f"[VARIETY] Forcing variety after {self.consecutive_same_action}x '{self.last_executed_action}' - skipping RL")
            
            # Use RL-based action selection if enabled (but not if forcing variety)
            if self.rl_learner is not None and use_rl:
                print("[PLANNING] Using RL-based action selection with LLM reasoning...")
                print(f"[PLANNING] RL reasoning neuron LLM status: {'Connected' if self.rl_reasoning_neuron.llm_interface else 'Using heuristics'}")
                
                # Track RL usage
                self.stats['rl_action_count'] += 1
                
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
                
                # RL reasoning neuron already connected to huihui
                
                # Start Huihui in background (don't wait)
                print("[PARALLEL] Starting Huihui in background, using fast heuristic for immediate action")
                
                # Get visual analysis from Qwen3-VL if available
                visual_analysis = perception.get('visual_analysis', '')
                if visual_analysis:
                    print(f"[QWEN3-VL] Passing visual analysis to Huihui: {visual_analysis[:100]}...")
                
                # Start Huihui task (runs in background, we don't await it)
                huihui_task = asyncio.create_task(
                    self.rl_reasoning_neuron.reason_about_q_values(
                        state=state_dict,
                        q_values=q_values,
                        available_actions=available_actions,
                        context={
                            'motivation': motivation.dominant_drive().value,
                            'terrain_type': self.skyrim_world.classify_terrain_from_scene(
                                scene_type.value,
                                game_state.in_combat
                            ),
                            'meta_strategy': meta_context,
                            'visual_analysis': visual_analysis  # Add Qwen3-VL's vision
                        }
                    )
                )
                
                # Compute fast heuristic immediately (no await needed)
                print("[HEURISTIC-FAST] Computing quick action for Phi-4...")
                
                # Quick Q-value based selection
                top_q_action = max(
                    [(a, q_values.get(a, 0.0)) for a in available_actions],
                    key=lambda x: x[1]
                )[0]
                
                # Context-aware adjustment
                if game_state.health < 30 and 'rest' in available_actions:
                    heuristic_action = 'rest'
                    heuristic_reason = "Low health emergency"
                elif game_state.in_combat and game_state.enemies_nearby > 2 and 'power_attack' in available_actions:
                    heuristic_action = 'power_attack'
                    heuristic_reason = "Multiple enemies"
                elif not game_state.in_combat and 'move_forward' in available_actions:
                    heuristic_action = 'move_forward'
                    heuristic_reason = "Safe exploration"
                else:
                    heuristic_action = top_q_action
                    heuristic_reason = f"Top Q-value action"
                
                print(f"[HEURISTIC-FAST] Quick recommendation: {heuristic_action} ({heuristic_reason})")
                print(f"[HUIHUI-BG] Strategic analysis running in background...")
                
                # Store heuristic for immediate use by Phi-4
                heuristic_recommendation = heuristic_action
                heuristic_reasoning = heuristic_reason
                
                # Store background task for later (optional: could check if done)
                huihui_background_task = huihui_task

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
            
            print(f"[META-STRATEGY] Evaluating layer switches (current: {current_layer})")
            
            # Combat situations - prioritize Combat layer if effective
            # Only engage in combat if there are actually enemies
            if (game_state.in_combat or scene_type == SceneType.COMBAT) and game_state.enemies_nearby > 0:
                if current_layer != "Combat":
                    combat_effectiveness = strategic_analysis['layer_effectiveness'].get('Combat', 0.5)
                    if combat_effectiveness > 0.6:
                        optimal_layer = "Combat"
                        print(f"[META-STRATEGY] Switching to Combat layer (effectiveness: {combat_effectiveness:.2f}, {game_state.enemies_nearby} enemies)")
                
                # Choose combat action based on context - only if enemies present
                if game_state.enemies_nearby > 2:
                    action = 'power_attack' if 'power_attack' in available_actions else 'attack'
                    if action in available_actions:
                        print(f"[META-STRATEGY] → {action} (multiple enemies)")
                        return action
                elif game_state.health < 50:
                    action = 'block' if 'block' in available_actions else 'attack'
                    if action in available_actions:
                        print(f"[META-STRATEGY] → {action} (defensive)")
                        return action
                elif 'attack' in available_actions:
                    print(f"[META-STRATEGY] → attack (engage enemy)")
                    return 'attack'
                # If no combat actions available, fall through

            # Low health - consider Menu layer for healing
            if game_state.health < 30:
                if current_layer != "Menu":
                    menu_effectiveness = strategic_analysis['layer_effectiveness'].get('Menu', 0.5)
                    if menu_effectiveness > 0.5:
                        optimal_layer = "Menu"
                        print(f"[META-STRATEGY] Switching to Menu layer for healing (health: {game_state.health:.0f})")
            
                if 'consume_item' in available_actions:
                    print(f"[META-STRATEGY] → Action: consume_item (critical health: {game_state.health:.0f})")
                    return 'consume_item'
                else:
                    print(f"[META-STRATEGY] → Action: rest (health recovery needed)")
                    return 'rest'

            # Stealth opportunities
            if (not game_state.in_combat and 
                len(game_state.nearby_npcs) > 0 and 
                motivation.dominant_drive().value == 'competence'):
                stealth_effectiveness = strategic_analysis['layer_effectiveness'].get('Stealth', 0.5)
                if stealth_effectiveness > 0.6 and current_layer != "Stealth":
                    optimal_layer = "Stealth"
                    print(f"[META-STRATEGY] Switching to Stealth layer (effectiveness: {stealth_effectiveness:.2f}, {len(game_state.nearby_npcs)} NPCs nearby)")

            # If we determined an optimal layer, suggest layer transition
            if optimal_layer and optimal_layer != current_layer:
                # Return a meta-action that will trigger layer switch
                print(f"[META-STRATEGY] → Action: switch_to_{optimal_layer.lower()}")
                return f'switch_to_{optimal_layer.lower()}'
            elif optimal_layer:
                print(f"[META-STRATEGY] Already in optimal layer: {current_layer}")

            # Always use Phi-4 for final action selection (fast, decisive)
            if self.action_planning_llm:
                print("[PLANNING] Using Phi-4 for final action selection...")
                try:
                    # Pass only fast heuristic to Phi-4 (Huihui still thinking in background)
                    huihui_context = None
                    if 'heuristic_recommendation' in locals():
                        huihui_context = {
                            'heuristic_recommendation': heuristic_recommendation,
                            'heuristic_reasoning': heuristic_reasoning,
                            'huihui_status': 'background' if 'huihui_background_task' in locals() else 'not_started'
                        }
                    
                    llm_action = await self._plan_action_with_llm(
                        perception, game_state, scene_type, current_layer, available_actions, 
                        strategic_analysis, motivation, huihui_context
                    )
                    if llm_action:
                        print(f"[PHI4] Final action: {llm_action}")
                        self.stats['llm_action_count'] += 1
                        return llm_action
                    else:
                        print("[PHI4] Phi-4 returned None, falling back to heuristics")
                except Exception as e:
                    print(f"[PHI4] Planning failed: {e}, using heuristics")
                    import traceback
                    traceback.print_exc()
            else:
                print("[PLANNING] Phi-4 not available, using heuristic planning...")
            
            # Track heuristic usage
            self.stats['heuristic_action_count'] += 1

            # Fallback: Action selection within current layer based on motivation
            # More intelligent heuristics with scene awareness
            dominant_drive = motivation.dominant_drive().value
            
            print(f"[HEURISTIC] Fallback planning (drive: {dominant_drive}, scene: {scene_type.value}, health: {game_state.health:.0f})")
            
            # Consider scene type for better context-aware decisions
            if scene_type == SceneType.COMBAT or game_state.in_combat:
                # In combat, prioritize survival
                if game_state.health < 40 and 'block' in available_actions:
                    print(f"[HEURISTIC] → block (defensive, low health)")
                    return 'block'  # Defensive when low health
                elif game_state.enemies_nearby > 2:
                    if 'power_attack' in available_actions:
                        print(f"[HEURISTIC] → power_attack (multiple enemies: {game_state.enemies_nearby})")
                        return 'power_attack'
                    elif 'attack' in available_actions:
                        print(f"[HEURISTIC] → attack (multiple enemies: {game_state.enemies_nearby})")
                        return 'attack'
                elif 'attack' in available_actions:
                    print(f"[HEURISTIC] → attack (standard engagement)")
                    return 'attack'
                # If no combat actions available, fall through to exploration
            
            # Scene-specific actions
            if scene_type in [SceneType.INDOOR_BUILDING, SceneType.INDOOR_DUNGEON]:
                # Indoor: prioritize interaction and careful exploration
                if 'activate' in available_actions and dominant_drive == 'curiosity':
                    print(f"[HEURISTIC] → activate (indoor curiosity)")
                    return 'activate'
                print(f"[HEURISTIC] → navigate (careful indoor movement)")
                return 'navigate'  # Careful indoor movement
            
            # STUCK DETECTION: Check if we're repeating the same action without progress
            is_stuck = self._detect_stuck()
            
            if is_stuck:
                print(f"[STUCK] Detected stuck state! Forcing recovery action...")
                # Force recovery actions when stuck
                recovery_actions = []
                if 'jump' in available_actions:
                    recovery_actions.append('jump')
                if game_state.health > 50:  # Only try risky moves if healthy
                    recovery_actions.extend(['turn_left', 'turn_right', 'move_backward'])
                
                if recovery_actions:
                    recovery = random.choice(recovery_actions)
                    print(f"[STUCK] → {recovery} (unstuck maneuver)")
                    self.consecutive_same_action = 0  # Reset counter
                    return recovery
            
            # Add variety to avoid repetitive behavior - humans don't just explore
            # Occasionally try to interact with objects (human-like curiosity)
            if random.random() < 0.15 and 'activate' in available_actions:
                print(f"[HEURISTIC] → activate (random curiosity)")
                return 'activate'
            
            # Occasionally look around (human-like awareness)
            if random.random() < 0.10:
                print(f"[HEURISTIC] → look_around (situational awareness)")
                return 'look_around'
            
            # Occasionally jump (human-like playfulness/testing)
            if random.random() < 0.08 and 'jump' in available_actions:
                print(f"[HEURISTIC] → jump (playful exploration)")
                return 'jump'
            
            # Motivation-based selection (for outdoor/general scenes)
            if dominant_drive == 'curiosity':
                # Curiosity: prioritize interaction and straight movement
                if 'activate' in available_actions and random.random() < 0.4:
                    print(f"[HEURISTIC] → activate (curiosity-driven interaction)")
                    return 'activate'  # Interact with world
                elif random.random() < 0.5:
                    print(f"[HEURISTIC] → move_forward (direct movement)")
                    return 'move_forward'  # Prefer straight movement
                print(f"[HEURISTIC] → explore (curiosity-driven exploration)")
                return 'explore'  # Less frequent now
            elif dominant_drive == 'competence':
                if 'power_attack' in available_actions and current_layer == "Combat":
                    print(f"[HEURISTIC] → power_attack (competence training)")
                    return 'power_attack'  # Practice advanced combat
                elif 'backstab' in available_actions and current_layer == "Stealth":
                    print(f"[HEURISTIC] → backstab (stealth practice)")
                    return 'backstab'  # Practice stealth
                elif random.random() < 0.2:
                    print(f"[HEURISTIC] → sneak (competence practice)")
                    return 'sneak'  # Practice stealth
                print(f"[HEURISTIC] → move_forward (competence through movement)")
                return 'move_forward'  # Prefer straight movement
            elif dominant_drive == 'coherence':
                # For coherence, prefer gentle exploration over rest unless critical
                if game_state.health < 30:
                    print(f"[HEURISTIC] → rest (coherence restoration, critical health)")
                    return 'rest'  # Only rest if low health
                elif random.random() < 0.4:
                    print(f"[HEURISTIC] → move_forward (coherent movement)")
                    return 'move_forward'
                print(f"[HEURISTIC] → move_forward (coherence through gentle movement)")
                return 'move_forward'  # Prefer straight movement
            else:  # autonomy or default
                # Add variety even for autonomy
                if random.random() < 0.2 and 'activate' in available_actions:
                    print(f"[HEURISTIC] → activate (autonomous interaction)")
                    return 'activate'
                elif random.random() < 0.6:
                    print(f"[HEURISTIC] → move_forward (autonomous movement)")
                    return 'move_forward'
                print(f"[HEURISTIC] → explore (autonomy/default)")
                return 'explore'  # Fallback only

        except Exception as e:
            print(f"[_plan_action] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _plan_action_with_llm(
        self,
        perception: Dict[str, Any],
        game_state,
        scene_type,
        current_layer: str,
        available_actions: list,
        strategic_analysis: dict,
        motivation,
        huihui_context: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Use Phi-4 for fast action selection, optionally informed by Huihui's strategic reasoning.
        
        Args:
            huihui_context: Optional dict with 'recommendation' and 'reasoning' from Huihui
        
        Returns:
            Action string if LLM planning succeeds, None otherwise
        """
        # Use dedicated action planning LLM if available, otherwise fallback to main
        llm_interface = self.action_planning_llm if self.action_planning_llm else (
            self.agi.consciousness_llm.llm_interface if hasattr(self.agi, 'consciousness_llm') 
            and self.agi.consciousness_llm and hasattr(self.agi.consciousness_llm, 'llm_interface') 
            else None
        )
        
        if not llm_interface:
            return None
        
        # Build compact context for fast mistral-nemo reasoning
        recommendations_section = ""
        if huihui_context:
            huihui_status = huihui_context.get('huihui_status', 'unknown')
            recommendations_section = f"""
FAST HEURISTIC ANALYSIS:
Recommended: {huihui_context.get('heuristic_recommendation', 'N/A')}
Reasoning: {huihui_context.get('heuristic_reasoning', 'N/A')}
(Huihui strategic analysis: {huihui_status})
"""
        
        context = f"""SKYRIM AGENT - QUICK ACTION DECISION

STATE: HP={game_state.health:.0f} MP={game_state.magicka:.0f} ST={game_state.stamina:.0f}
SCENE: {scene_type.value} | COMBAT: {game_state.in_combat} | LAYER: {current_layer}
LOCATION: {game_state.location_name}
DRIVE: {motivation.dominant_drive().value}

ACTIONS: {', '.join(available_actions[:8])}{recommendations_section}

TERRAIN STRATEGY:
- Indoor/Menu: interact, navigate exits
- Outdoor: explore forward, scan horizon  
- Combat: use terrain, power_attack if strong, block if weak
- Low HP: rest or switch_to_menu for healing

QUICK DECISION - Choose ONE action from available list:"""

        try:
            print("[MISTRAL-ACTION] Fast action planning with mistral-nemo...")
            
            # Use dedicated LLM interface directly for faster response
            if self.action_planning_llm:
                result = await self.action_planning_llm.generate(
                    prompt=context,
                    max_tokens=300  # Enough for reasoning + action selection
                )
                # Debug: Check what we got back
                print(f"[MISTRAL-ACTION] DEBUG - Result type: {type(result)}")
                print(f"[MISTRAL-ACTION] DEBUG - Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                
                # LMStudioClient.generate() returns a dict with 'content' key (not 'response')
                response = result.get('content', result.get('response', '')) if isinstance(result, dict) else str(result)
            else:
                # Fallback to base AGI consciousness LLM
                result = await self.agi.process(context)
                response = result.get('consciousness_response', {}).get('response', '')
            
            print(f"[MISTRAL-ACTION] Response ({len(response)} chars): {response[:200] if len(response) > 200 else response}")
        except Exception as e:
            print(f"[MISTRAL-ACTION] ERROR during LLM action planning: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Parse LLM response for action with improved extraction
        response_lower = response.lower().strip()
        
        # Try to extract action using multiple strategies
        
        # Strategy 1: Check for layer transition actions ONLY if they're in available_actions
        layer_transitions = {
            'switch_to_combat': ['switch_to_combat', 'switch to combat', 'combat layer'],
            'switch_to_menu': ['switch_to_menu', 'switch to menu', 'menu layer', 'open inventory'],
            'switch_to_stealth': ['switch_to_stealth', 'switch to stealth', 'stealth layer', 'sneak mode'],
            'switch_to_exploration': ['switch_to_exploration', 'switch to exploration', 'exploration layer']
        }
        
        for action, patterns in layer_transitions.items():
            # Only extract layer transitions if they're actually available
            if action in available_actions and any(pattern in response_lower for pattern in patterns):
                return action
        
        # Strategy 2: Look for exact matches with available actions
        for action in available_actions:
            # Check for exact word match (word boundaries)
            action_lower = action.lower()
            if f' {action_lower} ' in f' {response_lower} ' or response_lower.startswith(action_lower) or response_lower.endswith(action_lower):
                return action
        
        # Strategy 3: Check for partial matches with available actions
        for action in available_actions:
            if action.lower() in response_lower:
                return action
        
        # Strategy 4: Map common LLM responses to standard actions
        action_mappings = {
            'combat': ['fight', 'attack', 'battle', 'engage enemy', 'power attack', 'strike'],
            'explore': ['wander', 'search', 'look around', 'investigate', 'move forward', 'go ahead'],
            'interact': ['use', 'activate', 'open', 'talk', 'speak', 'examine'],
            'stealth': ['sneak', 'hide', 'crouch', 'stay quiet'],
            'rest': ['wait', 'sleep', 'heal', 'recover', 'take a break'],
            'navigate': ['walk', 'move', 'go', 'travel', 'head to']
        }
        
        for standard_action, synonyms in action_mappings.items():
            if any(synonym in response_lower for synonym in synonyms):
                # Only return if it makes sense in context
                if standard_action in available_actions or standard_action in ['combat', 'explore', 'interact', 'rest', 'navigate']:
                    return standard_action
        
        # Strategy 5: Extract quoted action if present
        import re
        quoted_match = re.search(r'["\']([^"\']+)["\']', response)
        if quoted_match:
            quoted_action = quoted_match.group(1).lower()
            if quoted_action in available_actions:
                return quoted_action
            # Check if quoted action is a standard action
            for standard_action in ['combat', 'explore', 'interact', 'stealth', 'rest', 'navigate']:
                if standard_action in quoted_action:
                    return standard_action
        
        # No match found
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



    async def _execute_action(self, action: str, scene_type: SceneType):
        """
        Execute planned action.

        Args:
            action: Action to execute
            scene_type: Current scene type
        """
        # Validate action
        if not action or not isinstance(action, str):
            print(f"[ACTION] Invalid action: {action}, using fallback")
            action = 'explore'
        
        # Handle menu interactions with learning ONLY when the action is menu-related
        menu_related_actions = ['open_inventory', 'open_map', 'open_magic', 'open_skills', 
                               'navigate_inventory', 'navigate_map', 'use_item', 'equip_item', 
                               'consume_item', 'favorite_item', 'exit_menu', 'exit']
        
        if scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE] and action in menu_related_actions:
            # We're in a menu AND trying to do menu actions - use menu learner
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
            
            if suggested_action and suggested_action in ['activate', 'navigate', 'exit', 'select', 'back']:
                print(f"[MENU] Using learned action: {suggested_action}")
                action = suggested_action
        else:
            # Not doing menu actions - ensure menu learner is exited
            if self.menu_learner.current_menu:
                self.menu_learner.exit_menu()
        
        # Sync action layer to context - ensure actions execute in correct layer
        if action in ('explore', 'navigate', 'quest_objective', 'practice', 'move_forward', 'move_backward', 
                      'move_left', 'move_right', 'jump', 'activate', 'look_around', 'turn_left', 'turn_right'):
            # Movement and exploration actions
            self.bindings.switch_to_exploration()
        elif action in ('combat', 'attack', 'power_attack', 'quick_attack', 'block', 'bash', 'dodge', 'retreat'):
            # Combat actions
            self.bindings.switch_to_combat()
        elif action in ('sneak', 'backstab'):
            # Stealth actions
            self.bindings.switch_to_stealth()
        elif action in ('interact', 'rest'):
            self.bindings.switch_to_exploration()
        elif action.startswith('switch_to_'):
            # Layer transitions handled below
            pass
        else:
            # Default to exploration for unknown actions
            if not scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                self.bindings.switch_to_exploration()

        # Execute actions with better variety and human-like behavior
        if action == 'explore':
            # Simple forward exploration to avoid circles
            print(f"[ACTION] Exploring forward")
            await self.actions.move_forward(duration=2.5)
            # Occasionally look around
            if random.random() < 0.3:
                await self.actions.look_around()
        elif action == 'combat':
            await self.actions.combat_sequence("Enemy")
        elif action in ('interact', 'activate'):
            # Look at target briefly before activating
            print(f"[ACTION] Interacting with object/NPC")
            await self.actions.execute(Action(ActionType.ACTIVATE, duration=0.3))
            await asyncio.sleep(0.5)  # Brief pause after activation
        elif action == 'navigate':
            await self.actions.move_forward(duration=2.0)
        elif action == 'rest':
            await self.actions.execute(Action(ActionType.WAIT, duration=1.0))
        elif action == 'practice':
            await self.actions.execute(Action(ActionType.ATTACK, duration=0.3))
        elif action == 'quest_objective':
            await self.actions.move_forward(duration=2.0)
        elif action == 'move_forward':
            await self.actions.move_forward(duration=1.5)
        elif action == 'jump':
            await self.actions.execute(Action(ActionType.JUMP, duration=0.2))
        elif action == 'sneak':
            await self.actions.execute(Action(ActionType.SNEAK, duration=0.5))
        elif action == 'attack':
            await self.actions.execute(Action(ActionType.ATTACK, duration=0.3))
        elif action in ('power_attack', 'quick_attack'):
            # Combat actions - hold attack longer for power attack
            await self.actions.execute(Action(ActionType.ATTACK, duration=0.8))
        elif action == 'block':
            # Block action - hold to block
            await self.actions.execute(Action(ActionType.BLOCK, duration=1.0))
        elif action in ('bash', 'shout'):
            # Special combat actions - use attack as fallback
            await self.actions.execute(Action(ActionType.ATTACK, duration=0.4))
        elif action == 'dodge':
            # Dodge by moving backward briefly
            await self.actions.move_backward(duration=0.5)
        elif action == 'retreat':
            # Retreat by moving backward
            await self.actions.move_backward(duration=1.5)
        elif action == 'heal':
            # Open inventory to heal (simplified)
            print("[ACTION] Attempting to heal (opening inventory)")
            await self.actions.execute(Action(ActionType.INVENTORY, duration=0.2))
            await asyncio.sleep(0.5)
        elif action == 'exit':
            # Exit menu/dialogue
            await self.actions.execute(Action(ActionType.BACK, duration=0.2))
        elif action in ('navigate_inventory', 'equip_item', 'consume_item', 'drop_item', 'favorite_item',
                        'navigate_map', 'navigate_magic', 'navigate_skills', 'use_item'):
            # Menu actions - navigate using D-pad or activate
            print(f"[ACTION] Menu action: {action}")
            await self.actions.execute(Action(ActionType.ACTIVATE, duration=0.2))
            await asyncio.sleep(0.3)
        elif action in ('open_inventory', 'open_map', 'open_magic', 'open_skills'):
            # Open specific menu
            print(f"[ACTION] Opening menu: {action}")
            await self.actions.execute(Action(ActionType.INVENTORY, duration=0.2))
            await asyncio.sleep(0.5)
        elif action == 'look_around':
            # Human-like looking behavior
            await self.actions.look_around()
        elif action == 'turn_left':
            # Recovery: turn left to avoid obstacles
            await self.actions.turn_left(duration=1.0)
        elif action == 'turn_right':
            # Recovery: turn right to avoid obstacles
            await self.actions.turn_right(duration=1.0)
        elif action == 'move_backward':
            # Recovery: back up from obstacles
            await self.actions.move_backward(duration=0.8)
        elif action.startswith('switch_to_'):
            # Handle layer transition actions
            target_layer = action.replace('switch_to_', '').title()
            print(f"[META-STRATEGY] Executing layer transition to {target_layer}")
            
            if target_layer == 'Combat':
                self.bindings.switch_to_combat()
                # Perform a combat-ready action
                await self.actions.execute(Action(ActionType.ATTACK, duration=0.3))
            elif target_layer == 'Menu':
                self.bindings.switch_to_menu()
                # Actually open the inventory
                print("[LAYER] Switching to Menu layer and opening inventory")
                await self.actions.execute(Action(ActionType.INVENTORY, duration=0.2))
                await asyncio.sleep(0.5)
            elif target_layer == 'Stealth':
                self.bindings.switch_to_stealth()
                # Enter sneak mode
                await self.actions.execute(Action(ActionType.SNEAK, duration=0.5))
            elif target_layer == 'Exploration':
                self.bindings.switch_to_exploration()
                # Continue exploration
                await self.actions.explore_with_waypoints(duration=2.0)
        else:
            # Fallback for unknown actions
            print(f"[ACTION] Unknown action '{action}', falling back to exploration")
            await self.actions.explore_with_waypoints(duration=2.0)

    def _detect_stuck(self) -> bool:
        """
        Detect if the AGI is stuck (repeating same action without progress).
        
        Returns:
            True if stuck, False otherwise
        """
        # Need at least a few actions to detect stuck
        if len(self.action_history) < self.stuck_detection_window:
            return False
        
        # Check if we're repeating the same action
        recent_actions = self.action_history[-self.stuck_detection_window:]
        if len(set(recent_actions)) == 1:  # All same action
            same_action = recent_actions[0]
            
            # Check if coherence is changing (sign of progress)
            if len(self.coherence_history) >= self.stuck_detection_window:
                recent_coherence = self.coherence_history[-self.stuck_detection_window:]
                coherence_change = max(recent_coherence) - min(recent_coherence)
                
                if coherence_change < self.stuck_threshold:
                    print(f"[STUCK] Repeating '{same_action}' {self.stuck_detection_window}x with minimal progress (Δ𝒞={coherence_change:.3f})")
                    return True
        
        # Check consecutive same action count
        if self.consecutive_same_action >= 8:  # 8+ same actions in a row
            print(f"[STUCK] Executed '{self.last_executed_action}' {self.consecutive_same_action} times consecutively")
            return True
        
        return False
    
    def _update_stuck_tracking(self, action: str, coherence: float):
        """Update stuck detection tracking."""
        # Track action history
        self.action_history.append(action)
        if len(self.action_history) > self.stuck_detection_window * 2:
            self.action_history.pop(0)
        
        # Track coherence history
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > self.stuck_detection_window * 2:
            self.coherence_history.pop(0)
        
        # Track consecutive same action
        if action == self.last_executed_action:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
        
        self.last_executed_action = action

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
        
        # Performance metrics
        success_rate = 0.0
        if self.stats['actions_taken'] > 0:
            success_rate = self.stats['action_success_count'] / self.stats['actions_taken']
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Action success rate: {success_rate:.1%}")
        print(f"  Successful actions: {self.stats['action_success_count']}")
        print(f"  Failed actions: {self.stats['action_failure_count']}")

        # Fast reactive loop stats
        if self.stats['fast_action_count'] > 0:
            print(f"\n⚡ Fast Reactive Loop:")
            print(f"  Fast actions taken: {self.stats['fast_action_count']}")
            if self.stats['actions_taken'] > 0:
                fast_ratio = 100 * self.stats['fast_action_count'] / self.stats['actions_taken']
                print(f"  Fast action ratio: {fast_ratio:.1f}%")
            if self.stats['fast_action_times']:
                avg_fast = sum(self.stats['fast_action_times']) / len(self.stats['fast_action_times'])
                print(f"  Avg fast action time: {avg_fast:.3f}s")

        # Planning method breakdown
        total_planning = self.stats['rl_action_count'] + self.stats['llm_action_count'] + self.stats['heuristic_action_count']
        if total_planning > 0:
            print(f"\n🧠 Planning Methods:")
            print(f"  RL-based: {self.stats['rl_action_count']} ({100*self.stats['rl_action_count']/total_planning:.1f}%)")
            print(f"  LLM-based: {self.stats['llm_action_count']} ({100*self.stats['llm_action_count']/total_planning:.1f}%)")
            print(f"  Heuristic: {self.stats['heuristic_action_count']} ({100*self.stats['heuristic_action_count']/total_planning:.1f}%)")
        
        # Timing metrics
        if self.stats['planning_times']:
            avg_planning = sum(self.stats['planning_times']) / len(self.stats['planning_times'])
            print(f"\n⏱️  Timing:")
            print(f"  Avg planning time: {avg_planning:.3f}s")
        if self.stats['execution_times']:
            avg_execution = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
            print(f"  Avg execution time: {avg_execution:.3f}s")
        
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
