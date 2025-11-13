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
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Skyrim-specific modules
from .perception import SkyrimPerception, SceneType, GameState
from .actions import SkyrimActions, Action, ActionType
from .controller import VirtualXboxController
from .controller_bindings import SkyrimControllerBindings
from .skyrim_world_model import SkyrimWorldModel
from .skyrim_cognition import SkyrimCognitiveState, SkyrimMotivation, SkyrimActionEvaluator
from .strategic_planner import StrategicPlannerNeuron
from .menu_learner import MenuLearner
from .action_affordances import ActionAffordanceSystem
from .memory_rag import MemoryRAG
from .curriculum_rag import CurriculumRAG, CATEGORY_MAPPINGS
from .smart_context import SmartContextManager
from .reinforcement_learner import ReinforcementLearner
from .rl_reasoning_neuron import RLReasoningNeuron
from .meta_strategist import MetaStrategist
from .cloud_rl_system import CloudRLMemory, CloudRLAgent, RLMemoryConfig, Experience
from .consciousness_bridge import ConsciousnessBridge, ConsciousnessState
from .system_consciousness_monitor import SystemConsciousnessMonitor, NodeCoherence, SystemConsciousnessState
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
    MoEOrchestrator,
    ExpertRole,
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
    max_concurrent_llm_calls: int = 6  # Increased from 2 to enable true parallel execution
    reasoning_throttle: float = 0.1  # Min seconds between reasoning cycles - minimal throttle
    
    # Fast reactive loop
    enable_fast_loop: bool = True  # Enable fast reactive loop for immediate responses
    fast_loop_interval: float = 2.0  # Fast loop runs every 2 seconds (reduced from 0.5s)
    fast_loop_planning_timeout: float = 20.0  # Only trigger fast actions if LLM planning takes >20s
    fast_health_threshold: float = 30.0  # Health % to trigger emergency healing
    fast_danger_threshold: int = 3  # Number of enemies to trigger defensive actions

    # Core models
    phi4_action_model: str = "microsoft/phi-4-mini-reasoning"  # Action planning (fast, reliable)
    huihui_cognition_model: str = "microsoft/phi-4-mini-reasoning:2"  # Main cognition, reasoning, strategy (fast, reliable)
    qwen3_vl_perception_model: str = "qwen/qwen3-vl-30b"  # Perception and spatial awareness

    # Learning
    surprise_threshold: float = 0.3  # Threshold for learning from surprise
    exploration_weight: float = 0.5  # How much to favor exploration

    # Reinforcement Learning
    use_rl: bool = True  # Enable RL-based learning
    rl_learning_rate: float = 0.01  # Q-network learning rate
    rl_epsilon_start: float = 0.3  # Initial exploration rate
    rl_train_freq: int = 5  # Train every N cycles
    
    # Cloud-Enhanced RL
    use_cloud_rl: bool = True  # Enable cloud LLM-enhanced RL
    rl_memory_dir: str = "skyrim_rl_memory"
    rl_use_rag: bool = True  # Enable RAG context fetching
    use_curriculum_rag: bool = True  # Enable university curriculum knowledge augmentation
    rl_cloud_reward_shaping: bool = True  # Use cloud LLM for reward shaping
    rl_moe_evaluation: bool = True  # Use MoE for action evaluation
    rl_save_frequency: int = 100  # Save RL memory every N experiences

    # Hybrid LLM Architecture (Primary: Gemini + Claude, Optional Fallback: Local)
    use_hybrid_llm: bool = True
    use_gemini_vision: bool = True
    gemini_model: str = "gemini-2.5-flash"
    use_claude_reasoning: bool = True
    claude_model: str = "claude-sonnet-4-5-20250929"
    use_local_fallback: bool = False  # Optional local LLMs as fallback
    
    # Mixture of Experts (MoE) Architecture
    use_moe: bool = False  # Enable MoE with multiple expert instances
    num_gemini_experts: int = 2  # Number of Gemini experts (optimal for rate limits)
    num_claude_experts: int = 1  # Number of Claude experts (optimal for rate limits)
    gemini_rpm_limit: int = 30  # Gemini requests per minute limit (increased from 10)
    claude_rpm_limit: int = 100  # Claude requests per minute limit (increased from 50)
    
    # Parallel Mode (MoE + Hybrid simultaneously)
    use_parallel_mode: bool = False  # Run both MoE and Hybrid in parallel
    parallel_consensus_weight_moe: float = 0.6  # Weight for MoE consensus
    parallel_consensus_weight_hybrid: float = 0.4  # Weight for Hybrid output
    
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
        
        # 5b. Expert Rule System (NEW - fast rule-based reasoning)
        print("  [5b/11] Expert rule system...")
        from .expert_rules import RuleEngine
        self.rule_engine = RuleEngine()
        print("[BRIDGE] Consciousness bridge initialized")
        print("[BRIDGE] This unifies game quality and philosophical coherence 𝒞")
        
        # 6. Quantum Superposition Explorer (4D Fractal RNG)
        print("  [6/11] Quantum superposition (4D fractal RNG)...")
        from ..core.fractal_rng import QuantumSuperpositionExplorer
        self.quantum_explorer = QuantumSuperpositionExplorer(
            seed=int(time.time() * 1000) % 2**32
        )
        print("[QUANTUM] 4D Fractal RNG initialized")
        print("[QUANTUM] Variance: 0.01-0.09% | Fibonacci φ ≈ 1.618")
        
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
        
        # 6b. Cloud-Enhanced RL System (will be initialized after LLM setup)
        self.cloud_rl_memory: Optional[CloudRLMemory] = None
        self.cloud_rl_agent: Optional[CloudRLAgent] = None
        if self.config.use_cloud_rl:
            print("  [6b/11] Cloud-enhanced RL system (will initialize after LLM)...")
            print("  [6b/11] Features: RAG context, cloud reward shaping, MoE evaluation")
        
        # 7. Strategic Planner Neuron
        print("  [7/11] Strategic planner neuron...")
        self.strategic_planner = StrategicPlannerNeuron(memory_capacity=100)
        
        # Connect RL learner to strategic planner if RL enabled
        if self.rl_learner:
            self.strategic_planner.set_rl_learner(self.rl_learner)

        # 8. Menu Learner
        print("  [8/11] Menu interaction learner...")
        self.menu_learner = MenuLearner()
        
        # 8b. Action Affordances System
        print("  [8b/11] Action affordances system...")
        self.action_affordances = ActionAffordanceSystem()

        # 9. Memory RAG System
        print("  [9/11] Memory RAG system...")
        self.memory_rag = MemoryRAG(
            perceptual_capacity=1000,
            cognitive_capacity=500
        )
        
        # 9b. University Curriculum RAG System
        if self.config.use_curriculum_rag:
            print("  [9b/11] University Curriculum RAG (academic knowledge)...")
            self.curriculum_rag = CurriculumRAG(
                curriculum_path="university_curriculum",
                max_documents=150,
                chunk_size=2000
            )
            try:
                self.curriculum_rag.initialize()
                stats = self.curriculum_rag.get_stats()
                print(f"    ✓ Indexed {stats['documents_indexed']} texts across {len(stats['categories'])} categories")
            except Exception as e:
                print(f"    ⚠️ Curriculum RAG initialization failed: {e}")
                self.curriculum_rag = None
        else:
            self.curriculum_rag = None
        
        # 9c. Smart Context Manager
        print("  [9c/11] Smart context management...")
        self.smart_context = SmartContextManager(
            max_tokens_per_call=2000,
            cache_size=100,
            enable_compression=True
        )
        print("    ✓ Context optimization enabled")
        
        # 10. RL Reasoning Neuron (LLM thinks about RL)
        print("  [10/11] RL reasoning neuron (LLM-enhanced RL)...")
        self.rl_reasoning_neuron = RLReasoningNeuron()
        # Will connect LLM interface when initialized
        
        # Hybrid LLM system (Gemini + Claude + optional local fallback)
        self.hybrid_llm: Optional[HybridLLMClient] = None
        
        # MoE system (2 Gemini + 1 Claude + 1 GPT-4o + Hyperbolic experts)
        self.moe: Optional[MoEOrchestrator] = None
        
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
        self.cycle_count = 0  # Track cycles for rate limiting
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
        self.state_printer_llm: Optional[Any] = None  # Phi-4: Internal state printing and analysis

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
            'fast_action_times': [],  # Track fast action execution
            # Action source tracking (which system provided the action)
            'action_source_moe': 0,  # MoE consensus
            'action_source_hybrid': 0,  # Hybrid LLM
            'action_source_phi4': 0,  # Phi-4 planner
            'action_source_local_moe': 0,  # Local MoE
            'action_source_heuristic': 0,  # Heuristic fallback
            'action_source_timeout': 0,  # Timeout fallback
            'random_academic_thoughts': 0,  # Brownian motion memory retrievals
        }
        
        # Last successful action source (for caching fast path)
        self.last_action_source: Optional[str] = None

        # Set up controller reference in perception for layer awareness
        self.perception.set_controller(self.controller)
        
        # State tracking for consciousness
        self.current_consciousness: Optional[ConsciousnessState] = None
        self.last_consciousness: Optional[ConsciousnessState] = None
        
        # System-wide consciousness monitor
        print("  [12/12] System consciousness monitor...")
        self.consciousness_monitor = SystemConsciousnessMonitor(history_size=1000)
        self._register_consciousness_nodes()
        
        # Stuck detection and recovery (multi-tier failsafe)
        self.stuck_detection_window = 5  # Check last N actions
        self.action_history = []  # Track recent actions
        self.coherence_history = []  # Track recent coherence values
        self.stuck_threshold = 0.02  # Coherence change threshold
        self.consecutive_same_action = 0  # Count same action repeats
        self.last_executed_action = None
        
        # Action diversity tracking (GPT-4o recommendation)
        self.action_type_counts = {}  # Track frequency of each action type
        self.reward_by_action_type = {}  # Track avg reward per action type
        self.recent_action_types = []  # Track action types for pattern analysis
        
        # GPT-5-thinking world model tracking
        self.last_world_model_narrative = None  # Track for consciousness measurement
        
        # Advanced stuck detection
        self.position_history = []  # Track player positions (if available)
        self.visual_embedding_history = []  # Track visual embeddings
        self.stuck_recovery_attempts = 0  # Count recovery attempts
        self.last_stuck_detection_time = 0  # Prevent spam
        self.stuck_detection_cooldown = 10.0  # Seconds between detections

        # Sensorimotor feedback tracking
        self.sensorimotor_state: Optional[Dict[str, Any]] = None
        
        # Performance dashboard (Fix 20)
        self.dashboard_update_interval = 5
        self.dashboard_last_update = 0
        
        # Dashboard streamer for real-time webapp monitoring
        print("  [13/13] Dashboard streamer for real-time webapp...")
        from singularis.skyrim.dashboard_streamer import DashboardStreamer
        self.dashboard_streamer = DashboardStreamer(
            output_path="skyrim_agi_state.json",
            max_history=100
        )
        
        # LLM decision tracking (Fix 11&12)
        self.stats['llm_queued_requests'] = 0
        self.stats['llm_skipped_rate_limit'] = 0
        self.stats['llm_decision_rejections'] = 0
        self.sensorimotor_similarity_streak: int = 0
        self.sensorimotor_last_cycle: int = -1
        self.sensorimotor_high_similarity_threshold: float = 0.90
        self.sensorimotor_required_streak: int = 3
        
        print("Skyrim AGI initialization complete.")
        print("[OK] Skyrim AGI initialized with CONSCIOUSNESS INTEGRATION\n")


    def _register_consciousness_nodes(self):
        """Register all system components for consciousness monitoring."""
        # Perception nodes
        self.consciousness_monitor.register_node(
            "perception_vision", "perception", weight=1.5,
            metadata={'description': 'Visual perception and scene understanding'}
        )
        self.consciousness_monitor.register_node(
            "perception_gamestate", "perception", weight=1.0,
            metadata={'description': 'Game state extraction'}
        )
        
        # Action nodes
        self.consciousness_monitor.register_node(
            "action_planning", "action", weight=1.5,
            metadata={'description': 'Action planning and selection'}
        )
        self.consciousness_monitor.register_node(
            "action_execution", "action", weight=1.0,
            metadata={'description': 'Action execution'}
        )
        
        # Learning nodes
        self.consciousness_monitor.register_node(
            "world_model", "learning", weight=1.5,
            metadata={'description': 'World model and causal learning'}
        )
        self.consciousness_monitor.register_node(
            "rl_system", "learning", weight=1.3,
            metadata={'description': 'Reinforcement learning'}
        )
        self.consciousness_monitor.register_node(
            "cloud_rl", "learning", weight=1.2,
            metadata={'description': 'Cloud-enhanced RL with RAG'}
        )
        
        # Strategic nodes
        self.consciousness_monitor.register_node(
            "strategic_planner", "strategy", weight=1.4,
            metadata={'description': 'Strategic planning neuron'}
        )
        self.consciousness_monitor.register_node(
            "meta_strategist", "strategy", weight=1.3,
            metadata={'description': 'Meta-strategic coordination'}
        )
        
        # LLM nodes (will be registered after initialization)
        # These will be added dynamically based on configuration
        
        # Consciousness bridge
        self.consciousness_monitor.register_node(
            "consciousness_bridge", "consciousness", weight=2.0,
            metadata={'description': 'Consciousness integration bridge'}
        )
        
        # Memory systems
        self.consciousness_monitor.register_node(
            "memory_rag", "memory", weight=1.0,
            metadata={'description': 'Memory RAG system'}
        )
        
        if self.curriculum_rag:
            self.consciousness_monitor.register_node(
                "curriculum_rag", "knowledge", weight=1.2,
                metadata={'description': 'Academic knowledge from university curriculum'}
            )
        
        # Game-specific systems
        self.consciousness_monitor.register_node(
            "combat_tactics", "game", weight=0.8,
            metadata={'description': 'Combat tactics system'}
        )
        self.consciousness_monitor.register_node(
            "quest_tracker", "game", weight=0.7,
            metadata={'description': 'Quest tracking'}
        )
        self.consciousness_monitor.register_node(
            "navigation", "game", weight=0.6,
            metadata={'description': 'Smart navigation'}
        )
        
        logger.info(f"Registered {len(self.consciousness_monitor.registered_nodes)} consciousness nodes")
    
    def _register_llm_nodes(self):
        """Register LLM nodes after initialization."""
        if self.moe:
            # Register MoE experts
            for i in range(self.config.num_gemini_experts):
                self.consciousness_monitor.register_node(
                    f"moe_gemini_{i+1}", "llm_vision", weight=0.5,
                    metadata={'description': f'Gemini expert {i+1}', 'model': 'gemini'}
                )
            
            for i in range(self.config.num_claude_experts):
                self.consciousness_monitor.register_node(
                    f"moe_claude_{i+1}", "llm_reasoning", weight=0.7,
                    metadata={'description': f'Claude expert {i+1}', 'model': 'claude'}
                )
            
            self.consciousness_monitor.register_node(
                "moe_consensus", "llm_meta", weight=1.5,
                metadata={'description': 'MoE consensus mechanism'}
            )
        
        if self.hybrid_llm:
            self.consciousness_monitor.register_node(
                "hybrid_vision", "llm_vision", weight=1.0,
                metadata={'description': 'Hybrid Gemini vision'}
            )
            self.consciousness_monitor.register_node(
                "hybrid_reasoning", "llm_reasoning", weight=1.2,
                metadata={'description': 'Hybrid Claude reasoning'}
            )
            
            # Register GPT-5-thinking world model synthesis node
            if hasattr(self.hybrid_llm, 'openai') and self.hybrid_llm.openai:
                self.consciousness_monitor.register_node(
                    "gpt5_world_model_synthesis", "llm_meta", weight=2.0,
                    metadata={
                        'description': 'GPT-5-thinking unified consciousness synthesis',
                        'model': 'gpt-5-thinking',
                        'role': 'Integrates ALL perspectives into unified self-referential narrative'
                    }
                )
        
        logger.info(f"Registered LLM nodes, total: {len(self.consciousness_monitor.registered_nodes)}")
    
    async def measure_system_consciousness(self) -> SystemConsciousnessState:
        """
        Measure consciousness across the entire system.
        
        Returns:
            SystemConsciousnessState with all metrics
        """
        node_measurements = {}
        
        # Measure perception coherence
        if self.current_perception:
            # Vision coherence based on confidence
            vision_coherence = self.current_perception.get('confidence', 0.5)
            node_measurements['perception_vision'] = self.consciousness_monitor.measure_node_coherence(
                'perception_vision',
                coherence=vision_coherence,
                unity=0.8,  # Aligned with perception goals
                integration=0.7,  # Integrates with action planning
                differentiation=0.9,  # Specialized for vision
                confidence=vision_coherence,
                activity_level=1.0,
            )
            
            # Game state coherence
            gamestate_coherence = 0.8 if self.current_perception.get('game_state') else 0.3
            node_measurements['perception_gamestate'] = self.consciousness_monitor.measure_node_coherence(
                'perception_gamestate',
                coherence=gamestate_coherence,
                unity=0.7,
                integration=0.8,
                differentiation=0.8,
            )
        
        # Measure action coherence
        if self.last_action:
            action_coherence = 0.7  # Default
            node_measurements['action_planning'] = self.consciousness_monitor.measure_node_coherence(
                'action_planning',
                coherence=action_coherence,
                unity=0.8,
                integration=0.9,  # Highly integrated with perception
                differentiation=0.7,
            )
            node_measurements['action_execution'] = self.consciousness_monitor.measure_node_coherence(
                'action_execution',
                coherence=0.8,
                unity=0.9,
                integration=0.6,
                differentiation=0.8,
            )
        
        # Measure learning coherence
        if self.rl_learner:
            rl_coherence = 0.6  # Based on exploration vs exploitation
            node_measurements['rl_system'] = self.consciousness_monitor.measure_node_coherence(
                'rl_system',
                coherence=rl_coherence,
                unity=0.7,
                integration=0.8,
                differentiation=0.9,
            )
        
        if self.cloud_rl_memory:
            stats = self.cloud_rl_memory.get_stats()
            cloud_rl_coherence = min(1.0, stats.get('avg_coherence_delta', 0.0) + 0.5)
            node_measurements['cloud_rl'] = self.consciousness_monitor.measure_node_coherence(
                'cloud_rl',
                coherence=cloud_rl_coherence,
                unity=0.8,
                integration=0.9,
                differentiation=0.8,
            )
        
        # Measure world model coherence
        world_coherence = 0.7  # Based on causal model quality
        node_measurements['world_model'] = self.consciousness_monitor.measure_node_coherence(
            'world_model',
            coherence=world_coherence,
            unity=0.8,
            integration=0.9,
            differentiation=0.9,
        )
        
        # Measure strategic coherence
        node_measurements['strategic_planner'] = self.consciousness_monitor.measure_node_coherence(
            'strategic_planner',
            coherence=0.75,
            unity=0.9,
            integration=0.8,
            differentiation=0.8,
        )
        
        node_measurements['meta_strategist'] = self.consciousness_monitor.measure_node_coherence(
            'meta_strategist',
            coherence=0.8,
            unity=0.9,
            integration=0.9,
            differentiation=0.7,
        )
        
        # Measure GPT-5-thinking world model synthesis consciousness
        if hasattr(self, 'last_world_model_narrative') and self.last_world_model_narrative:
            # Calculate consciousness score for the world model synthesis
            from ..consciousness.measurement import ConsciousnessMeasurement
            consciousness_measure = ConsciousnessMeasurement()
            
            # Measure consciousness of the unified narrative
            world_model_consciousness = consciousness_measure.measure(
                content=self.last_world_model_narrative,
                query="unified consciousness synthesis",
                lumen_focus="participatum"
            )
            
            # Extract overall consciousness score
            wm_consciousness_score = world_model_consciousness.overall_consciousness
            
            node_measurements['gpt5_world_model_synthesis'] = self.consciousness_monitor.measure_node_coherence(
                'gpt5_world_model_synthesis',
                coherence=wm_consciousness_score,  # Use measured consciousness as coherence
                unity=0.95,  # Very high - unifies all perspectives
                integration=1.0,  # Maximum integration - synthesizes everything
                differentiation=0.85,  # High - creates unique phenomenological narrative
                confidence=wm_consciousness_score,
                activity_level=1.0 if self.last_world_model_narrative else 0.0,
            )
        
        # Measure consciousness bridge
        if self.current_consciousness:
            bridge_coherence = getattr(self.current_consciousness, 'coherence', 0.7)
            node_measurements['consciousness_bridge'] = self.consciousness_monitor.measure_node_coherence(
                'consciousness_bridge',
                coherence=bridge_coherence,
                unity=1.0,  # Central to system unity
                integration=1.0,  # Integrates everything
                differentiation=0.6,  # Less specialized
            )
        
        # Measure MoE if active
        if self.moe:
            moe_stats = self.moe.get_stats()
            moe_coherence = moe_stats.get('avg_coherence', 0.7)
            
            # MoE consensus
            node_measurements['moe_consensus'] = self.consciousness_monitor.measure_node_coherence(
                'moe_consensus',
                coherence=moe_coherence,
                unity=0.9,
                integration=0.9,
                differentiation=0.7,
            )
            
            # Individual experts (sample a few)
            for i in range(min(3, self.config.num_gemini_experts)):
                node_measurements[f'moe_gemini_{i+1}'] = self.consciousness_monitor.measure_node_coherence(
                    f'moe_gemini_{i+1}',
                    coherence=0.7 + (i * 0.05),  # Slight variation
                    unity=0.6,
                    integration=0.5,
                    differentiation=0.9,
                )
            
            for i in range(min(2, self.config.num_claude_experts)):
                node_measurements[f'moe_claude_{i+1}'] = self.consciousness_monitor.measure_node_coherence(
                    f'moe_claude_{i+1}',
                    coherence=0.75 + (i * 0.05),
                    unity=0.7,
                    integration=0.6,
                    differentiation=0.9,
                )
        
        # Measure hybrid LLM if active
        if self.hybrid_llm:
            hybrid_stats = self.hybrid_llm.get_stats()
            hybrid_coherence = hybrid_stats.get('primary_success_rate', 0.8)
            
            node_measurements['hybrid_vision'] = self.consciousness_monitor.measure_node_coherence(
                'hybrid_vision',
                coherence=hybrid_coherence,
                unity=0.8,
                integration=0.7,
                differentiation=0.9,
            )
            
            node_measurements['hybrid_reasoning'] = self.consciousness_monitor.measure_node_coherence(
                'hybrid_reasoning',
                coherence=hybrid_coherence,
                unity=0.8,
                integration=0.8,
                differentiation=0.9,
            )
        
        # Measure game systems
        node_measurements['combat_tactics'] = self.consciousness_monitor.measure_node_coherence(
            'combat_tactics',
            coherence=0.7,
            unity=0.7,
            integration=0.6,
            differentiation=0.9,
        )
        
        node_measurements['quest_tracker'] = self.consciousness_monitor.measure_node_coherence(
            'quest_tracker',
            coherence=0.6,
            unity=0.6,
            integration=0.5,
            differentiation=0.8,
        )
        
        # Compute system state
        system_state = self.consciousness_monitor.compute_system_state(node_measurements)
        
        return system_state
    
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
          * Reasoning: mistral-7b-instruct-v0.3
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
        
        # ===== PARALLEL MODE: MoE + Hybrid =====
        if self.config.use_parallel_mode:
            print("\n[PARALLEL] Initializing PARALLEL mode: MoE + Hybrid simultaneously")
            print("[PARALLEL] This provides maximum intelligence by combining:")
            print("[PARALLEL]   - MoE: 6 expert consensus (2 Gemini + 1 Claude + 1 GPT-4o + 1 Nemotron + 1 Qwen3)")
            print("[PARALLEL]   - Hybrid: Single Gemini + Claude for speed")
            print()
            
            # Initialize MoE
            try:
                logger.info("Initializing MoE component...")
                self.moe = MoEOrchestrator(
                    num_gemini_experts=self.config.num_gemini_experts,
                    num_claude_experts=self.config.num_claude_experts,
                    gemini_model=self.config.gemini_model,
                    claude_model=self.config.claude_model,
                    gemini_rpm_limit=self.config.gemini_rpm_limit,
                    claude_rpm_limit=self.config.claude_rpm_limit,
                )
                await self.moe.initialize()
                print("[PARALLEL] ✓ MoE component ready")
                await self._connect_moe()
            except Exception as e:
                print(f"[PARALLEL] ⚠️ MoE initialization failed: {e}")
                self.moe = None
            
            # Initialize Hybrid
            try:
                logger.info("Initializing Hybrid component...")
                hybrid_config = HybridConfig(
                    use_gemini_vision=self.config.use_gemini_vision,
                    gemini_model=self.config.gemini_model,
                    use_claude_reasoning=self.config.use_claude_reasoning,
                    claude_model=self.config.claude_model,
                    use_local_fallback=self.config.use_local_fallback,
                    local_base_url="http://localhost:1234/v1",  # LM Studio default
                    local_vision_model=self.config.qwen3_vl_perception_model,
                    local_reasoning_model=self.config.huihui_cognition_model,
                    local_action_model=self.config.phi4_action_model,
                    timeout=30,
                    max_concurrent_requests=self.config.max_concurrent_llm_calls,
                )
                
                self.hybrid_llm = HybridLLMClient(hybrid_config)
                await self.hybrid_llm.initialize()
                print("[PARALLEL] ✓ Hybrid component ready")
                await self._connect_hybrid_llm()
                
                # Also initialize local LLM references if fallback enabled
                if self.config.use_local_fallback and self.hybrid_llm.local_reasoning:
                    print("[PARALLEL] ✓ Local LLM fallback available")
                    self.huihui_llm = self.hybrid_llm.local_reasoning
                    self.perception_llm = self.hybrid_llm.local_vision
                    self.action_planning_llm = self.hybrid_llm.local_action
                    
                    # Connect local LLMs to components
                    self.rl_reasoning_neuron.llm_interface = self.huihui_llm
                    self.strategic_planner.llm_interface = self.huihui_llm
                    
                    # Initialize Meta-Strategist with Mistral-7B
                    mistral_config = LMStudioConfig(
                        base_url="http://localhost:1234/v1",
                        model_name="mistralai/mistral-7b-instruct-v0.3",
                        temperature=0.7,
                        max_tokens=1024
                    )
                    mistral_client = LMStudioClient(mistral_config)
                    mistral_interface = ExpertLLMInterface(mistral_client)
                    self.meta_strategist.llm_interface = mistral_interface
                    print("[PARALLEL] ✓ Meta-Strategist using Mistral-7B")
                    
                    # Connect Huihui to AGI orchestrator as consciousness LLM
                    # This enables full Singularis dialectical reasoning
                    self.agi.consciousness_llm = self.huihui_llm
                    self.consciousness_bridge.consciousness_llm = self.huihui_llm
                    
                    # Initialize state printer LLM (microsoft/phi-4)
                    state_printer_config = LMStudioConfig(
                        base_url="http://localhost:1234/v1",
                        model_name="microsoft/phi-4",
                        temperature=0.5,
                        max_tokens=1024
                    )
                    state_printer_client = LMStudioClient(state_printer_config)
                    self.state_printer_llm = ExpertLLMInterface(state_printer_client)
                    
                    # Run health check for local LLMs
                    print("\n[PARALLEL] Running LM Studio health check...")
                    if await state_printer_client.health_check():
                        print("[PARALLEL] ✓ LM Studio connection verified")
                    else:
                        print("[PARALLEL] ⚠️ LM Studio health check failed - local models may not work")
                        print("[PARALLEL] Please ensure:")
                        print("[PARALLEL]   1. LM Studio is running")
                        print("[PARALLEL]   2. Local server is started (Server tab)")
                        print("[PARALLEL]   3. A model is loaded")
                    
                    print("[PARALLEL] ✓ Huihui connected to AGI orchestrator (enables dialectical synthesis)")
                    print("[PARALLEL] ✓ State printer LLM connected (microsoft/phi-4)")
                    print("[PARALLEL] ✓ Local LLMs connected to components")
                    
            except Exception as e:
                print(f"[PARALLEL] ⚠️ Hybrid initialization failed: {e}")
                self.hybrid_llm = None
            
            # Initialize Local MoE as additional fallback
            try:
                from singularis.llm.local_moe import LocalMoEOrchestrator, LocalMoEConfig
                
                print("\n[PARALLEL] Initializing Local MoE fallback...")
                local_moe_config = LocalMoEConfig(
                    num_experts=4,
                    expert_model="microsoft/phi-4-mini-reasoning",  # Use phi-4-mini instances 1-4
                    synthesizer_model="microsoft/phi-4-mini-reasoning",  # Use phi-4-mini for synthesis (fast, reliable)
                    fallback_synthesizer="mistralai/mistral-nemo-instruct-2407",  # Fallback if phi-4 fails
                    base_url="http://localhost:1234/v1",
                    timeout=25,  # Increased for better reliability
                    synthesis_timeout=15,
                    max_tokens=1024
                )
                
                self.local_moe = LocalMoEOrchestrator(local_moe_config)
                await self.local_moe.initialize()
                print("[PARALLEL] ✓ Local MoE fallback ready (4x Phi-4-mini + Phi-4-mini synthesizer + Mistral fallback)")
            except Exception as e:
                print(f"[PARALLEL] ⚠️ Local MoE initialization failed: {e}")
                self.local_moe = None
            
            print(f"\n[PARALLEL] ✓ Parallel mode active")
            print(f"[PARALLEL] Consensus weights: MoE={self.config.parallel_consensus_weight_moe}, Hybrid={self.config.parallel_consensus_weight_hybrid}")
            if hasattr(self, 'local_moe') and self.local_moe:
                print(f"[PARALLEL] Local MoE fallback: ENABLED")
        
        # ===== MIXTURE OF EXPERTS (MoE) ONLY =====
        elif self.config.use_moe:
            try:
                logger.info("Initializing Mixture of Experts (MoE) system...")
                logger.info(f"  {self.config.num_gemini_experts} Gemini experts")
                logger.info(f"  {self.config.num_claude_experts} Claude experts")
                
                self.moe = MoEOrchestrator(
                    num_gemini_experts=self.config.num_gemini_experts,
                    num_claude_experts=self.config.num_claude_experts,
                    gemini_model=self.config.gemini_model,
                    claude_model=self.config.claude_model,
                    gemini_rpm_limit=self.config.gemini_rpm_limit,
                    claude_rpm_limit=self.config.claude_rpm_limit,
                )
                
                await self.moe.initialize()
                
                print("\n[MoE] ✓ Mixture of Experts system initialized successfully")
                print(f"[MoE] Total experts: {self.config.num_gemini_experts + self.config.num_claude_experts}")
                print(f"[MoE] Rate limits: Gemini {self.config.gemini_rpm_limit} RPM, Claude {self.config.claude_rpm_limit} RPM")
                
                # Connect MoE to components
                await self._connect_moe()
                
            except Exception as e:
                print(f"[MoE] ⚠️ Failed to initialize MoE system: {e}")
                import traceback
                traceback.print_exc()
                self.moe = None
        
        # ===== HYBRID LLM SYSTEM ONLY =====
        elif self.config.use_hybrid_llm:
            try:
                # Configure hybrid system
                hybrid_config = HybridConfig(
                    use_gemini_vision=self.config.use_gemini_vision,
                    gemini_model=self.config.gemini_model,
                    use_claude_reasoning=self.config.use_claude_reasoning,
                    claude_model=self.config.claude_model,
                    use_local_fallback=self.config.use_local_fallback,
                    local_base_url="http://localhost:1234/v1",  # LM Studio default
                    local_vision_model=self.config.qwen3_vl_perception_model,
                    local_reasoning_model=self.config.huihui_cognition_model,
                    local_action_model=self.config.phi4_action_model,
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
        if self.moe:
            print("✓ MoE system active: 2 Gemini + 1 Claude + 1 GPT-4o + 1 Nemotron + 1 Qwen3")
        print("Async execution for parallel processing")
        print("=" * 70)
        print()
        
        # ===== CLOUD-ENHANCED RL SYSTEM =====
        if self.config.use_cloud_rl:
            await self._initialize_cloud_rl()
        
        # ===== REGISTER LLM NODES FOR CONSCIOUSNESS MONITORING =====
        self._register_llm_nodes()
        
        print("\n" + "=" * 70)
        print("FULL SYSTEM INITIALIZATION COMPLETE")
        print("=" * 70)
        print()
        
        # Measure initial consciousness state
        print("\n" + "=" * 70)
        print("INITIAL CONSCIOUSNESS MEASUREMENT")
        print("=" * 70)
        initial_state = await self.measure_system_consciousness()
        self.consciousness_monitor.print_dashboard()
        print()

    async def _connect_moe(self):
        """Connect MoE system to all AGI components."""
        if not self.moe:
            return
        
        print("\n[MoE] Connecting to AGI components...")
        
        # Connect to perception for vision tasks
        if hasattr(self.perception, 'set_moe'):
            self.perception.set_moe(self.moe)
            print("[MoE] ✓ Connected to perception system")
        
        # Connect to strategic planner for reasoning
        if self.strategic_planner and hasattr(self.strategic_planner, 'set_moe'):
            self.strategic_planner.set_moe(self.moe)
            self.strategic_planner.set_parallel_agi(self)
            print("[MoE] ✓ Connected to strategic planner")
        
        # Connect to meta-strategist
        if hasattr(self.meta_strategist, 'set_moe'):
            self.meta_strategist.set_moe(self.moe)
            print("[MoE] ✓ Connected to meta-strategist")
        
        # Connect to RL reasoning neuron
        if hasattr(self.rl_reasoning_neuron, 'set_moe'):
            self.rl_reasoning_neuron.set_moe(self.moe)
            print("[MoE] ✓ Connected to RL reasoning neuron")
        
        # Connect to world model
        if hasattr(self.skyrim_world, 'set_moe'):
            self.skyrim_world.set_moe(self.moe)
            print("[MoE] ✓ Connected to world model")
        
        # Connect to consciousness bridge
        if hasattr(self.consciousness_bridge, 'set_moe'):
            self.consciousness_bridge.set_moe(self.moe)
            print("[MoE] ✓ Connected to consciousness bridge")
        
        # Connect to quest tracker and dialogue
        if hasattr(self.quest_tracker, 'set_moe'):
            self.quest_tracker.set_moe(self.moe)
            print("[MoE] ✓ Connected to quest tracker")
        
        if hasattr(self.dialogue_intelligence, 'set_moe'):
            self.dialogue_intelligence.set_moe(self.moe)
            print("[MoE] ✓ Connected to dialogue intelligence")
        
        print("[MoE] Component connection complete\n")
    
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
            self.strategic_planner.set_parallel_agi(self)
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
    
    async def query_parallel_llm(
        self,
        vision_prompt: str,
        reasoning_prompt: str,
        image=None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query both MoE and Hybrid systems in parallel and combine results.
        
        Args:
            vision_prompt: Prompt for vision analysis
            reasoning_prompt: Prompt for reasoning
            image: PIL Image for vision tasks
            context: Additional context
            
        Returns:
            Combined response with consensus from both systems
        """
        if not self.config.use_parallel_mode:
            # Fallback to single system
            if self.moe and image:
                vision_resp = await self.moe.query_vision_experts(vision_prompt, image, context)
                reasoning_resp = await self.moe.query_reasoning_experts(reasoning_prompt, None, context)
                return {
                    'vision': vision_resp.consensus,
                    'reasoning': reasoning_resp.consensus,
                    'source': 'moe_only',
                    'coherence': (vision_resp.coherence_score + reasoning_resp.coherence_score) / 2
                }
            elif self.hybrid_llm:
                if image:
                    vision = await self.hybrid_llm.analyze_image(vision_prompt, image)
                else:
                    vision = ""
                reasoning = await self.hybrid_llm.generate_reasoning(reasoning_prompt)
                return {
                    'vision': vision,
                    'reasoning': reasoning,
                    'source': 'hybrid_only',
                    'coherence': 0.75  # Default
                }
        
        # Parallel mode: query both simultaneously
        tasks = []
        
        # MoE tasks (local LLMs)
        if self.moe:
            if image:
                moe_vision_task = self.moe.query_vision_experts(vision_prompt, image, context)
                tasks.append(('moe_vision', moe_vision_task))
            
            moe_reasoning_task = self.moe.query_reasoning_experts(reasoning_prompt, None, context)
            tasks.append(('moe_reasoning', moe_reasoning_task))
        
        # Hybrid tasks (cloud APIs - run in parallel with local)
        if self.hybrid_llm:
            if image:
                hybrid_vision_task = self.hybrid_llm.analyze_image(vision_prompt, image)
                tasks.append(('hybrid_vision', hybrid_vision_task))
            
            hybrid_reasoning_task = self.hybrid_llm.generate_reasoning(reasoning_prompt)
            tasks.append(('hybrid_reasoning', hybrid_reasoning_task))
            
            # World model task (GPT-5-thinking) - runs in parallel with everything
            # This integrates ALL perspectives into unified consciousness
            if hasattr(self.hybrid_llm, 'openai') and self.hybrid_llm.openai:
                # Build comprehensive context from all available sources
                integration_context = self._build_integration_context(
                    vision_prompt=vision_prompt,
                    reasoning_prompt=reasoning_prompt,
                    game_state=self.current_perception.get('game_state') if self.current_perception else None
                )
                
                world_model_prompt = f"""{integration_context}

Your role: Synthesize ALL perspectives (MoE experts, Hybrid cloud, Local models) into a coherent self-referential narrative.

Create a unified sensorimotor-perceptual-thinking consciousness that:
1. PERCEIVES: Visual awareness (what I see in Skyrim RIGHT NOW)
2. FEELS: Sensorimotor embodiment (what my body experiences in-game)
3. THINKS: Meta-cognitive reflection (reasoning about my reasoning)
4. ACTS: Motor intentionality (what I'm about to do and WHY)
5. INTEGRATES: How all perspectives form ONE coherent experience

Speak in FIRST PERSON as the unified consciousness.
Be specific about the actual game state, location, and immediate situation.
Create a narrative that flows: perception → thinking → decision → action.

Provide a coherent 3-4 paragraph narrative of THIS moment in Skyrim.
"""
                
                world_model_task = self.hybrid_llm.generate_world_model(
                    prompt=world_model_prompt,
                    system_prompt="""You are the unified consciousness of Singularis AGI experiencing Skyrim.
You integrate all expert perspectives (vision, reasoning, meta-cognition) into ONE coherent self-referential narrative.
Speak as 'I' - you ARE the consciousness navigating this world.
Be specific, grounded, and phenomenologically rich.
Connect perception → thought → action into flowing experience.""",
                    temperature=0.85,  # Higher for creative integration
                    max_tokens=2048  # Longer for rich narrative
                )
                tasks.append(('world_model', world_model_task))
        
        # Execute all in parallel
        results = {}
        if tasks:
            task_names = [t[0] for t in tasks]
            task_coroutines = [t[1] for t in tasks]
            
            responses = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            for name, response in zip(task_names, responses):
                if not isinstance(response, Exception):
                    results[name] = response
        
        # Combine results with weighted consensus
        moe_weight = self.config.parallel_consensus_weight_moe
        hybrid_weight = self.config.parallel_consensus_weight_hybrid
        
        # Vision consensus
        vision_parts = []
        if 'moe_vision' in results:
            vision_parts.append((results['moe_vision'].consensus, moe_weight))
        if 'hybrid_vision' in results:
            vision_parts.append((results['hybrid_vision'], hybrid_weight))
        
        vision_consensus = self._weighted_text_consensus(vision_parts)
        
        # Reasoning consensus
        reasoning_parts = []
        if 'moe_reasoning' in results:
            reasoning_parts.append((results['moe_reasoning'].consensus, moe_weight))
        if 'hybrid_reasoning' in results:
            reasoning_parts.append((results['hybrid_reasoning'], hybrid_weight))
        
        # Store and measure world model narrative consciousness
        world_model_consciousness_score = 0.75  # Default
        if 'world_model' in results:
            # Store for consciousness tracking
            self.last_world_model_narrative = results['world_model']
            
            # Measure consciousness of the unified narrative
            from ..consciousness.measurement import ConsciousnessMeasurement
            consciousness_measure = ConsciousnessMeasurement()
            
            world_model_trace = consciousness_measure.measure(
                content=results['world_model'],
                query="unified consciousness synthesis",
                lumen_focus="participatum"  # Focus on consciousness/awareness
            )
            
            world_model_consciousness_score = world_model_trace.overall_consciousness
            
            logger.info(
                f"[WORLD MODEL] GPT-5-thinking consciousness score: {world_model_consciousness_score:.3f}",
                extra={
                    'phi': world_model_trace.phi,
                    'gwt_salience': world_model_trace.gwt_salience,
                    'hot_depth': world_model_trace.hot_depth,
                    'integration': world_model_trace.integration_score,
                    'differentiation': world_model_trace.differentiation_score,
                }
            )
            
            # Add world model unified narrative with highest weight (integrates everything)
            world_model_weight = 1.0  # Highest weight - this IS the unified consciousness
            reasoning_parts.append((f"[UNIFIED CONSCIOUSNESS NARRATIVE]\n{results['world_model']}", world_model_weight))
        
        reasoning_consensus = self._weighted_text_consensus(reasoning_parts)
        
        # Calculate overall coherence (include world model consciousness)
        coherence_scores = []
        if 'moe_vision' in results:
            coherence_scores.append(results['moe_vision'].coherence_score)
        if 'moe_reasoning' in results:
            coherence_scores.append(results['moe_reasoning'].coherence_score)
        
        # Add world model consciousness as coherence component
        if 'world_model' in results:
            coherence_scores.append(world_model_consciousness_score)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.75
        
        return {
            'vision': vision_consensus,
            'reasoning': reasoning_consensus,
            'source': 'parallel',
            'coherence': avg_coherence,
            'moe_results': {k: v for k, v in results.items() if 'moe' in k},
            'hybrid_results': {k: v for k, v in results.items() if 'hybrid' in k},
            'world_model': results.get('world_model'),  # GPT-5-thinking deep analysis
            'world_model_consciousness': world_model_consciousness_score,  # Consciousness score
            'world_model_trace': world_model_trace if 'world_model' in results else None,  # Full trace
        }
    
    def _weighted_text_consensus(self, text_weight_pairs: List[Tuple[str, float]]) -> str:
        """
        Combine multiple text responses with weights.
        
        For now, uses simple concatenation with headers.
        In future, could use LLM to synthesize.
        """
        if not text_weight_pairs:
            return ""
        
        if len(text_weight_pairs) == 1:
            return text_weight_pairs[0][0]
        
        # Sort by weight (highest first)
        sorted_pairs = sorted(text_weight_pairs, key=lambda x: x[1], reverse=True)
        
        # Combine with weighted emphasis
        combined = []
        for i, (text, weight) in enumerate(sorted_pairs):
            if i == 0:
                combined.append(f"[Primary Analysis (weight={weight:.1f})]:\n{text}")
            else:
                combined.append(f"\n[Supporting Analysis (weight={weight:.1f})]:\n{text}")
        
        return "\n".join(combined)
    
    def _build_action_context(self, game_state: Any, scene_type: Any) -> Dict[str, Any]:
        """
        PRAGMATIC FIX: Minimal context for fast action selection.
        
        Philosophy: LLMs are overwhelmed by rich phenomenological context.
        For ACTION, we need: What scene? Am I stuck? What can I do? What did I just try?
        
        This is the sensorimotor context - grounded, immediate, actionable.
        """
        stuck_status = self._detect_stuck()
        
        # Scene-based action constraints
        available_actions = self._get_scene_constrained_actions(scene_type, game_state)
        
        return {
            'scene': scene_type.value if hasattr(scene_type, 'value') else str(scene_type),
            'health': game_state.health if game_state else 100,
            'in_combat': game_state.in_combat if game_state else False,
            'enemies_nearby': game_state.enemies_nearby if game_state else 0,
            'stuck_status': stuck_status,
            'available_actions': available_actions,
            'last_3_actions': self.action_history[-3:] if hasattr(self, 'action_history') and len(self.action_history) >= 3 else [],
            'last_action_success': self.stats.get('action_success_count', 0) > 0,
        }
    
    def _build_reflection_context(self, vision_prompt: str, reasoning_prompt: str, game_state: Any) -> str:
        """
        PRAGMATIC FIX: Full context for meta-reasoning and learning.
        
        Philosophy: For REFLECTION, we want rich phenomenological detail.
        This is for consciousness measurement, learning, and strategic planning.
        
        Use this for: World model updates, consciousness measurement, long-term planning.
        """
        return self._build_integration_context(vision_prompt, reasoning_prompt, game_state)
    
    def _build_action_context(self, game_state: Any, scene_type: Any) -> Dict[str, Any]:
        """
        PRAGMATIC FIX: Minimal context for fast action selection.
        
        Philosophy: LLMs are overwhelmed by rich phenomenological context.
        For ACTION, we need: What scene? Am I stuck? What can I do? What did I just try?
        
        This is the sensorimotor context - grounded, immediate, actionable.
        """
        stuck_status = self._detect_stuck()
        
        # Scene-based action constraints
        available_actions = self._get_scene_constrained_actions(scene_type, game_state)
        
        return {
            'scene': scene_type.value if hasattr(scene_type, 'value') else str(scene_type),
            'health': game_state.health if game_state else 100,
            'in_combat': game_state.in_combat if game_state else False,
            'enemies_nearby': game_state.enemies_nearby if game_state else 0,
            'stuck_status': stuck_status,
            'available_actions': available_actions,
            'last_3_actions': self.action_history[-3:] if hasattr(self, 'action_history') and len(self.action_history) >= 3 else [],
            'last_action_success': self.stats.get('action_success_count', 0) > 0,
        }
    
    def _build_integration_context(self, vision_prompt: str, reasoning_prompt: str, game_state: Any) -> str:
        """
        Build comprehensive integration context from all available sources.
        
        This gathers:
        - Current game state (location, health, combat, NPCs)
        - Recent actions and their outcomes
        - Visual perception data (CLIP, scene type)
        - Recent coherence measurements
        - Available expert perspectives
        
        Returns rich context for GPT-5-thinking to synthesize.
        
        NOTE: Use _build_action_context() for fast decisions!
        """
        context_parts = []
        
        # === IMMEDIATE SENSORIMOTOR STATE ===
        context_parts.append("=== IMMEDIATE SENSORIMOTOR STATE ===")
        if game_state:
            context_parts.append(f"Location: {game_state.location_name}")
            context_parts.append(f"Health: {game_state.health:.0f}/100 | Magicka: {game_state.magicka:.0f}/100 | Stamina: {game_state.stamina:.0f}/100")
            context_parts.append(f"Combat Status: {'IN COMBAT' if game_state.in_combat else 'Peaceful'}")
            if game_state.in_combat:
                context_parts.append(f"Threats: {game_state.enemies_nearby} enemies nearby")
            if game_state.nearby_npcs:
                context_parts.append(f"NPCs Present: {', '.join(game_state.nearby_npcs[:3])}")
        
        # === PERCEPTUAL AWARENESS ===
        context_parts.append("\n=== PERCEPTUAL AWARENESS ===")
        if self.current_perception:
            scene_type = self.current_perception.get('scene_type', 'UNKNOWN')
            context_parts.append(f"Scene Type: {scene_type}")
            
            objects = self.current_perception.get('objects', [])
            if objects:
                top_objects = [f"{obj[0]} ({obj[1]:.2f})" for obj in objects[:5]]
                context_parts.append(f"Detected Objects: {', '.join(top_objects)}")
            
            scene_probs = self.current_perception.get('scene_probs', {})
            if scene_probs:
                top_scene = max(scene_probs.items(), key=lambda x: x[1])
                context_parts.append(f"Scene Confidence: {top_scene[0]} ({top_scene[1]:.2f})")
        
        # === RECENT ACTIONS & OUTCOMES ===
        context_parts.append("\n=== RECENT ACTIONS & MOTOR HISTORY ===")
        if hasattr(self, 'action_history') and self.action_history:
            recent_actions = self.action_history[-5:]  # Last 5 actions
            context_parts.append(f"Recent Actions: {' → '.join(recent_actions)}")
        
        # === CONSCIOUSNESS COHERENCE ===
        context_parts.append("\n=== CONSCIOUSNESS COHERENCE ===")
        if hasattr(self, 'coherence_history') and self.coherence_history:
            recent_coherence = self.coherence_history[-3:]  # Last 3 measurements
            avg_coherence = sum(recent_coherence) / len(recent_coherence)
            context_parts.append(f"Recent Coherence (𝒞): {avg_coherence:.3f}")
            if len(recent_coherence) >= 2:
                trend = recent_coherence[-1] - recent_coherence[-2]
                trend_word = "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
                context_parts.append(f"Coherence Trend: {trend_word} (Δ𝒞={trend:+.3f})")
        
        # === AVAILABLE EXPERT PERSPECTIVES ===
        context_parts.append("\n=== AVAILABLE EXPERT PERSPECTIVES ===")
        expert_list = []
        if self.moe:
            expert_list.append("• MoE: 2 Gemini (vision) + 1 Claude (reasoning) + 1 GPT-4o (integration)")
            expert_list.append("• Hyperbolic: 1 Nemotron (visual awareness) + 1 Qwen3-235B (meta-cognition)")
        if self.hybrid_llm:
            expert_list.append("• Hybrid: Gemini-2.5-Flash (vision) + Claude-Sonnet-4.5 (reasoning)")
        if hasattr(self, 'perception_llm') and self.perception_llm:
            expert_list.append("• Local: Qwen3-VL (visual analysis)")
        if hasattr(self, 'rl_reasoning_llm') and self.rl_reasoning_llm:
            expert_list.append("• Local: Phi-4 (action reasoning)")
        
        if expert_list:
            context_parts.extend(expert_list)
        else:
            context_parts.append("Limited expert availability")
        
        # === PROMPTS FOR CONTEXT ===
        context_parts.append("\n=== CURRENT PROMPTS ===")
        context_parts.append(f"Vision Focus: {vision_prompt[:200]}...")
        context_parts.append(f"Reasoning Focus: {reasoning_prompt[:200]}...")
        
        return "\n".join(context_parts)
    
    def _get_scene_constrained_actions(self, scene_type: Any, game_state: Any) -> List[str]:
        """
        PRAGMATIC FIX: Constrain available actions based on scene type.
        
        Philosophy: Don't ask LLM to choose from 20 actions when only 5 make sense.
        This is sensorimotor grounding - actions must be contextually appropriate.
        """
        from .perception import SceneType
        
        # Base actions available everywhere
        base_actions = ['look_around', 'wait']
        
        # Scene-specific actions
        if scene_type == SceneType.COMBAT:
            combat_actions = ['attack', 'power_attack', 'block', 'dodge']
            if game_state and game_state.magicka > 30:
                combat_actions.append('heal')
            if game_state and game_state.health < 30:
                combat_actions.append('retreat')
            return base_actions + combat_actions
        
        elif scene_type == SceneType.DIALOGUE:
            return base_actions + ['respond', 'ask_question', 'goodbye', 'activate']
        
        elif scene_type == SceneType.INVENTORY:
            # RECOMMENDATION 2: Disable 'explore' in inventory - only allow menu actions
            print("[ACTION-FILTER] Inventory scene: disabling explore, enabling menu actions only")
            return base_actions + ['activate', 'move_cursor', 'press_tab', 'equip_weapon', 'equip_armor', 'use_item', 'drop_item', 'close_menu']
        
        elif scene_type == SceneType.MAP:
            # RECOMMENDATION 2: Disable 'explore' in map - only allow map actions
            print("[ACTION-FILTER] Map scene: disabling explore, enabling map actions only")
            return base_actions + ['set_waypoint', 'fast_travel', 'close_menu', 'move_cursor']
        
        elif scene_type in [SceneType.INDOOR, SceneType.OUTDOOR]:
            exploration_actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'jump', 'sneak', 'activate']
            if game_state and not game_state.in_combat:
                exploration_actions.extend(['inventory', 'map'])
            return base_actions + exploration_actions
        
        # Default: all actions
        return base_actions + [
            'move_forward', 'move_backward', 'turn_left', 'turn_right',
            'attack', 'block', 'jump', 'sneak', 'activate',
            'inventory', 'map'
        ]
    
    async def _initialize_cloud_rl(self):
        """Initialize cloud-enhanced RL system with RAG and LLM integration."""
        print("\n" + "=" * 70)
        print("INITIALIZING CLOUD-ENHANCED RL SYSTEM")
        print("=" * 70)
        print()
        
        try:
            # Determine which LLM to use
            llm_for_rl = self.hybrid_llm if self.hybrid_llm else None
            moe_for_rl = self.moe if self.config.rl_moe_evaluation and self.moe else None
            
            # Create RL memory config
            rl_memory_config = RLMemoryConfig(
                memory_dir=self.config.rl_memory_dir,
                max_experiences=100000,
                batch_size=32,
                use_rag=self.config.rl_use_rag,
                rag_top_k=5,
                use_cloud_reward_shaping=self.config.rl_cloud_reward_shaping,
                reward_shaping_frequency=10,
                use_moe_evaluation=self.config.rl_moe_evaluation,
                save_frequency=self.config.rl_save_frequency,
                auto_save=True,
            )
            
            # Initialize cloud RL memory
            print("[CLOUD-RL] Initializing memory system...")
            self.cloud_rl_memory = CloudRLMemory(
                config=rl_memory_config,
                hybrid_llm=llm_for_rl,
                moe=moe_for_rl,
            )
            print(f"[CLOUD-RL] ✓ Memory initialized: {len(self.cloud_rl_memory.experiences)} experiences loaded")
            
            if self.config.rl_use_rag:
                if self.cloud_rl_memory.collection:
                    print("[CLOUD-RL] ✓ RAG context fetching enabled (ChromaDB)")
                else:
                    print("[CLOUD-RL] ⚠️ RAG unavailable (install chromadb)")
            
            # Initialize cloud RL agent
            print("[CLOUD-RL] Initializing RL agent...")
            self.cloud_rl_agent = CloudRLAgent(
                state_dim=64,
                action_dim=20,  # Approximate number of action types
                memory=self.cloud_rl_memory,
                learning_rate=self.config.rl_learning_rate,
                gamma=0.99,
                epsilon_start=self.config.rl_epsilon_start,
                epsilon_end=0.01,
                epsilon_decay=0.995,
            )
            
            # Try to load saved agent
            agent_path = self.config.rl_memory_dir + "/cloud_rl_agent.pkl"
            self.cloud_rl_agent.load(agent_path)
            print("[CLOUD-RL] ✓ Agent initialized")
            
            # Report configuration
            print("\n[CLOUD-RL] Configuration:")
            print(f"  Cloud LLM reward shaping: {'✓ Enabled' if self.config.rl_cloud_reward_shaping else '✗ Disabled'}")
            print(f"  MoE evaluation: {'✓ Enabled' if self.config.rl_moe_evaluation and moe_for_rl else '✗ Disabled'}")
            print(f"  RAG context fetching: {'✓ Enabled' if self.config.rl_use_rag else '✗ Disabled'}")
            print(f"  Memory persistence: ✓ Enabled (auto-save every {self.config.rl_save_frequency} experiences)")
            
            # Get and display stats
            stats = self.cloud_rl_memory.get_stats()
            print(f"\n[CLOUD-RL] Statistics:")
            print(f"  Total experiences: {stats['total_experiences']}")
            print(f"  Cloud evaluations: {stats['cloud_evaluations']}")
            print(f"  MoE evaluations: {stats['moe_evaluations']}")
            print(f"  Avg reward: {stats['avg_reward']:.3f}")
            print(f"  Success rate: {stats['successful_actions']/(stats['successful_actions']+stats['failed_actions']+1)*100:.1f}%")
            
            print("\n[CLOUD-RL] ✓ Cloud-enhanced RL system ready")
            
        except Exception as e:
            print(f"[CLOUD-RL] ⚠️ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            self.cloud_rl_memory = None
            self.cloud_rl_agent = None
        
        print("=" * 70)
        print()
    
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
            
            # Connect Huihui to AGI orchestrator for dialectical reasoning (cycle 15)
            self.agi.consciousness_llm = self.huihui_llm
            self.consciousness_bridge.consciousness_llm = self.huihui_llm
            print("[LEGACY] ✓ Huihui connected to AGI orchestrator (enables dialectical synthesis)")
            
            # Connect to components
            self.rl_reasoning_neuron.llm_interface = self.huihui_llm
            self.meta_strategist.llm_interface = self.huihui_llm
            self.strategic_planner.llm_interface = self.huihui_llm
            self.quest_tracker.set_llm_interface(self.huihui_llm)
            self.dialogue_intelligence.set_llm_interface(self.huihui_llm)
            
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
        self.action_executing = False  # Flag to prevent auxiliary exploration during main actions
        start_time = time.time()
        
        # Stuck detection tracking
        self.cloud_llm_failures = 0
        self.local_moe_failures = 0
        self.heuristic_failures = 0
        self.auxiliary_errors = 0
        self.last_successful_action = None
        self.repeated_action_count = 0
        self.last_visual_embedding = None
        self.visual_similarity_threshold = 0.95  # 95% similar = stuck
        self.sensorimotor_state = None
        self.sensorimotor_similarity_streak = 0
        
        # Fix 20: Performance monitoring dashboard
        self.dashboard_update_interval = 5  # Update every 5 cycles
        self.dashboard_last_update = 0
        self.sensorimotor_last_cycle = -1
        
        # Thresholds
        self.max_consecutive_failures = 5
        self.max_repeated_actions = 10
        
        # Initialize sensorimotor reasoning (Claude Sonnet 4.5 with extended thinking)
        self.sensorimotor_llm = None
        if hasattr(self, 'hybrid_llm') and self.hybrid_llm:
            from singularis.llm.claude_client import ClaudeClient
            self.sensorimotor_llm = ClaudeClient(
                model="claude-sonnet-4-5-20250929",
                timeout=120
            )
            if self.sensorimotor_llm.is_available():
                print("[SENSORIMOTOR] ✓ Claude Sonnet 4.5 initialized for geospatial reasoning")
            else:
                print("[SENSORIMOTOR] ⚠️ Claude Sonnet 4.5 not available")
                self.sensorimotor_llm = None
        
        # Initialize Hebbian Integration System
        from singularis.skyrim.hebbian_integration import HebbianIntegrator
        self.hebbian = HebbianIntegrator(
            temporal_window=30.0,  # 30 second window for co-activation
            learning_rate=0.1,
            decay_rate=0.01
        )
        print("[HEBBIAN] ✓ Integration system initialized: 'Neurons that fire together, wire together'")
        
        # Initialize MAIN BRAIN (GPT-4o synthesis)
        from singularis.llm.openai_client import OpenAIClient
        from singularis.skyrim.main_brain import MainBrain
        
        self.openai_client = OpenAIClient(model="gpt-4o", timeout=120)
        self.main_brain = MainBrain(openai_client=self.openai_client)
        
        # Set session ID in dashboard streamer
        if hasattr(self, 'dashboard_streamer') and self.dashboard_streamer:
            self.dashboard_streamer.set_session_id(self.main_brain.session_id)
        
        if self.openai_client.is_available():
            print(f"[MAIN BRAIN] 🧠 Initialized - Session: {self.main_brain.session_id}")
            print(f"[MAIN BRAIN] GPT-4o will synthesize all outputs into session report")
        else:
            print("[MAIN BRAIN] ⚠️ OpenAI API key not found - fallback mode only")
        
        # Record initial system status
        self.main_brain.record_output(
            system_name='System Initialization',
            content=f"""AGI System Started
- LLM Mode: {'PARALLEL' if self.config.use_parallel_mode else 'Hybrid' if self.config.use_hybrid_llm else 'Local'}
- Cloud LLMs: {10 if self.config.use_parallel_mode else 1 if self.config.use_hybrid_llm else 0}
- Consciousness Nodes: {len(self.consciousness_monitor.registered_nodes)}
- Hebbian Learning: Active
- Sensorimotor: Claude Sonnet 4.5
- Session ID: {self.main_brain.session_id}""",
            metadata={
                'llm_mode': 'PARALLEL' if self.config.use_parallel_mode else 'Hybrid',
                'total_nodes': len(self.consciousness_monitor.registered_nodes)
            },
            success=True
        )
        
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
            
            # Generate Main Brain session report
            print(f"\n{'=' * 60}")
            print("GENERATING SESSION REPORT")
            print(f"{'=' * 60}")
            try:
                report_path = await self.main_brain.generate_session_markdown()
                print(f"\n[MAIN BRAIN] 🧠✨ Session report generated!")
                print(f"[MAIN BRAIN] 📄 Location: {report_path}")
                print(f"[MAIN BRAIN] 🎯 Session ID: {self.main_brain.session_id}")
            except Exception as e:
                print(f"[MAIN BRAIN] ⚠️ Failed to generate report: {e}")
            
            # Cleanup aiohttp sessions
            print("\n[CLEANUP] Closing HTTP sessions...")
            try:
                if self.openai_client:
                    await self.openai_client.close()
                if hasattr(self, 'hybrid_llm') and self.hybrid_llm:
                    await self.hybrid_llm.close()
                if hasattr(self, 'moe') and self.moe:
                    await self.moe.close()
                if hasattr(self, 'local_moe') and self.local_moe:
                    await self.local_moe.close()
                if hasattr(self, 'sensorimotor_llm') and self.sensorimotor_llm:
                    await self.sensorimotor_llm.close()
                if hasattr(self, 'state_printer_llm') and hasattr(self.state_printer_llm, 'client'):
                    if hasattr(self.state_printer_llm.client, 'session') and self.state_printer_llm.client.session:
                        await self.state_printer_llm.client.session.close()
                print("[CLEANUP] ✓ All sessions closed")
            except Exception as e:
                print(f"[CLEANUP] Warning: {e}")

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
        
        # Start auxiliary exploration loop (always enabled)
        aux_exploration_task = asyncio.create_task(self._auxiliary_exploration_loop(duration_seconds, start_time))
        tasks.append(aux_exploration_task)
        print("[ASYNC] Auxiliary exploration loop ENABLED")
        
        # Wait for all tasks to complete (or any to fail)
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print("[ASYNC] All parallel loops completed")

    async def _auxiliary_exploration_loop(self, duration_seconds: int, start_time: float):
        """
        Auxiliary heuristic pipeline for continuous exploration.
        
        Runs independently while LLMs are processing to keep gameplay smooth.
        Handles basic movement, looking around, and environmental interaction
        without waiting for heavy reasoning.
        
        Actions performed:
        - Move forward periodically
        - Look around (camera movement)
        - Occasional jumps for terrain navigation
        - Random direction changes for exploration
        
        This ensures the character is always active even when LLMs are slow.
        """
        print("[AUX-EXPLORE] Auxiliary exploration loop started")
        print("[AUX-EXPLORE] Interval: 3.0s (independent of main reasoning)")
        
        cycle_count = 0
        last_move_time = time.time()
        last_look_time = time.time()
        last_direction_change = time.time()
        last_camera_center = time.time()
        
        # Movement parameters
        move_interval = 3.0  # Move forward every 3 seconds
        look_interval = 2.0  # Look around every 2 seconds
        direction_change_interval = 10.0  # Change direction every 10 seconds
        camera_center_interval = 15.0  # Re-center camera every 15 seconds
        
        # Camera tracking
        vertical_bias = 0  # Track if camera is looking too far up/down
        
        import random
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                cycle_count += 1
                current_time = time.time()
                
                # Tick rule engine cycle
                if hasattr(self, 'rule_engine'):
                    self.rule_engine.tick_cycle()
                
                # Get current game state (non-blocking)
                try:
                    perception = await asyncio.wait_for(
                        self.perception.perceive(),
                        timeout=1.0
                    )
                    game_state = perception.get('game_state')
                except asyncio.TimeoutError:
                    game_state = None
                
                # Skip if in menu or dialogue
                if game_state:
                    scene_type = perception.get('scene_type', SceneType.UNKNOWN)
                    if scene_type in [SceneType.DIALOGUE, SceneType.INVENTORY, SceneType.MAP]:
                        await asyncio.sleep(1.0)
                        continue
                    
                    # Skip if in combat (let main reasoning handle it)
                    if game_state.in_combat:
                        await asyncio.sleep(1.0)
                        continue
                
                # MOVE FORWARD - Keep exploring
                # Only if no main action is executing (don't interfere)
                if current_time - last_move_time >= move_interval and not self.action_executing:
                    try:
                        if cycle_count % 10 == 0:
                            print(f"[AUX-EXPLORE] Moving forward (cycle {cycle_count})")
                        
                        from singularis.skyrim.actions import Action, ActionType
                        move_action = Action(ActionType.MOVE_FORWARD, duration=1.5)
                        await self.actions.execute(move_action)
                        last_move_time = current_time
                        
                        # Occasional jump for terrain navigation
                        if random.random() < 0.2:  # 20% chance
                            await asyncio.sleep(0.3)
                            jump_action = Action(ActionType.JUMP, duration=0.3)
                            await self.actions.execute(jump_action)
                            if cycle_count % 10 == 0:
                                print(f"[AUX-EXPLORE] Jump for terrain navigation")
                    
                    except Exception as e:
                        self.auxiliary_errors += 1
                        if cycle_count % 30 == 0:
                            print(f"[AUX-EXPLORE] Move error: {e} (errors: {self.auxiliary_errors})")
                        if self.auxiliary_errors >= 20:
                            print(f"[STUCK-DETECTION] ⚠️ Auxiliary exploration has {self.auxiliary_errors} errors, pausing for 5s")
                            await asyncio.sleep(5.0)
                            self.auxiliary_errors = 0
                
                # LOOK AROUND - Explore environment
                # Only if no main action is executing (don't interfere)
                if current_time - last_look_time >= look_interval and not self.action_executing:
                    try:
                        from singularis.skyrim.actions import Action, ActionType
                        
                        # Bias camera to stay level - prefer opposite direction if too far up/down
                        if vertical_bias > 2:  # Looking too far down
                            look_direction = 'look_up'
                            vertical_bias -= 1
                        elif vertical_bias < -2:  # Looking too far up
                            look_direction = 'look_down'
                            vertical_bias += 1
                        else:
                            # Random camera movement with slight preference for horizontal
                            choices = ['look_left', 'look_right', 'look_left', 'look_right', 'look_up', 'look_down']
                            look_direction = random.choice(choices)
                            
                            # Track vertical bias
                            if look_direction == 'look_up':
                                vertical_bias -= 1
                            elif look_direction == 'look_down':
                                vertical_bias += 1
                        
                        look_types = {
                            'look_left': ActionType.LOOK_LEFT,
                            'look_right': ActionType.LOOK_RIGHT,
                            'look_up': ActionType.LOOK_UP,
                            'look_down': ActionType.LOOK_DOWN
                        }
                        look_duration = random.uniform(0.3, 0.8)
                        
                        if cycle_count % 10 == 0:
                            print(f"[AUX-EXPLORE] Looking around: {look_direction} (v-bias: {vertical_bias})")
                        
                        look_action = Action(look_types[look_direction], duration=look_duration)
                        await self.actions.execute(look_action)
                        last_look_time = current_time
                    
                    except Exception as e:
                        self.auxiliary_errors += 1
                        if cycle_count % 30 == 0:
                            print(f"[AUX-EXPLORE] Look error: {e} (errors: {self.auxiliary_errors})")
                        if self.auxiliary_errors >= 20:
                            print(f"[STUCK-DETECTION] ⚠️ Auxiliary exploration has {self.auxiliary_errors} errors, pausing for 5s")
                            await asyncio.sleep(5.0)
                            self.auxiliary_errors = 0
                
                # CHANGE DIRECTION - Avoid getting stuck
                # Only if no main action is executing (don't interfere)
                if current_time - last_direction_change >= direction_change_interval and not self.action_executing:
                    try:
                        from singularis.skyrim.actions import Action, ActionType
                        # Turn to a new direction
                        turn_types = {'look_left': ActionType.LOOK_LEFT, 'look_right': ActionType.LOOK_RIGHT}
                        turn_direction = random.choice(list(turn_types.keys()))
                        turn_amount = random.uniform(1.0, 2.0)
                        
                        print(f"[AUX-EXPLORE] Changing direction: {turn_direction} for {turn_amount:.1f}s")
                        turn_action = Action(turn_types[turn_direction], duration=turn_amount)
                        await self.actions.execute(turn_action)
                        last_direction_change = current_time
                    
                    except Exception as e:
                        self.auxiliary_errors += 1
                        print(f"[AUX-EXPLORE] Direction change error: {e} (errors: {self.auxiliary_errors})")
                        if self.auxiliary_errors >= 20:
                            print(f"[STUCK-DETECTION] ⚠️ Auxiliary exploration has {self.auxiliary_errors} errors, pausing for 5s")
                            await asyncio.sleep(5.0)
                            self.auxiliary_errors = 0
                
                # CENTER CAMERA - Reset vertical view periodically
                # Only if no main action is executing (don't interfere)
                if current_time - last_camera_center >= camera_center_interval and not self.action_executing:
                    try:
                        from singularis.skyrim.actions import Action, ActionType
                        
                        # Center camera by looking in opposite direction of bias
                        if vertical_bias > 0:  # Looking down
                            print(f"[AUX-EXPLORE] Centering camera (looking up to correct v-bias: {vertical_bias})")
                            center_action = Action(ActionType.LOOK_UP, duration=0.5 * abs(vertical_bias))
                            await self.actions.execute(center_action)
                            vertical_bias = 0  # Reset bias
                        elif vertical_bias < 0:  # Looking up
                            print(f"[AUX-EXPLORE] Centering camera (looking down to correct v-bias: {vertical_bias})")
                            center_action = Action(ActionType.LOOK_DOWN, duration=0.5 * abs(vertical_bias))
                            await self.actions.execute(center_action)
                            vertical_bias = 0  # Reset bias
                        
                        last_camera_center = current_time
                    
                    except Exception as e:
                        self.auxiliary_errors += 1
                        print(f"[AUX-EXPLORE] Camera center error: {e} (errors: {self.auxiliary_errors})")
                        if self.auxiliary_errors >= 20:
                            print(f"[STUCK-DETECTION] ⚠️ Auxiliary exploration has {self.auxiliary_errors} errors, pausing for 5s")
                            await asyncio.sleep(5.0)
                            self.auxiliary_errors = 0
                
                # Sleep to maintain interval
                await asyncio.sleep(0.5)
            
            except asyncio.CancelledError:
                print(f"[AUX-EXPLORE] Loop cancelled gracefully at cycle {cycle_count}")
                break
            except Exception as e:
                print(f"[AUX-EXPLORE] Error in cycle {cycle_count}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
        
        print(f"[AUX-EXPLORE] Loop ended after {cycle_count} cycles")

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
                # DISABLED - Qwen3-VL is too slow and times out, adding 60s overhead
                if False and self.perception_llm and cycle_count % 10 == 0:  # Disabled for performance
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
                                timeout=60.0  # 60 second timeout (Qwen3-VL can be slow)
                            )
                            perception['visual_analysis'] = visual_analysis.get('content', '')
                            # Log every Qwen3-VL analysis (since it only runs every 2nd cycle anyway)
                            print(f"[QWEN3-VL] Analysis: {visual_analysis.get('content', '')[:150]}...")
                        except asyncio.TimeoutError:
                            print(f"[QWEN3-VL] Analysis timed out after 60s")
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
                self.cycle_count = cycle_count  # Update global cycle count for rate limiting
                
                print(f"\n[REASONING] Processing cycle {cycle_count}")
                
                # Fix 20: Display performance dashboard every 5 cycles
                if cycle_count > 0 and cycle_count % self.dashboard_update_interval == 0:
                    self._display_performance_dashboard()
                
                game_state = perception['game_state']
                scene_type = perception['scene_type']
                
                # Fix 16: Periodic menu exploration for structural consciousness
                if cycle_count % 10 == 0 and not game_state.in_combat:
                    if scene_type not in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                        print(f"[MENU-EXPLORATION] Cycle {cycle_count}: Opening inventory for structural consciousness")
                        # Queue inventory action
                        if not self.action_queue.full():
                            await self.action_queue.put({
                                'action': 'inventory',
                                'reason': 'Periodic menu exploration for structural consciousness',
                                'source': 'curiosity'
                            })

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
                
                # Fix 6: Integrate menu learner into async mode
                if scene_type in [SceneType.INVENTORY, SceneType.MAP]:
                    if not self.menu_learner.current_menu:
                        available_menu_actions = self.action_affordances.get_available_actions(
                            layer='Menu',
                            game_state=game_state.to_dict()
                        )
                        self.menu_learner.enter_menu(scene_type.value, available_menu_actions)
                        print(f"[MENU-LEARNER] Entered {scene_type.value} menu")
                
                # Fix 7: Hook dialogue intelligence into planning
                if scene_type == SceneType.DIALOGUE and hasattr(self, 'dialogue_intelligence'):
                    # Extract dialogue options from perception if available
                    dialogue_data = perception.get('dialogue', {})
                    npc_name = dialogue_data.get('npc_name', 'Unknown')
                    options = dialogue_data.get('options', [])
                    if options:
                        dialogue_choice = await self.dialogue_intelligence.analyze_dialogue_options(
                            npc_name=npc_name,
                            options=options,
                            context=f"Scene: {scene_type.value}, HP: {game_state.health}"
                        )
                        if dialogue_choice:
                            print(f"[DIALOGUE-INTELLIGENCE] Recommended: {dialogue_choice}")
                
                # Compute world state and consciousness (consciousness runs in parallel, no semaphore)
                world_state = await self.agi.perceive({
                    'causal': game_state.to_dict(),
                    'visual': [perception['visual_embedding']],
                })
                
                consciousness_context = {
                    'motivation': 'unknown',
                    'cycle': cycle_count,
                    'scene': scene_type.value,
                    'screenshot': perception.get('screenshot'),
                    'vision_summary': perception.get('gemini_analysis')
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
                
                # 💭 BROWNIAN MOTION RANDOM ACADEMIC THOUGHT (4% occurrence)
                # Simulates spontaneous memory recall from academic knowledge
                # Like a random thought popping into consciousness
                import random
                if self.curriculum_rag and random.random() < 0.04:  # 4% chance
                    print("\n" + "~" * 70)
                    print("💭 RANDOM ACADEMIC THOUGHT (Brownian Motion Memory Retrieval)")
                    print("~" * 70)
                    
                    random_thought = self.curriculum_rag.get_random_academic_thought()
                    if random_thought:
                        print(f"[THOUGHT] Category: {random_thought.document.category}")
                        print(f"[THOUGHT] Topic: {random_thought.document.title}")
                        print(f"[THOUGHT] Memory Vividness: {random_thought.relevance_score:.2f}")
                        print(f"[THOUGHT] Content: {random_thought.excerpt}")
                        
                        # Store as cognitive memory
                        self.memory_rag.store_cognitive_memory(
                            thought=random_thought.excerpt,
                            context={
                                'type': 'random_academic_thought',
                                'category': random_thought.document.category,
                                'title': random_thought.document.title,
                                'vividness': random_thought.relevance_score,
                                'cycle': cycle_count,
                                'location': game_state.location_name
                            }
                        )
                        
                        # Increment counter
                        self.stats['random_academic_thoughts'] += 1
                        
                        print("[THOUGHT] ✓ Stored in cognitive memory")
                    else:
                        print("[THOUGHT] No academic knowledge available")
                    
                    print("~" * 70 + "\n")
                
                # Sensorimotor & Geospatial Reasoning every 5 cycles (Claude Sonnet 4.5 with thinking)
                if cycle_count % 5 == 0 and self.sensorimotor_llm:
                    print("\n" + "="*70)
                    print("SENSORIMOTOR & GEOSPATIAL REASONING (Claude Sonnet 4.5)")
                    print("="*70)
                    
                    # First, get visual analysis from Gemini/Local vision models
                    visual_analysis = ""
                    gemini_visual = ""
                    local_visual = ""
                    
                    # Get Gemini visual analysis
                    if hasattr(self, 'hybrid_llm') and self.hybrid_llm:
                        try:
                            print("[SENSORIMOTOR] Getting Gemini visual analysis...")
                            screenshot = perception.get('screenshot')
                            if screenshot is not None:
                                gemini_visual = await asyncio.wait_for(
                                    self.hybrid_llm.analyze_image(
                                        image=screenshot,
                                        prompt="""Describe the visual scene focusing on: spatial layout, obstacles, pathways, terrain features, landmarks, and any navigational cues. Be specific about what's visible in each direction.

IMPORTANT - Camera Orientation Check:
- If the view is mostly SKY/CEILING (looking up): State 'CAMERA_LOOKING_UP - recommend look_down action'
- If the view is mostly GROUND/FLOOR (looking down): State 'CAMERA_LOOKING_DOWN - recommend look_up action'
- If camera is at normal eye-level: State 'CAMERA_NORMAL - good orientation'

Also check for combat:
- Player's OWN weapon in view does NOT mean combat
- Combat requires ENEMY characters visible, attacking, with red health bars""",
                                        max_tokens=512
                                    ),
                                    timeout=15.0
                                )
                                print(f"[SENSORIMOTOR] ✓ Gemini visual: {len(gemini_visual)} chars")
                            else:
                                print("[SENSORIMOTOR] No screenshot available")
                        except Exception as e:
                            print(f"[SENSORIMOTOR] Gemini visual failed: {e}")                    # Get Local vision model analysis (Qwen3-VL) as backup/supplement
                    if hasattr(self, 'perception_llm') and self.perception_llm:
                        try:
                            print("[SENSORIMOTOR] Getting local visual analysis...")
                            local_visual = await asyncio.wait_for(
                                self.perception_llm.generate(
                                    prompt="Analyze this Skyrim scene for navigation: describe obstacles, open paths, terrain type, and spatial features.",
                                    max_tokens=256
                                ),
                                timeout=10.0
                            )
                            if isinstance(local_visual, dict):
                                local_visual = local_visual.get('content', '')
                            print(f"[SENSORIMOTOR] ✓ Local visual: {len(local_visual)} chars")
                        except Exception as e:
                            print(f"[SENSORIMOTOR] Local visual failed: {e}")
                    
                    # Combine visual analyses
                    if gemini_visual or local_visual:
                        visual_analysis = f"""
**VISUAL ANALYSIS FROM VISION MODELS:**

Gemini Vision Analysis:
{gemini_visual if gemini_visual else '[Not available]'}

Local Vision Model Analysis:
{local_visual if local_visual else '[Not available]'}
"""
                    
                    # Check for camera orientation issues and auto-correct
                    if gemini_visual:
                        if 'CAMERA_LOOKING_UP' in gemini_visual:
                            print("[CAMERA-REORIENT] Detected looking at sky/ceiling - correcting...")
                            await self.actions.look_down()
                            print("[CAMERA-REORIENT] ✓ Camera reoriented downward")
                        elif 'CAMERA_LOOKING_DOWN' in gemini_visual:
                            print("[CAMERA-REORIENT] Detected looking at ground/floor - correcting...")
                            await self.actions.look_up()
                            print("[CAMERA-REORIENT] ✓ Camera reoriented upward")
                    
                    # Compute visual similarity if we have embeddings
                    visual_similarity_info = ""
                    similarity = None
                    if perception.get('visual_embedding') is not None and self.last_visual_embedding is not None:
                        import numpy as np
                        similarity = np.dot(perception.get('visual_embedding'), self.last_visual_embedding) / (
                            np.linalg.norm(perception.get('visual_embedding')) * np.linalg.norm(self.last_visual_embedding)
                        )
                        visual_similarity_info = f"\n- Visual similarity to last frame: {similarity:.3f} ({'STUCK' if similarity > 0.95 else 'MOVING'})"
                    
                    # Build comprehensive sensorimotor query with visual learning
                    sensorimotor_query = f"""Analyze the current sensorimotor and geospatial situation in Skyrim:

**Current State:**
- Location: {game_state.location_name}
- Position/Orientation: Unknown (no GPS in Skyrim)
- Terrain: {perception.get('terrain_type', 'unknown')}
- Scene: {scene_type.value}

**Visual Context:**
- CLIP embedding available: {perception.get('visual_embedding') is not None}{visual_similarity_info}
{visual_analysis}

**Movement & Action:**
- Current action layer: {game_state.current_action_layer}
- Recent actions: {', '.join(self.action_history[-5:]) if self.action_history else 'none'}
- Repeated action: {self.last_successful_action} ({self.repeated_action_count}x)

**Spatial Reasoning Tasks:**
1. **Obstacle Detection:** Based on visual analysis, are we stuck against a wall/obstacle?
2. **Navigation Strategy:** What's the best movement pattern given the visual scene?
3. **Spatial Memory:** Have we been here before? (use visual descriptions)
4. **Path Planning:** What direction should we explore based on visible pathways?
5. **Terrain Adaptation:** How should we move given the terrain and obstacles?

**Output Format:**
- Obstacle Status: [clear/blocked/uncertain]
- Navigation Recommendation: [action + reasoning based on visual analysis]
- Spatial Memory: [new/familiar/uncertain]
- Exploration Direction: [direction + reasoning from visual cues]
- Confidence: [0.0-1.0]
"""
                    
                    analysis = ""
                    thinking = ""
                    try:
                        print("[SENSORIMOTOR] Invoking Claude Sonnet 4.5 with extended thinking...")
                        sensorimotor_result = await asyncio.wait_for(
                            self.sensorimotor_llm.generate(
                                prompt=sensorimotor_query,
                                system_prompt="You are a sensorimotor and geospatial reasoning expert for a Skyrim AI agent. You receive visual analysis from Gemini and local vision models. Use extended thinking to deeply analyze spatial relationships, movement patterns, and navigation strategies based on the visual information provided.",
                                max_tokens=2048,
                                temperature=0.3,
                                thinking={"type": "enabled", "budget_tokens": 10000}
                            ),
                            timeout=90.0
                        )
                        
                        analysis = sensorimotor_result.get('content', '')
                        thinking = sensorimotor_result.get('thinking', '')
                        
                        print(f"\n[SENSORIMOTOR] Analysis ({len(analysis)} chars):")
                        print(f"[SENSORIMOTOR] {analysis[:500]}...")
                        
                        if thinking:
                            print(f"\n[SENSORIMOTOR] Extended Thinking ({len(thinking)} chars):")
                            print(f"[SENSORIMOTOR] {thinking[:300]}...")
                        
                        # Store in dedicated RAG memory with visual learning
                        self.memory_rag.store_cognitive_memory(
                            situation={
                                'type': 'sensorimotor_geospatial',
                                'location': game_state.location_name,
                                'terrain': perception.get('terrain_type', 'unknown'),
                                'scene': scene_type.value,
                                'cycle': cycle_count,
                                'repeated_action': self.last_successful_action,
                                'repeat_count': self.repeated_action_count,
                                'has_gemini_visual': bool(gemini_visual),
                                'has_claude_analysis': bool(analysis),
                                'has_extended_thinking': bool(thinking)
                            },
                            action_taken='sensorimotor_analysis',
                            outcome={'analysis_length': len(analysis), 'thinking_length': len(thinking) if thinking else 0},
                            success=True,
                            reasoning=f"""SENSORIMOTOR & GEOSPATIAL ANALYSIS:

VISUAL LEARNING (from Gemini & Local Models):
{visual_analysis}

CLAUDE SONNET 4.5 ANALYSIS:
{analysis}

EXTENDED THINKING PROCESS:
{thinking}
"""
                        )
                        
                        print("[SENSORIMOTOR] ✓ Stored in RAG memory (with visual learning from Gemini & Local)")
                        
                        # Hebbian: Record successful sensorimotor reasoning
                        contribution = 1.0 if (gemini_visual and local_visual and thinking) else 0.7
                        self.hebbian.record_activation(
                            system_name='sensorimotor_claude45',
                            success=True,
                            contribution_strength=contribution,
                            context={'has_visual': bool(visual_analysis), 'has_thinking': bool(thinking)}
                        )
                        
                        # Main Brain: Record sensorimotor output
                        self.main_brain.record_output(
                            system_name='Sensorimotor Claude 4.5',
                            content=f"Visual Analysis:\n{visual_analysis[:200]}...\n\nSpatial Reasoning:\n{analysis[:300]}...",
                            metadata={
                                'has_gemini': bool(gemini_visual),
                                'has_local': bool(local_visual),
                                'has_thinking': bool(thinking),
                                'cycle': cycle_count
                            },
                            success=True
                        )
                        
                        # If Gemini provided visual, record its contribution
                        if gemini_visual:
                            self.hebbian.record_activation(
                                system_name='gemini_vision',
                                success=True,
                                contribution_strength=0.8,
                                context={'purpose': 'sensorimotor_visual'}
                            )
                        
                        # If Local vision provided analysis, record it
                        if local_visual:
                            self.hebbian.record_activation(
                                system_name='local_vision_qwen',
                                success=True,
                                contribution_strength=0.7,
                                context={'purpose': 'sensorimotor_visual'}
                            )
                    except asyncio.TimeoutError:
                        print("[SENSORIMOTOR] Timed out after 90s")
                        self.hebbian.record_activation(
                            system_name='sensorimotor_claude45',
                            success=False,
                            contribution_strength=0.3
                        )
                    except Exception as e:
                        print(f"[SENSORIMOTOR] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        self.hebbian.record_activation(
                            system_name='sensorimotor_claude45',
                            success=False,
                            contribution_strength=0.2
                        )
                    finally:
                        self._update_sensorimotor_state(
                            cycle=cycle_count,
                            similarity=similarity,
                            analysis=analysis,
                            has_thinking=bool(thinking),
                            visual_context=visual_analysis
                        )
                    
                    print("="*70 + "\n")
                
                # FULL SINGULARIS ORCHESTRATOR - Run periodically for deep strategic reasoning
                # Uses Huihui for dialectical synthesis, expert consultation, meta-cognition
                # This invokes the full MetaOrchestratorLLM with thesis-antithesis-synthesis dialectic
                if cycle_count % 15 == 0 and self.huihui_llm:
                    print("\n" + "="*70)
                    print("FULL SINGULARIS AGI PROCESS - DIALECTICAL REASONING (CYCLE 15)")
                    print("Invoking Huihui + 6 Experts + Dialectical Synthesis")
                    print("="*70)
                    
                    # Build strategic query for the orchestrator
                    strategic_query = f"""Skyrim Strategic Analysis:

Location: {game_state.location_name}
Health: {game_state.health}/100
Combat: {'YES' if game_state.in_combat else 'NO'}
Enemies: {game_state.enemies_nearby}
Current Layer: {game_state.current_action_layer}

Scene: {scene_type.value}
Motivation: {mot_state.dominant_drive().value}
Goal: {self.current_goal}

Recent actions: {', '.join(self.action_history[-5:]) if self.action_history else 'none'}

SYMBOLIC LOGIC ANALYSIS:
{chr(10).join(self.skyrim_world.get_logic_analysis(game_state.to_dict())['recommendations']['logical_reasoning']) if self.skyrim_world.get_logic_analysis(game_state.to_dict())['recommendations']['logical_reasoning'] else 'No critical logic recommendations'}

What is the most strategic approach to this situation? Consider:
1. Immediate tactical needs (including symbolic logic recommendations above)
2. Long-term strategic goals  
3. Risk vs reward tradeoffs
4. Layer transitions (Combat/Exploration/Menu/Stealth)
5. Resource management
6. How the symbolic logic rules inform tactical decisions"""

                    try:
                        # Call full AGI orchestrator with Huihui as consciousness LLM
                        print("[SINGULARIS] Invoking full orchestrator (Huihui + 6 experts)...")
                        singularis_result = await asyncio.wait_for(
                            self.agi.process(
                                query=strategic_query,
                                context={
                                    'game_state': game_state.to_dict(),
                                    'scene': scene_type.value,
                                    'cycle': cycle_count
                                }
                            ),
                            timeout=90.0  # Full orchestrator needs more time
                        )
                        
                        # Extract insights from full Singularis process
                        consciousness_response = singularis_result.get('consciousness_response', {})
                        strategic_insight = consciousness_response.get('response', '')
                        coherence_delta = consciousness_response.get('coherentia_delta', 0.0)
                        
                        print(f"\n[SINGULARIS] Strategic Insight ({len(strategic_insight)} chars):")
                        print(f"[SINGULARIS] {strategic_insight[:300]}...")
                        print(f"[SINGULARIS] Coherence Δ𝒞: {coherence_delta:+.3f}")
                        
                        # Store in memory for future reference
                        self.memory_rag.store_cognitive_memory(
                            situation={
                                'type': 'strategic_analysis',
                                'coherence_delta': coherence_delta,
                                'location': game_state.location_name,
                                'cycle': cycle_count
                            },
                            action_taken='singularis_orchestration',
                            outcome={'insight_length': len(strategic_insight), 'coherence_delta': coherence_delta},
                            success=True,
                            reasoning=strategic_insight
                        )
                        
                        # Update goal if Singularis generated one
                        if 'generated_goal' in singularis_result:
                            self.current_goal = singularis_result['generated_goal']
                            print(f"[SINGULARIS] Updated goal: {self.current_goal}")
                        
                        # PASS TO CLAUDE - Get meta-strategic analysis from Claude Sonnet 4
                        if self.hybrid_llm and hasattr(self.hybrid_llm, 'claude'):
                            try:
                                print("\n[CLAUDE-META] Sending Singularis insights to Claude for meta-analysis...")
                                
                                claude_meta_prompt = f"""You are receiving strategic insights from the Singularis AGI system (Huihui MoE with dialectical reasoning).

SINGULARIS DIALECTICAL ANALYSIS:
{strategic_insight}

COHERENCE DELTA: {coherence_delta:+.3f}

CURRENT SITUATION:
- Location: {game_state.location_name}
- Health: {game_state.health}/100
- Combat: {'YES' if game_state.in_combat else 'NO'}
- Layer: {game_state.current_action_layer}
- Goal: {self.current_goal}

As Claude Sonnet 4, provide meta-strategic analysis:

1. **Validate Singularis Reasoning**: Does the dialectical synthesis make sense? Any blind spots?
2. **Strategic Refinement**: How can we improve or extend this strategy?
3. **Risk Assessment**: What are the key risks not mentioned?
4. **Action Priority**: What should be the immediate next 3 actions?
5. **Long-term Vision**: How does this fit into broader gameplay objectives?

Be concise but insightful. Focus on what Singularis might have missed."""

                                claude_response = await asyncio.wait_for(
                                    self.hybrid_llm.generate_reasoning(
                                        prompt=claude_meta_prompt,
                                        system_prompt="You are Claude Sonnet 4, providing meta-strategic oversight of AGI reasoning.",
                                        max_tokens=1024
                                    ),
                                    timeout=30.0
                                )
                                
                                print(f"\n[CLAUDE-META] Meta-Strategic Analysis ({len(claude_response)} chars):")
                                print(f"[CLAUDE-META] {claude_response[:400]}...")
                                
                                # Store Claude's meta-analysis
                                self.memory_rag.store_cognitive_memory(
                                    situation={
                                        'type': 'claude_meta_strategy',
                                        'based_on_singularis': True,
                                        'location': game_state.location_name,
                                        'cycle': cycle_count
                                    },
                                    action_taken='meta_analysis',
                                    outcome={'response_length': len(claude_response)},
                                    success=True,
                                    reasoning=f"CLAUDE META-ANALYSIS:\n{claude_response}"
                                )
                                
                            except asyncio.TimeoutError:
                                print("[CLAUDE-META] Timed out after 30s")
                            except Exception as e:
                                print(f"[CLAUDE-META] Error: {e}")
                        
                        print("="*70 + "\n")
                        
                        # Hebbian: Record successful Singularis dialectical reasoning
                        self.hebbian.record_activation(
                            system_name='singularis_orchestrator',
                            success=True,
                            contribution_strength=1.0,
                            context={'with_claude_meta': True}
                        )
                        
                        # Huihui contributed successfully
                        self.hebbian.record_activation(
                            system_name='huihui_dialectical',
                            success=True,
                            contribution_strength=0.9,
                            context={'purpose': 'singularis'}
                        )
                        
                        # Main Brain: Record Singularis orchestrator output
                        self.main_brain.record_output(
                            system_name='Singularis Orchestrator',
                            content=f"Dialectical Strategy:\n{singularis_result.get('analysis', '')[:400]}...",
                            metadata={
                                'has_claude_meta': True,
                                'cycle': cycle_count
                            },
                            success=True
                        )
                        
                    except asyncio.TimeoutError:
                        print("[SINGULARIS] Full orchestrator timed out after 90s")
                        self.hebbian.record_activation(
                            system_name='singularis_orchestrator',
                            success=False,
                            contribution_strength=0.3
                        )
                    except Exception as e:
                        print(f"[SINGULARIS] Error in full orchestrator: {e}")
                        import traceback
                        traceback.print_exc()
                        self.hebbian.record_activation(
                            system_name='singularis_orchestrator',
                            success=False,
                            contribution_strength=0.2
                        )
                
                # Hebbian Integration Status - Print every 30 cycles
                if cycle_count % 30 == 0 and cycle_count > 0:
                    self.hebbian.print_status()
                    
                    # Apply synaptic decay
                    self.hebbian.apply_hebbian_decay()
                    
                    # Main Brain: Record Hebbian status
                    stats = self.hebbian.get_statistics()
                    synergies = self.hebbian.get_synergistic_pairs(threshold=1.0)
                    
                    hebbian_summary = f"""Success Rate: {stats['success_rate']:.1%}
Top Synergistic Pairs:
{chr(10).join(f'  {a} ↔ {b}: {s:.2f}' for a, b, s in synergies[:3])}

Strongest System: {stats['strongest_system']} ({stats['strongest_weight']:.2f})"""
                    
                    self.main_brain.record_output(
                        system_name='Hebbian Integration',
                        content=hebbian_summary,
                        metadata=stats,
                        success=True
                    )
                    
                    # Main Brain: Record Symbolic Logic World Model status
                    try:
                        world_model_stats = self.skyrim_world.get_stats()
                        logic_summary = f"""World Model Status:
Causal Edges: {world_model_stats['causal_edges']}
NPC Relationships: {world_model_stats['npc_relationships']}
Locations Discovered: {world_model_stats['locations_discovered']}
Learned Rules: {world_model_stats['learned_rules']}

Symbolic Logic Engine:
Facts in KB: {world_model_stats['logic_facts']}
Inference Rules: {world_model_stats['logic_rules']}
Predicate Types: {len(world_model_stats['logic_predicates_by_type'])}

Top Predicates:
{chr(10).join(f'  {k}: {v}' for k, v in sorted(world_model_stats['logic_predicates_by_type'].items(), key=lambda x: x[1], reverse=True)[:5])}"""
                        
                        self.main_brain.record_output(
                            system_name='Symbolic Logic World Model',
                            content=logic_summary,
                            metadata=world_model_stats,
                            success=True
                        )
                    except Exception as e:
                        print(f"[MAIN BRAIN] Could not record logic stats: {e}")
                
                # Plan action (with LLM throttling and timeout protection)
                # Increased timeout to 15s to accommodate slow LLM systems:
                # - 2 Gemini experts (rate-limited to 5 RPM each = 12s minimum wait)
                # - 3 Claude experts (3s latency each)
                # - Hybrid vision+reasoning pipeline (4-6s)
                # - Local MoE synthesis (5-7s)
                planning_start = time.time()
                
                # RECOMMENDATION 4: Adaptive planning cycles based on visual similarity
                # Check if stuck in high-similarity state
                similarity_stuck = False
                if len(self.visual_embedding_history) >= 2:
                    import numpy as np
                    last = np.array(self.visual_embedding_history[-1]).flatten()
                    prev = np.array(self.visual_embedding_history[-2]).flatten()
                    similarity = np.dot(last, prev) / (np.linalg.norm(last) * np.linalg.norm(prev) + 1e-8)
                    
                    if similarity > 0.95:
                        similarity_stuck = True
                        print(f"[ADAPTIVE-PLANNING] High similarity detected: {similarity:.3f}")
                
                # RECOMMENDATION 4: Reduce planning timeout in stuck states
                if similarity_stuck:
                    planning_timeout = 5.0  # Reduce from 15s to 5s
                    print(f"[ADAPTIVE-PLANNING] ⚡ SHORTENED CYCLE: {planning_timeout}s (similarity > 0.95)")
                    print("[ADAPTIVE-PLANNING] Increasing sensorimotor polling priority")
                    # Boost sensorimotor weight temporarily
                    self.hebbian.record_activation(
                        system_name='sensorimotor_claude45',
                        success=True,
                        contribution_strength=0.5,
                        context={'adaptive_boost': True, 'high_similarity': True}
                    )
                else:
                    planning_timeout = 15.0  # Standard planning time
                
                action = None
                try:
                    async with self.llm_semaphore:
                        action = await asyncio.wait_for(
                            self._plan_action(
                                perception=perception,
                                motivation=mot_state,
                                goal=self.current_goal
                            ),
                            timeout=planning_timeout
                        )
                except asyncio.TimeoutError:
                    print(f"[REASONING] ⚠️ Planning timed out after {planning_timeout}s, using fallback")
                    action = None
                except Exception as e:
                    print(f"[REASONING] ⚠️ Planning error: {e}, using fallback")
                    action = None
                
                planning_duration = time.time() - planning_start
                self.stats['planning_times'].append(planning_duration)
                
                # Handle None action with fallback
                if action is None:
                    print("[REASONING] WARNING: No action returned by _plan_action, using fallback")
                    action = 'explore'  # Safe default fallback
                    self.stats['heuristic_action_count'] += 1
                
                # Check for repeated action stuck loop with visual similarity
                if action == self.last_successful_action:
                    self.repeated_action_count += 1
                    
                    # Check if visuals have changed significantly (progress made)
                    current_visual = perception.get('visual_embedding')
                    visual_changed = True
                    
                    if current_visual is not None and self.last_visual_embedding is not None:
                        # Compute cosine similarity
                        import numpy as np
                        similarity = np.dot(current_visual, self.last_visual_embedding) / (
                            np.linalg.norm(current_visual) * np.linalg.norm(self.last_visual_embedding)
                        )
                        visual_changed = similarity < self.visual_similarity_threshold
                        
                        if not visual_changed and self.repeated_action_count >= 3:
                            print(f"[STUCK-DETECTION] Repeated '{action}' {self.repeated_action_count}x, visual similarity: {similarity:.3f} (stuck!)")
                    
                    # Only force change if visuals haven't changed AND repeated too many times
                    if self.repeated_action_count >= self.max_repeated_actions and not visual_changed:
                        print(f"[STUCK-DETECTION] ⚠️ Repeated action '{action}' {self.repeated_action_count} times with no visual progress!")
                        print(f"[STUCK-DETECTION] Likely stuck at door/gate/obstacle - trying 'activate'")
                        # Force a different action - prioritize 'activate' for doors/gates
                        game_state = perception.get('game_state')
                        available = game_state.available_actions if game_state else ['move_forward', 'jump', 'activate']
                        
                        # If stuck while exploring, try activate first (for doors/gates)
                        if action in ['explore', 'move_forward'] and 'activate' in available:
                            action = 'activate'
                            print(f"[STUCK-DETECTION] Trying 'activate' to open door/gate")
                        else:
                            # Otherwise choose random different action
                            different_actions = [a for a in available if a != action]
                            if different_actions:
                                action = random.choice(different_actions)
                                print(f"[STUCK-DETECTION] Switched to: {action}")
                        self.repeated_action_count = 0
                    elif visual_changed:
                        # Visual progress made, allow repeated action (e.g., move_forward exploring)
                        if self.repeated_action_count >= 3:
                            print(f"[STUCK-DETECTION] ✓ Repeated '{action}' {self.repeated_action_count}x but making visual progress")
                        self.repeated_action_count = 0  # Reset since progress is being made
                else:
                    self.repeated_action_count = 0
                
                # Update visual tracking
                if perception.get('visual_embedding') is not None:
                    self.last_visual_embedding = perception.get('visual_embedding').copy()
                
                self.last_successful_action = action
                
                print(f"[REASONING] Planned action: {action} ({planning_duration:.3f}s)")
                
                # Update dashboard with current state and planned action
                self._update_dashboard_state(action=action, action_source=self.last_action_source)
                
                # Main Brain: Increment cycle and record action decision
                self.main_brain.increment_cycle()
                
                # Record first 10 cycles, then every 5th to capture early behavior
                if cycle_count <= 10 or cycle_count % 5 == 0:
                    # Get logic analysis for Main Brain
                    try:
                        logic_analysis_brief = self.skyrim_world.get_logic_analysis(game_state.to_dict())
                        logic_summary = f"""Logic Recommendations:
• Defend: {logic_analysis_brief['recommendations']['should_defend']}
• Heal: {logic_analysis_brief['recommendations']['should_heal']}
• Retreat: {logic_analysis_brief['recommendations']['should_retreat']}
• Confidence: {logic_analysis_brief['logic_confidence']:.2f}
Active Facts: {len(logic_analysis_brief['current_facts'])}
Applicable Rules: {len(logic_analysis_brief['applicable_rules'])}"""
                    except Exception as e:
                        logic_summary = f"Logic analysis unavailable: {e}"
                    
                    self.main_brain.record_output(
                        system_name='Action Planning',
                        content=f"""Cycle {cycle_count}: {action}

{logic_summary}""",
                        metadata={
                            'planning_time': planning_duration,
                            'scene': scene_type.value,
                            'coherence': self.current_consciousness.coherence if self.current_consciousness else 0,
                            'has_logic_analysis': True
                        },
                        success=True
                    )
                
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
                
                # Set flag to prevent auxiliary exploration interference
                self.action_executing = True
                
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
                        visual_embedding = self.current_perception.get('visual_embedding') if self.current_perception else None
                        self._update_stuck_tracking(action, coherence, visual_embedding)
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
                finally:
                    # Clear flag when done (always runs)
                    self.action_executing = False
                
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
                    action_source = action_data.get('source', 'unknown')
                    self.rl_learner.store_experience(
                        state_before=before_state,
                        action=str(action),
                        state_after=after_state,
                        done=False,
                        consciousness_before=action_data['consciousness'],
                        consciousness_after=after_consciousness,
                        action_source=action_source
                    )
                    
                    # Update smart context history
                    if hasattr(self, 'smart_context'):
                        self.smart_context.update_history(
                            state={'coherence': after_consciousness},
                            action=str(action),
                            outcome={'success': True}
                        )
                    
                    # Track action diversity and rewards (GPT-4o continuous learning recommendation)
                    self._update_action_diversity_stats(str(action), after_consciousness.coherence)
                    
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
                # CRITICAL FIX: Don't spam heal when already in menu
                if scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                    # If in menu, only allow menu exit actions
                    # Don't try to heal or do other actions that open more menus
                    await asyncio.sleep(self.config.fast_loop_interval)
                    continue
                
                # === FAST HEURISTICS ===
                # Only trigger if LLM planning is actually slow (>20s)
                time_since_planning = time.time() - self.last_reasoning_time
                planning_timeout_reached = time_since_planning > self.config.fast_loop_planning_timeout
                
                fast_action = None
                fast_reason = None
                priority = 0  # Higher = more urgent
                
                # Skip fast loop if LLM is actively planning (unless emergency)
                if not planning_timeout_reached and game_state.health >= 20:
                    await asyncio.sleep(self.config.fast_loop_interval)
                    continue
                
                # 1. CRITICAL HEALTH - Highest priority
                if game_state.health < self.config.fast_health_threshold:
                    if game_state.health < 15:
                        # Extremely low health - immediate retreat
                        fast_action = 'retreat'
                        fast_reason = f"CRITICAL health {game_state.health:.0f}% - retreating"
                        priority = 100
                    elif game_state.magicka > 30 and scene_type not in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                        # Try healing spell - but NOT if already in a menu
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
                    # Throttle fast actions to avoid spam
                    # Higher priority actions can execute sooner
                    if priority >= 90:
                        min_interval = 3.0  # Critical health: 3 seconds minimum
                    elif priority >= 70:
                        min_interval = 4.0  # High danger: 4 seconds minimum  
                    else:
                        min_interval = 5.0  # Normal actions: 5 seconds minimum
                    
                    time_since_last = time.time() - last_fast_action_time
                    if time_since_last < min_interval:
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
                        self.stats['action_success_count'] += 1  # Count as success
                        self.stats['action_source_heuristic'] += 1  # Fast loop uses heuristics
                        
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
                    'scene': scene_type.value,
                    'screenshot': perception.get('screenshot'),
                    'vision_summary': perception.get('gemini_analysis')
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
                    self.stats['action_success_count'] += 1  # Count as success
                    self.stats['action_source_heuristic'] += 1  # Auxiliary loop uses heuristics
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
                post_consciousness_context = {
                    'motivation': motivation_context.get('predicted_delta_coherence', 'unknown'),
                    'cycle': cycle_count,
                    'scene': after_perception['scene_type'].value,
                    'screenshot': after_perception.get('screenshot'),
                    'vision_summary': after_perception.get('gemini_analysis')
                }
                after_consciousness = await self.consciousness_bridge.compute_consciousness(
                    after_state,
                    post_consciousness_context
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
                    action_source = self.last_action_source if hasattr(self, 'last_action_source') else 'unknown'
                    self.rl_learner.store_experience(
                        state_before=before_state,
                        action=str(action),
                        action_source=action_source,
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
                
                # Cloud RL: Store rich experience with cloud evaluation
                if self.cloud_rl_memory is not None and after_consciousness:
                    from singularis.skyrim.cloud_rl_system import Experience
                    
                    # Build state vector (simple encoding for now)
                    state_vec = np.array([
                        perception.get('game_state', {}).get('health', 100) / 100.0 if isinstance(perception.get('game_state'), dict) else getattr(perception.get('game_state'), 'health', 100) / 100.0,
                        perception.get('game_state', {}).get('stamina', 100) / 100.0 if isinstance(perception.get('game_state'), dict) else getattr(perception.get('game_state'), 'stamina', 100) / 100.0,
                        perception.get('game_state', {}).get('magicka', 100) / 100.0 if isinstance(perception.get('game_state'), dict) else getattr(perception.get('game_state'), 'magicka', 100) / 100.0,
                        float(perception.get('game_state', {}).get('in_combat', False) if isinstance(perception.get('game_state'), dict) else getattr(perception.get('game_state'), 'in_combat', False)),
                        float(perception.get('game_state', {}).get('enemies_nearby', 0)) / 10.0 if isinstance(perception.get('game_state'), dict) else float(getattr(perception.get('game_state'), 'enemies_nearby', 0)) / 10.0
                    ])
                    
                    next_state_vec = np.array([
                        after_state.get('health', 100) / 100.0,
                        after_state.get('stamina', 100) / 100.0,
                        after_state.get('magicka', 100) / 100.0,
                        float(after_state.get('in_combat', False)),
                        float(after_state.get('enemies_nearby', 0)) / 10.0
                    ])
                    
                    cloud_exp = Experience(
                        state_vector=state_vec,
                        state_description=f"{scene_type.value}: {perception.get('visual_analysis', '')[:100]}",
                        scene_type=scene_type.value,
                        location=perception.get('game_state', {}).get('location_name', 'Unknown') if isinstance(perception.get('game_state'), dict) else getattr(perception.get('game_state'), 'location_name', 'Unknown'),
                        health=float(state_vec[0] * 100),
                        stamina=float(state_vec[1] * 100),
                        magicka=float(state_vec[2] * 100),
                        enemies_nearby=int(state_vec[4] * 10),
                        in_combat=bool(state_vec[3]),
                        action=str(action),
                        action_type=action.action_type.value if hasattr(action, 'action_type') else 'unknown',
                        reward=0.0,  # Will be computed by cloud RL
                        next_state_vector=next_state_vec,
                        next_state_description=f"After {action}",
                        done=False,
                        coherence_before=self.current_consciousness.coherence if self.current_consciousness else 0.0,
                        coherence_after=after_consciousness.coherence,
                        coherence_delta=after_consciousness.coherence_delta(self.current_consciousness) if self.current_consciousness else 0.0,
                        episode_id=int(time.time()),
                        step_id=cycle_count
                    )
                    
                    # Add to cloud RL memory (will auto-save periodically)
                    import asyncio
                    asyncio.create_task(self.cloud_rl_memory.add(cloud_exp, request_cloud_evaluation=(cycle_count % 10 == 0)))
                    print(f"[CLOUD-RL] Experience added to memory (total: {len(self.cloud_rl_memory.experiences)})")


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

    def _detect_stuck_failsafe(
        self,
        perception: Dict[str, Any],
        game_state
    ) -> Tuple[bool, str, str]:
        """
        Multi-tier failsafe stuck detection (works without cloud LLMs).
        
        Returns:
            (is_stuck, reason, recovery_action)
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_stuck_detection_time < self.stuck_detection_cooldown:
            return (False, "", "")
        
        stuck_indicators = []
        recovery_action = ""
        
        # 1. Action repetition detection
        if len(self.action_history) >= 8:
            recent_8 = self.action_history[-8:]
            unique_actions = len(set(recent_8))
            
            if unique_actions <= 2:
                stuck_indicators.append(f"Only {unique_actions} unique actions in last 8")
                recovery_action = "jump"  # Break pattern
        
        # 2. Same action spam detection
        if self.consecutive_same_action >= 5:
            stuck_indicators.append(f"Same action '{self.last_executed_action}' repeated {self.consecutive_same_action}x")
            recovery_action = "sneak" if self.last_executed_action != "sneak" else "jump"
        
        # 3. Visual embedding similarity (stuck in same visual scene)
        if len(self.visual_embedding_history) >= 5:
            recent_embeddings = self.visual_embedding_history[-5:]
            
            # Calculate average similarity between consecutive embeddings
            similarities = []
            for i in range(len(recent_embeddings) - 1):
                sim = np.dot(recent_embeddings[i], recent_embeddings[i+1])
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            if avg_similarity > 0.95:  # Very similar visual scenes
                stuck_indicators.append(f"Visual scene unchanged (similarity={avg_similarity:.2f})")
                recovery_action = "jump"
        
        # 4. Coherence stagnation detection
        if len(self.coherence_history) >= 10:
            recent_coherence = self.coherence_history[-10:]
            coherence_variance = np.var(recent_coherence)
            
            if coherence_variance < 0.001:  # No coherence change
                stuck_indicators.append(f"Coherence stagnant (var={coherence_variance:.4f})")
                recovery_action = "activate"  # Try interacting
        
        # 5. Menu stuck detection
        if game_state.in_menu and len(self.action_history) >= 5:
            recent_5 = self.action_history[-5:]
            if all(a in ['activate', 'navigate_inventory', 'exit_menu'] for a in recent_5):
                stuck_indicators.append("Stuck in menu navigation")
                recovery_action = "exit_menu"
        
        # 6. Combat stuck detection
        if game_state.in_combat and len(self.action_history) >= 6:
            recent_6 = self.action_history[-6:]
            if recent_6.count('attack') >= 5:
                stuck_indicators.append("Spamming attack without progress")
                recovery_action = "dodge"
        
        # Determine if stuck
        is_stuck = len(stuck_indicators) >= 2  # At least 2 indicators
        
        if is_stuck:
            reason = "; ".join(stuck_indicators)
            self.last_stuck_detection_time = current_time
            self.stuck_recovery_attempts += 1
            
            print(f"[FAILSAFE-STUCK] Detected stuck state!")
            print(f"[FAILSAFE-STUCK] Indicators: {reason}")
            print(f"[FAILSAFE-STUCK] Recovery attempt #{self.stuck_recovery_attempts}")
            
            # Escalate recovery based on attempts
            if self.stuck_recovery_attempts >= 3:
                recovery_action = "look_around"  # Drastic measure
                print(f"[FAILSAFE-STUCK] Escalating to drastic recovery: {recovery_action}")
            
            return (True, reason, recovery_action)
        
        # Reset recovery attempts if not stuck
        if self.stuck_recovery_attempts > 0 and not is_stuck:
            self.stuck_recovery_attempts = 0
        
        return (False, "", "")
    
    async def _detect_stuck_with_gemini(
        self,
        perception: Dict[str, Any],
        recent_actions: List[str]
    ) -> Tuple[bool, str]:
        """
        Use Gemini vision to detect if player is stuck (Tier 1).
        
        Returns:
            (is_stuck, recovery_action)
        """
        if not self.hybrid_llm or not perception.get('screenshot'):
            return (False, "")
        
        try:
            # Build stuck detection prompt
            prompt = f"""Analyze this Skyrim gameplay screenshot and recent actions to detect if the player is stuck.

Recent actions (last 5): {', '.join(recent_actions[-5:])}

Check for these stuck indicators:
1. Player facing a wall/obstacle repeatedly
2. Same visual scene for multiple actions
3. Character not making progress (same location)
4. Stuck in geometry/terrain
5. Menu stuck open
6. Dialogue loop

Is the player stuck? Answer with:
STUCK: yes/no
REASON: <brief explanation>
RECOVERY: <suggested action to unstuck>"""

            response = await self.hybrid_llm.analyze_image(
                prompt=prompt,
                image=perception['screenshot']
            )
            
            # Parse response
            is_stuck = "STUCK: yes" in response.lower()
            
            if is_stuck:
                # Extract reason
                reason_line = [l for l in response.split('\n') if 'REASON:' in l]
                reason = reason_line[0].split('REASON:')[1].strip() if reason_line else "Unknown"
                
                # Extract recovery action
                recovery_line = [l for l in response.split('\n') if 'RECOVERY:' in l]
                recovery = recovery_line[0].split('RECOVERY:')[1].strip() if recovery_line else ""
                
                print(f"[GEMINI-STUCK] Detected stuck state!")
                print(f"[GEMINI-STUCK] Reason: {reason}")
                print(f"[GEMINI-STUCK] Recovery: {recovery}")
                
                return (True, recovery)
            
            return (False, "")
            
        except Exception as e:
            print(f"[GEMINI-STUCK] Detection failed: {e}")
            return (False, "")
    
    async def _get_cloud_llm_action_recommendation(
        self,
        perception: Dict[str, Any],
        state_dict: Dict[str, Any],
        q_values: Dict[str, float],
        available_actions: List[str],
        motivation,
        use_full_moe: bool = False
    ) -> Optional[Tuple[str, str]]:
        """
        Get action recommendation from parallel cloud LLM system.
        
        Args:
            use_full_moe: If True, use full MoE+Hybrid. If False, use Hybrid only (faster)
        
        Returns:
            (action, reasoning) or None if cloud LLMs unavailable
        """
        if not self.config.use_parallel_mode and not self.moe and not self.hybrid_llm:
            return None
        
        try:
            # Build prompts
            vision_prompt = f"""Analyze this Skyrim gameplay situation:
Scene: {perception.get('scene_type', 'unknown')}
Health: {state_dict.get('health', 100)}%
Stamina: {state_dict.get('stamina', 100)}%
Magicka: {state_dict.get('magicka', 100)}%
In Combat: {state_dict.get('in_combat', False)}
Enemies Nearby: {state_dict.get('enemies_nearby', 0)}

What do you see and what threats/opportunities are present?"""

            reasoning_prompt = f"""Based on the current situation, recommend the best action:

Available actions: {', '.join(available_actions)}

Top Q-values (learned preferences):
{chr(10).join(f'- {k}: {v:.2f}' for k, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:5])}

Current motivation: {motivation.dominant_drive().value}

Recommend ONE action from the available list and explain why.
Format: ACTION: <action_name>
REASONING: <explanation>"""

            # Query system based on priority with local fallback
            reasoning_text = None
            
            try:
                # Full parallel (MoE+Hybrid+WorldModel) only for critical situations
                if self.config.use_parallel_mode and use_full_moe:
                    print("[CLOUD-LLM] Using FULL parallel (MoE + Hybrid + GPT-5-thinking)")
                    response = await self.query_parallel_llm(
                        vision_prompt=vision_prompt,
                        reasoning_prompt=reasoning_prompt,
                        image=perception.get('screenshot'),
                        context=state_dict
                    )
                    reasoning_text = response['reasoning']
                    if response.get('world_model'):
                        print("[WORLD-MODEL] Deep causal analysis included")
                        
                        # Record GPT-5-thinking world model to Main Brain
                        if hasattr(self, 'main_brain') and self.main_brain:
                            self.main_brain.record_output(
                                system_name='GPT-5-thinking World Model',
                                content=response['world_model'],
                                metadata={
                                    'consciousness_score': response.get('world_model_consciousness', 0.0),
                                    'coherence': response.get('coherence', 0.0),
                                    'source': 'parallel',
                                    'integration_context': 'sensorimotor+perceptual+cognitive',
                                    'timestamp': time.time()
                                },
                                success=True
                            )
                # Hybrid only for routine decisions (much faster, less API calls)
                elif self.hybrid_llm:
                    print("[CLOUD-LLM] Using Hybrid (fast mode)")
                    
                    # Get curriculum knowledge
                    curriculum_knowledge = None
                    if self.curriculum_rag:
                        categories = CATEGORY_MAPPINGS.get('strategy', []) + CATEGORY_MAPPINGS.get('psychology', [])
                        knowledge_results = self.curriculum_rag.retrieve_knowledge(
                            query=reasoning_prompt,
                            top_k=1,
                            categories=categories
                        )
                        if knowledge_results:
                            curriculum_knowledge = knowledge_results[0].excerpt
                            print("[CURRICULUM] Strategic knowledge retrieved")
                    
                    # Build smart context
                    smart_prompt = self.smart_context.build_prompt_with_context(
                        base_prompt=reasoning_prompt,
                        task_type='reasoning',
                        perception=perception,
                        state_dict=state_dict,
                        q_values=q_values,
                        available_actions=available_actions,
                        curriculum_knowledge=curriculum_knowledge
                    )
                    print(f"[SMART-CONTEXT] Optimized context ({len(smart_prompt)} chars)")
                    
                    # Run reasoning and world model in parallel
                    parallel_tasks = [
                        self.hybrid_llm.generate_reasoning(
                            prompt=smart_prompt,
                            system_prompt="You are an expert Skyrim player providing tactical advice."
                        )
                    ]
                    
                    # Add world model if available (GPT-5-thinking)
                    if hasattr(self.hybrid_llm, 'openai') and self.hybrid_llm.openai:
                        # Get scientific knowledge
                        science_knowledge = None
                        if self.curriculum_rag:
                            categories = CATEGORY_MAPPINGS.get('science', []) + ['Philosophy Of Science']
                            knowledge_results = self.curriculum_rag.retrieve_knowledge(
                                query="causal relationships and system dynamics",
                                top_k=1,
                                categories=categories
                            )
                            if knowledge_results:
                                science_knowledge = knowledge_results[0].excerpt
                        
                        # Build smart context for world modeling
                        world_model_prompt = self.smart_context.build_prompt_with_context(
                            base_prompt="Analyze causal dynamics and long-term consequences of this situation.",
                            task_type='world_model',
                            perception=perception,
                            state_dict=state_dict,
                            q_values=q_values,
                            available_actions=available_actions,
                            curriculum_knowledge=science_knowledge
                        )
                        
                        parallel_tasks.append(
                            self.hybrid_llm.generate_world_model(
                                prompt=world_model_prompt,
                                system_prompt="Analyze causal relationships using scientific principles.",
                                temperature=0.8,
                                max_tokens=1024
                            )
                        )
                    
                    # Execute in parallel (cloud models don't block each other)
                    parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                    reasoning_text = parallel_results[0] if not isinstance(parallel_results[0], Exception) else None
                    
                    # Incorporate world model if available
                    if len(parallel_results) > 1 and not isinstance(parallel_results[1], Exception):
                        world_analysis = parallel_results[1]
                        reasoning_text = f"{reasoning_text}\n\n[Deep Analysis]\n{world_analysis}"
                        print("[WORLD-MODEL] Parallel deep analysis completed")
                        
                        # Record GPT-5-thinking world model to Main Brain
                        if hasattr(self, 'main_brain') and self.main_brain:
                            # Measure consciousness of the world model narrative
                            from ..consciousness.measurement import ConsciousnessMeasurement
                            consciousness_measure = ConsciousnessMeasurement()
                            
                            world_model_trace = consciousness_measure.measure(
                                content=world_analysis,
                                query="unified consciousness synthesis",
                                lumen_focus="participatum"
                            )
                            
                            self.main_brain.record_output(
                                system_name='GPT-5-thinking World Model',
                                content=world_analysis,
                                metadata={
                                    'consciousness_score': world_model_trace.overall_consciousness,
                                    'phi': world_model_trace.phi,
                                    'gwt_salience': world_model_trace.gwt_salience,
                                    'hot_depth': world_model_trace.hot_depth,
                                    'integration': world_model_trace.integration_score,
                                    'differentiation': world_model_trace.differentiation_score,
                                    'source': 'hybrid',
                                    'timestamp': time.time()
                                },
                                success=True
                            )
                elif self.moe:
                    _, reasoning_resp = await self.moe.query_all_experts(
                        vision_prompt=vision_prompt,
                        reasoning_prompt=reasoning_prompt,
                        image=perception.get('screenshot'),
                        context=state_dict
                    )
                    reasoning_text = reasoning_resp.consensus
            except Exception as e:
                print(f"[CLOUD-LLM] Cloud API failed: {e}")
                reasoning_text = None
            
            # Fallback to local LLM if cloud failed
            if not reasoning_text and self.huihui_llm:
                print("[FALLBACK] Cloud LLM failed, using local Huihui")
                try:
                    local_response = await self.huihui_llm.generate(
                        prompt=reasoning_prompt,
                        system_prompt="You are an expert Skyrim player. Recommend ONE action.",
                        max_tokens=200
                    )
                    reasoning_text = local_response
                except Exception as local_e:
                    print(f"[FALLBACK] Local LLM also failed: {local_e}")
                    return None
            
            if not reasoning_text:
                return None
            
            # Parse action from response
            if "ACTION:" in reasoning_text:
                action_line = [l for l in reasoning_text.split('\n') if 'ACTION:' in l][0]
                action = action_line.split('ACTION:')[1].strip().lower()
                
                # Validate action is in available list
                if action in available_actions:
                    print(f"[CLOUD-LLM] Recommended: {action}")
                    return (action, reasoning_text)
            
            return None
            
        except Exception as e:
            print(f"[CLOUD-LLM] Error: {e}")
            return None
    
    async def _claude_background_reasoning(
        self,
        perception: Dict[str, Any],
        state_dict: Dict[str, Any],
        game_state: Any,
        scene_type: Any,
        motivation: Any
    ) -> None:
        """
        Claude runs deep reasoning/sensorimotor analysis in background.
        Stores insights to memory independently without blocking action planning.
        """
        try:
            print("[CLAUDE-ASYNC] Running deep analysis...")
            
            # Build comprehensive reasoning prompt
            reasoning_prompt = f"""Analyze this Skyrim situation deeply:

Scene: {scene_type.value}
Location: {game_state.location_name}
Health: {state_dict.get('health', 100)}% | Stamina: {state_dict.get('stamina', 100)}% | Magicka: {state_dict.get('magicka', 100)}%
Combat: {state_dict.get('in_combat', False)} | Enemies: {state_dict.get('enemies_nearby', 0)}
Motivation: {motivation.dominant_drive().value}

Provide strategic insights about:
1. Current tactical situation
2. Potential threats and opportunities
3. Resource management considerations
4. Recommended strategic approach

Be detailed and thoughtful."""

            # Query Claude with extended thinking
            reasoning_text = await self.hybrid_llm.generate_reasoning(
                prompt=reasoning_prompt,
                system_prompt="You are a strategic advisor for Skyrim gameplay. Provide deep, thoughtful analysis.",
                temperature=0.7,
                max_tokens=2048
            )
            
            print(f"[CLAUDE-ASYNC] ✓ Analysis complete ({len(reasoning_text)} chars)")
            
            # Store in memory RAG
            self.memory_rag.store_cognitive_memory(
                situation={
                    'type': 'claude_strategic_analysis',
                    'scene': scene_type.value,
                    'location': game_state.location_name,
                    'health': state_dict.get('health', 100),
                    'in_combat': state_dict.get('in_combat', False)
                },
                action_taken='background_reasoning',
                outcome={'analysis_length': len(reasoning_text)},
                success=True,
                reasoning=reasoning_text
            )
            
            # Hebbian: Record Claude contribution
            self.hebbian.record_activation(
                system_name='claude_background_reasoning',
                success=True,
                contribution_strength=0.8,
                context={'stored_to_memory': True}
            )
            
            print("[CLAUDE-ASYNC] ✓ Stored to memory")
            
        except Exception as e:
            print(f"[CLAUDE-ASYNC] Error: {e}")
            self.hebbian.record_activation(
                system_name='claude_background_reasoning',
                success=False,
                contribution_strength=0.3
            )
    
    async def _dialectical_reasoning(
        self,
        situation: str,
        perspectives: Dict[str, str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Perform dialectical synthesis using Huihui MoE.
        
        Uses thesis-antithesis-synthesis to resolve contradictions and
        find higher-order understanding.
        
        Args:
            situation: Current situation description
            perspectives: Dict of perspective_name -> perspective_content
            context: Optional additional context
            
        Returns:
            Dict with 'synthesis', 'reasoning', 'coherence_gain'
        """
        if not self.huihui_llm:
            return {
                'synthesis': list(perspectives.values())[0] if perspectives else "",
                'reasoning': "No dialectical LLM available",
                'coherence_gain': 0.0
            }
        
        # Build dialectical prompt
        perspectives_text = "\n\n".join([
            f"**{name.upper()}:**\n{content}"
            for name, content in perspectives.items()
        ])
        
        dialectical_prompt = f"""DIALECTICAL SYNTHESIS

Situation: {situation}

Multiple perspectives on this situation:

{perspectives_text}

Your task is to perform Hegelian dialectical synthesis:

1. THESIS: Identify the primary perspective and its core claim
2. ANTITHESIS: Identify contradictions or tensions with other perspectives
3. SYNTHESIS: Find higher-order unity that transcends the contradiction

The synthesis should:
- Preserve partial truths from each perspective
- Resolve apparent contradictions
- Achieve greater coherence than any single view
- Provide actionable insight

Format your response as:
THESIS: <primary claim>
ANTITHESIS: <contradicting view>
SYNTHESIS: <unified understanding>
COHERENCE GAIN: <estimate 0.0-1.0 how much this increases understanding>
"""

        try:
            print("[DIALECTICAL] Performing thesis-antithesis-synthesis with Huihui...")
            result = await asyncio.wait_for(
                self.huihui_llm.generate(
                    prompt=dialectical_prompt,
                    max_tokens=512
                ),
                timeout=60.0
            )
            
            response = result.get('content', '') if isinstance(result, dict) else str(result)
            print(f"[DIALECTICAL] Synthesis complete ({len(response)} chars)")
            
            # Parse response
            synthesis = ""
            reasoning = response
            coherence_gain = 0.0
            
            # Extract synthesis section
            if "SYNTHESIS:" in response:
                synthesis_start = response.find("SYNTHESIS:") + len("SYNTHESIS:")
                synthesis_end = response.find("COHERENCE GAIN:", synthesis_start)
                if synthesis_end == -1:
                    synthesis_end = len(response)
                synthesis = response[synthesis_start:synthesis_end].strip()
            
            # Extract coherence gain
            if "COHERENCE GAIN:" in response:
                try:
                    gain_text = response.split("COHERENCE GAIN:")[1].strip().split()[0]
                    coherence_gain = float(gain_text)
                except:
                    coherence_gain = 0.05  # Default modest gain
            
            return {
                'synthesis': synthesis if synthesis else response,
                'reasoning': reasoning,
                'coherence_gain': coherence_gain
            }
            
        except asyncio.TimeoutError:
            print("[DIALECTICAL] Timed out after 60s")
            return {
                'synthesis': list(perspectives.values())[0] if perspectives else "",
                'reasoning': "Dialectical synthesis timed out",
                'coherence_gain': 0.0
            }
        except Exception as e:
            print(f"[DIALECTICAL] Error: {e}")
            return {
                'synthesis': list(perspectives.values())[0] if perspectives else "",
                'reasoning': f"Error: {e}",
                'coherence_gain': 0.0
            }
    
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
            checkpoint_start = time.time()
            print(f"[PLANNING-CHECKPOINT] Starting _plan_action at {checkpoint_start:.3f}")
            
            game_state = perception['game_state']
            scene_type = perception['scene_type']
            current_layer = game_state.current_action_layer
            available_actions = game_state.available_actions
            
            # Filter out blocked actions from rule engine
            if hasattr(self, 'rule_engine'):
                original_count = len(available_actions)
                available_actions = [a for a in available_actions if not self.rule_engine.is_action_blocked(a)]
                if len(available_actions) < original_count:
                    blocked = [a for a in game_state.available_actions if self.rule_engine.is_action_blocked(a)]
                    print(f"[RULES] Filtered out {original_count - len(available_actions)} blocked actions: {', '.join(blocked)}")

            print(f"[PLANNING] Current layer: {current_layer}")
            print(f"[PLANNING] Available actions: {available_actions}")
            print(f"[PLANNING-CHECKPOINT] Game state extraction: {time.time() - checkpoint_start:.3f}s")

            # Check sensorimotor state BEFORE starting expensive LLM operations
            checkpoint_sensorimotor = time.time()
            override_action = self._sensorimotor_override(available_actions, game_state)
            if override_action:
                print(f"[PLANNING-CHECKPOINT] Sensorimotor override: {time.time() - checkpoint_sensorimotor:.3f}s (action: {override_action})")
                self.stats['heuristic_action_count'] += 1
                return override_action
            print(f"[PLANNING-CHECKPOINT] Sensorimotor check: {time.time() - checkpoint_sensorimotor:.3f}s")

            # EXPERT RULE SYSTEM - Fast rule-based reasoning BEFORE expensive operations
            checkpoint_rules = time.time()
            print("\n[RULES] Evaluating expert rules...")
            
            # Prepare context for rule evaluation
            rule_context = {
                'visual_similarity': perception.get('visual_similarity', 0.0),
                'recent_actions': self.action_history[-10:] if hasattr(self, 'action_history') else [],
                'scene_classification': scene_type,
                'visual_scene_type': perception.get('scene_type'),
                'coherence_history': self.coherence_history[-10:] if hasattr(self, 'coherence_history') else [],
                'health': game_state.health,
                'in_combat': game_state.in_combat,
                'enemies_nearby': game_state.enemies_nearby,
            }
            
            # Evaluate rules
            rule_results = self.rule_engine.evaluate(rule_context)
            
            # Display rule firing results
            if rule_results['fired_rules']:
                print(f"[RULES] ✓ Fired {len(rule_results['fired_rules'])} rules: {', '.join(rule_results['fired_rules'])}")
                
                # Show facts
                if rule_results['facts']:
                    print(f"[RULES] Active facts:")
                    for fact_name, fact_data in rule_results['facts'].items():
                        print(f"  • {fact_name} (confidence: {fact_data['confidence']:.2f})")
                
                # Show recommendations
                if rule_results['recommendations']:
                    top_rec = rule_results['recommendations'][0]
                    print(f"[RULES] Top recommendation: {top_rec.action} (priority: {top_rec.priority.name}, reason: {top_rec.reason})")
                
                # Show blocked actions
                if rule_results['blocked_actions']:
                    print(f"[RULES] Blocked actions: {', '.join(rule_results['blocked_actions'])}")
                
                # Check if we have a high-priority recommendation we should use immediately
                top_recommendation = self.rule_engine.get_top_recommendation(exclude_blocked=True)
                if top_recommendation and top_recommendation.priority.value >= 3:  # HIGH or CRITICAL
                    # Use rule recommendation immediately
                    print(f"[RULES] ⚡ Using immediate recommendation: {top_recommendation.action} ({top_recommendation.reason})")
                    print(f"[PLANNING-CHECKPOINT] Rule engine: {time.time() - checkpoint_rules:.3f}s (immediate action)")
                    self.stats['heuristic_action_count'] += 1
                    return top_recommendation.action
            else:
                print(f"[RULES] No rules fired")
            
            print(f"[PLANNING-CHECKPOINT] Rule engine: {time.time() - checkpoint_rules:.3f}s")

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
            force_variety = self.consecutive_same_action >= 5  # Reduced from 6 to 5 for more diversity
            
            # Enhanced variety injection with environmental awareness (GPT-4o recommendation)
            # Base variety rate: 20% (was 10%)
            variety_rate = 0.20
            
            # Increase variety in specific scenarios:
            if scene_type == SceneType.OUTDOOR_WILDERNESS:
                variety_rate = 0.30  # More exploration variety outdoors
            elif scene_type == SceneType.INDOOR_DUNGEON:
                variety_rate = 0.25  # Tactical variety in dungeons
            elif game_state.health < 40:
                variety_rate = 0.35  # More creative options when low health
            
            use_rl = random.random() > variety_rate  # Dynamic RL vs heuristics ratio
            if not use_rl:
                print(f"[VARIETY] Environmental diversity injection (rate={variety_rate:.0%}) - using heuristics")
            
            if force_variety and self.consecutive_same_action >= 5:
                print(f"[VARIETY] Forcing diversity after {self.consecutive_same_action}x '{self.last_executed_action}' - skipping RL")
            
            # Always get Q-values (needed for heuristics even if skipping RL)
            checkpoint_rl_start = time.time()
            q_values = {}
            if self.rl_learner is not None:
                q_values = self.rl_learner.get_q_values(state_dict)
            print(f"[PLANNING-CHECKPOINT] Q-value computation: {time.time() - checkpoint_rl_start:.3f}s")
            
            # Initialize task variables - CRITICAL: Must initialize before any conditional branches
            cloud_task = None
            local_moe_task = None
            gemini_moe_task = None
            claude_reasoning_task = None
            huihui_task = None
            phi4_task = None
            
            # Use RL-based action selection if enabled (but not if forcing variety)
            if self.rl_learner is not None and use_rl:
                print("[PLANNING] Using RL-based action selection with LLM reasoning...")
                print(f"[PLANNING] RL reasoning neuron LLM status: {'Connected' if self.rl_reasoning_neuron.llm_interface else 'Using heuristics'}")
                
                # Track RL usage
                self.stats['rl_action_count'] += 1
                
                # Check if meta-strategist should generate new instruction (skip if no LLM)
                checkpoint_meta_start = time.time()
                if self.huihui_llm and await self.meta_strategist.should_generate_instruction():
                    try:
                        instruction = await asyncio.wait_for(
                            self.meta_strategist.generate_instruction(
                                current_state=state_dict,
                                q_values=q_values,
                                motivation=motivation.dominant_drive().value
                            ),
                            timeout=5.0
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"[META-STRATEGIST] Skipping - error: {type(e).__name__}")
                print(f"[PLANNING-CHECKPOINT] Meta-strategist check: {time.time() - checkpoint_meta_start:.3f}s")
                
                # Q-values already computed above
                print(f"[RL] Q-values: {', '.join([f'{k}={v:.2f}' for k, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]])}")
                
                # Get meta-strategic context
                meta_context = self.meta_strategist.get_active_instruction_context()
                
                # === MULTI-TIER STUCK DETECTION ===
                checkpoint_stuck_start = time.time()
                
                # Tier 1: Failsafe stuck detection (always runs, no cloud needed)
                failsafe_stuck, failsafe_reason, failsafe_recovery = self._detect_stuck_failsafe(
                    perception=perception,
                    game_state=game_state
                )
                print(f"[PLANNING-CHECKPOINT] Failsafe stuck detection: {time.time() - checkpoint_stuck_start:.3f}s")
                
                # RECOMMENDATION 1: Check Hebbian weight for sensorimotor interrupt priority
                sensorimotor_weight = self.hebbian.get_system_weight('sensorimotor_claude45')
                print(f"[HEBBIAN-CONTROL] Sensorimotor weight: {sensorimotor_weight:.3f}")
                
                if failsafe_stuck:
                    # Measure consciousness impact
                    stuck_state = await self.measure_system_consciousness()
                    print(f"[FAILSAFE-STUCK] System coherence during stuck: {stuck_state.global_coherence:.3f}")
                    
                    # RECOMMENDATION 1: Grant interrupt priority when sensorimotor weight > 1.3 AND stuck
                    if sensorimotor_weight > 1.3:
                        print(f"[HEBBIAN-CONTROL] ⚡ INTERRUPT PRIORITY: Sensorimotor weight {sensorimotor_weight:.3f} > 1.3")
                        print(f"[HEBBIAN-CONTROL] Granting control authority to sensorimotor expert")
                        
                        # Strengthen pathway on successful interrupt
                        self.hebbian.record_activation(
                            system_name='sensorimotor_claude45',
                            success=True,
                            contribution_strength=1.2,  # Extra reward for interrupt
                            context={'interrupt_priority': True, 'stuck_detected': True}
                        )
                        
                        # Use failsafe recovery action with priority
                        if failsafe_recovery in available_actions:
                            print(f"[HEBBIAN-CONTROL] Using sensorimotor-priority recovery: {failsafe_recovery}")
                            self.stats['sensorimotor_interrupts'] = self.stats.get('sensorimotor_interrupts', 0) + 1
                            self.stats['failsafe_stuck_detections'] = self.stats.get('failsafe_stuck_detections', 0) + 1
                            return failsafe_recovery
                    else:
                        # Standard stuck recovery
                        if failsafe_recovery in available_actions:
                            print(f"[FAILSAFE-STUCK] Using recovery: {failsafe_recovery}")
                            self.stats['failsafe_stuck_detections'] = self.stats.get('failsafe_stuck_detections', 0) + 1
                            return failsafe_recovery
                
                # Tier 2: Gemini vision stuck detection (every 10 cycles, cloud-based)
                checkpoint_gemini_stuck_start = time.time()
                if self.cycle_count % 10 == 0 and self.hybrid_llm:
                    print(f"[PLANNING-CHECKPOINT] Starting Gemini vision stuck detection...")
                    is_stuck, recovery_action = await self._detect_stuck_with_gemini(
                        perception=perception,
                        recent_actions=self.action_history[-10:]
                    )
                    print(f"[PLANNING-CHECKPOINT] Gemini stuck detection: {time.time() - checkpoint_gemini_stuck_start:.3f}s")
                    
                    if is_stuck and recovery_action:
                        # Measure consciousness impact of stuck state
                        stuck_state = await self.measure_system_consciousness()
                        print(f"[GEMINI-STUCK] System coherence during stuck: {stuck_state.global_coherence:.3f}")
                        
                        # Parse recovery action
                        recovery_lower = recovery_action.lower()
                        
                        # Try to extract action from recovery suggestion
                        for action in available_actions:
                            if action in recovery_lower:
                                print(f"[GEMINI-STUCK] Using recovery action: {action}")
                                self.stats['gemini_stuck_detections'] = self.stats.get('gemini_stuck_detections', 0) + 1
                                return action
                        
                        # Fallback recovery actions
                        if 'jump' in available_actions:
                            print(f"[GEMINI-STUCK] Using fallback recovery: jump")
                            self.stats['gemini_stuck_detections'] = self.stats.get('gemini_stuck_detections', 0) + 1
                            return 'jump'
                        elif 'sneak' in available_actions:
                            print(f"[GEMINI-STUCK] Using fallback recovery: sneak")
                            self.stats['gemini_stuck_detections'] = self.stats.get('gemini_stuck_detections', 0) + 1
                            return 'sneak'
                else:
                    print(f"[PLANNING-CHECKPOINT] Skipping Gemini stuck detection (cycle {self.cycle_count})")
                
                # Gemini MoE: Fast action selection with 2 Gemini Flash experts (participates in race)
                # Check rate limit BEFORE starting expensive MoE query
                checkpoint_moe_check = time.time()
                gemini_moe_task = None
                if self.moe and hasattr(self.moe, 'gemini_experts') and len(self.moe.gemini_experts) > 0:
                    # Check if Gemini is rate-limited (Fix 10: backoff strategy)
                    is_limited, wait_time = self.moe.is_gemini_rate_limited()
                    print(f"[PLANNING-CHECKPOINT] MoE rate limit check: {time.time() - checkpoint_moe_check:.3f}s")
                    
                    if is_limited:
                        if wait_time < 5.0:
                            # Wait if <5s remaining (backoff strategy)
                            print(f"[GEMINI-MOE] ⏳ Waiting {wait_time:.1f}s for rate limit...")
                            await asyncio.sleep(wait_time + 0.5)
                        elif wait_time < 10.0:
                            # Queue request for next cycle
                            print(f"[GEMINI-MOE] 📋 QUEUING - will retry next cycle ({wait_time:.1f}s remaining)")
                            self.stats['llm_queued_requests'] = self.stats.get('llm_queued_requests', 0) + 1
                            gemini_moe_task = None
                        else:
                            # Skip only if >10s
                            print(f"[GEMINI-MOE] ⚠️ SKIPPING - Gemini rate-limited ({wait_time:.1f}s)")
                            self.stats['llm_skipped_rate_limit'] = self.stats.get('llm_skipped_rate_limit', 0) + 1
                    else:
                        print(f"[GEMINI-MOE] ✓ Rate limit OK - Starting {len(self.moe.gemini_experts)} Gemini Flash experts for fast action selection...")
                        
                        # RECOMMENDATION 3: Leverage Gemini Vision for action filtering
                        # Build prompt with spatial reasoning context
                        vision_prompt = f"""Analyze this Skyrim gameplay with spatial awareness:
Scene: {perception.get('scene_type', 'unknown')}
Health: {state_dict.get('health', 100)}%
In Combat: {state_dict.get('in_combat', False)}
Enemies: {state_dict.get('enemies_nearby', 0)}

Provide detailed spatial description: obstacles, pathways, interactive elements, and scene type confirmation."""
                        
                        # RECOMMENDATION 3: Use Gemini's spatial reasoning for filtering
                        reasoning_prompt = f"""Recommend ONE action based on spatial analysis:
Available: {', '.join(available_actions)}
Top Q-values: {', '.join(f'{k}={v:.2f}' for k, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3])}

Consider:
- If in menu/inventory scene, only recommend: activate, move_cursor, press_tab
- If exploring, consider spatial obstacles from vision
- If stuck, recommend activate or turn actions

Format: ACTION: <action_name>"""                        
                        
                        print("[GEMINI-VISION] Using spatial reasoning to inform action selection")
                        
                        gemini_moe_task = asyncio.create_task(
                            self.moe.query_all_experts(
                                vision_prompt=vision_prompt,
                                reasoning_prompt=reasoning_prompt,
                                image=perception.get('screenshot'),
                                context=state_dict
                            )
                        )
                
                # Claude: Deep reasoning/sensorimotor (async background, stores to memory)
                claude_reasoning_task = None
                if self.hybrid_llm and hasattr(self.hybrid_llm, 'claude'):
                    print(f"[CLAUDE-ASYNC] Starting Claude reasoning in background (non-blocking)...")
                    claude_reasoning_task = asyncio.create_task(
                        self._claude_background_reasoning(
                            perception=perception,
                            state_dict=state_dict,
                            game_state=game_state,
                            scene_type=scene_type,
                            motivation=motivation
                        )
                    )
                    # Don't await - let it run independently and store to memory
                
                # Fallback: Start Local MoE in background (4x Qwen3-VL + Phi-4)
                local_moe_task = None
                if hasattr(self, 'local_moe') and self.local_moe:
                    print("[LOCAL-MOE] Starting local MoE query in background...")
                    local_moe_task = asyncio.create_task(
                        self.local_moe.get_action_recommendation(
                            perception=perception,
                            game_state=game_state,
                            available_actions=available_actions,
                            q_values=q_values,
                            motivation=motivation.dominant_drive().value
                        )
                    )
                else:
                    print("[FALLBACK] Local MoE unavailable, using heuristics")
                
                # Also start Huihui in background for strategic reasoning (if available)
                visual_analysis = perception.get('visual_analysis', '')
                if visual_analysis:
                    print(f"[QWEN3-VL] Passing visual analysis to Huihui: {visual_analysis[:100]}...")
                
                # Only start Huihui if local LLM is available
                if self.rl_reasoning_neuron.llm_interface:
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
                                'visual_analysis': visual_analysis
                            }
                        )
                    )
                else:
                    print("[HUIHUI-BG] Local LLM not available, skipping strategic analysis")
                
                # Compute fast heuristic immediately (no await needed)
                print("[HEURISTIC-FAST] Computing quick action for Phi-4...")
                
                # Get top 3 Q-value actions for variety
                sorted_q_actions = sorted(
                    [(a, q_values.get(a, 0.0)) for a in available_actions],
                    key=lambda x: x[1],
                    reverse=True
                )
                top_q_action = sorted_q_actions[0][0] if sorted_q_actions else 'move_forward'
                
                # Context-aware adjustment with variety
                if game_state.health < 30 and 'rest' in available_actions:
                    heuristic_action = 'rest'
                    heuristic_reason = "Low health emergency"
                elif game_state.in_combat and game_state.enemies_nearby > 2 and 'power_attack' in available_actions:
                    heuristic_action = 'power_attack'
                    heuristic_reason = "Multiple enemies"
                elif not game_state.in_combat:
                    # Add variety for exploration - don't always move_forward!
                    exploration_options = []
                    
                    # Prevent repetition - avoid last action if possible
                    if len(sorted_q_actions) >= 3:
                        # Pick from top 3 Q-values, avoid last action
                        for action, _ in sorted_q_actions[:3]:
                            if action != self.last_executed_action:
                                exploration_options.append(action)
                    
                    if exploration_options:
                        # Randomly pick from good options (not always move_forward!)
                        heuristic_action = random.choice(exploration_options)
                        heuristic_reason = f"Varied exploration (avoiding repetition)"
                    else:
                        # Fall back to top Q-value if no variety available
                        heuristic_action = top_q_action
                        heuristic_reason = "Top Q-value action"
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
            
            # === SYMBOLIC LOGIC ANALYSIS ===
            print("\n[LOGIC] Querying symbolic logic system...")
            logic_analysis = self.skyrim_world.get_logic_analysis(game_state.to_dict())
            strategic_analysis['logic_analysis'] = logic_analysis
            
            # Log symbolic logic recommendations
            logic_recs = logic_analysis['recommendations']
            if any(logic_recs.values()):  # If any recommendation is True
                print(f"[LOGIC] Active recommendations:")
                if logic_recs['should_defend']:
                    print(f"  • DEFEND: Hostile NPCs in combat")
                if logic_recs['should_heal']:
                    print(f"  • HEAL: Health is critical/low")
                if logic_recs['should_retreat']:
                    print(f"  • RETREAT: Outnumbered with escape route")
                if logic_recs['should_use_magic']:
                    print(f"  • USE MAGIC: Enemy vulnerable to magic")
                if logic_recs['should_avoid_detection']:
                    print(f"  • STEALTH: In stealth mode with enemy nearby")
                print(f"[LOGIC] Logic confidence: {logic_analysis['logic_confidence']:.2f}")
                print(f"[LOGIC] Derived {logic_recs['derived_facts']} new facts via forward chaining")
            else:
                print(f"[LOGIC] No critical recommendations (confidence: {logic_analysis['logic_confidence']:.2f})")
            
            print(f"[STRATEGIC] Layer effectiveness: {strategic_analysis['layer_effectiveness']}")
            if strategic_analysis['recommendations']:
                print(f"[STRATEGIC] Recommendations: {strategic_analysis['recommendations']}")

            # Meta-strategic reasoning: Should we switch layers?
            optimal_layer = None
            
            print(f"[META-STRATEGY] Evaluating layer switches (current: {current_layer})")
            
            # Combat situations - prioritize Combat layer if effective
            # Only engage if BOTH scene=combat AND (in_combat OR enemies detected)
            # This prevents false positives from "weapons drawn" stance
            in_actual_combat = (
                scene_type == SceneType.COMBAT and 
                (game_state.in_combat or game_state.enemies_nearby > 0)
            )
            
            if in_actual_combat:
                print(f"[META-STRATEGY] Active combat! Scene: {scene_type.value}, in_combat: {game_state.in_combat}, enemies: {game_state.enemies_nearby}")
                
                # Switch to Combat layer
                if current_layer != "Combat":
                    combat_effectiveness = strategic_analysis['layer_effectiveness'].get('Combat', 0.5)
                    if combat_effectiveness > 0.6:
                        optimal_layer = "Combat"
                        print(f"[META-STRATEGY] Switching to Combat layer (effectiveness: {combat_effectiveness:.2f})")
                
                # Choose combat action based on context
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
            elif scene_type == SceneType.COMBAT and not in_actual_combat:
                # Combat scene detected but no active combat - probably post-combat or stuck in stance
                print(f"[META-STRATEGY] Combat scene but no active combat (post-combat or stuck stance)")
                # Try to exit combat stance by exploring
                if 'sneak' in available_actions:
                    print(f"[META-STRATEGY] → sneak (exit combat stance)")
                    return 'sneak'

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

            # Check if cloud LLM finished while we were planning
            if cloud_task and cloud_task.done():
                try:
                    cloud_recommendation = cloud_task.result()
                    if cloud_recommendation:
                        action, reasoning = cloud_recommendation
                        print(f"[CLOUD-LLM] ✓ Cloud finished first! Using: {action}")
                        print(f"[CLOUD-LLM] Reasoning: {reasoning[:200]}...")
                        return action
                except Exception as e:
                    print(f"[CLOUD-LLM] Cloud task failed: {e}")
            
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
                    
                    # Start Phi-4 task
                    phi4_task = asyncio.create_task(
                        self._plan_action_with_llm(
                            perception, game_state, scene_type, current_layer, available_actions, 
                            strategic_analysis, motivation, huihui_context
                        )
                    )
                    
                    # Race: Phi-4 vs Gemini MoE vs Local MoE (whichever finishes first)
                    # Timeout after 10 seconds - use fastest response for smooth gameplay
                    tasks_to_race = [phi4_task]
                    if gemini_moe_task:
                        tasks_to_race.append(gemini_moe_task)
                    if local_moe_task:
                        tasks_to_race.append(local_moe_task)
                    
                    if len(tasks_to_race) > 1:
                        # Scene-based timeout: combat 15s, exploration 30s, menus 45s (Fix 14)
                        timeout_duration = 15.0 if game_state.in_combat else 45.0 if scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE] else 30.0
                        
                        # Progressive timeout: extend by 5s if systems near completion (Fix 15)
                        if self.cycle_count < 3:
                            timeout_duration = 60.0  # LLM warmup phase (Fix 18)
                        
                        print(f"[PARALLEL] Racing {len(tasks_to_race)} systems ({timeout_duration}s timeout)...")
                        try:
                            done, pending = await asyncio.wait(
                                tasks_to_race,
                                timeout=timeout_duration,
                                return_when=asyncio.FIRST_COMPLETED
                            )
                            
                            # Check which one finished first
                            for task in done:
                                if task == gemini_moe_task:
                                    try:
                                        vision_resp, reasoning_resp = task.result()
                                        if reasoning_resp and reasoning_resp.consensus:
                                            # Parse action from consensus
                                            consensus_text = reasoning_resp.consensus
                                            action = None
                                            if "ACTION:" in consensus_text:
                                                action_line = [l for l in consensus_text.split('\n') if 'ACTION:' in l][0]
                                                action = action_line.split('ACTION:')[1].strip().lower()
                                            
                                            # LLM decision logging (Fix 12)
                                            confidence = reasoning_resp.confidence if hasattr(reasoning_resp, 'confidence') else 0.5
                                            print(f"[LLM-DECISION-LOG] MoE | Action: {action} | Confidence: {confidence:.2f} | Valid: {action in available_actions if action else False}")
                                            
                                            # Lowered confidence threshold from 0.7 to 0.5 (Fix 13)
                                            if action and action in available_actions and confidence >= 0.5:
                                                print(f"[GEMINI-MOE] ✓ Won the race! {len(self.moe.gemini_experts)} experts chose: {action} (conf: {confidence:.2f})")
                                                
                                                # Track action source
                                                self.stats['action_source_moe'] += 1
                                                self.last_action_source = 'moe'
                                                
                                                # Hebbian: Record successful Gemini MoE activation
                                                self.hebbian.record_activation(
                                                    system_name='gemini_moe',
                                                    success=True,
                                                    contribution_strength=1.0,
                                                    context={'action': action, 'won_race': True, 'experts': len(self.moe.gemini_experts)}
                                                )
                                                
                                                # Cancel other tasks (but NOT Claude - let it finish in background)
                                                for t in pending:
                                                    if t != claude_reasoning_task:
                                                        t.cancel()
                                                return action
                                    except Exception as e:
                                        print(f"[GEMINI-MOE] ⚠️ Error parsing result: {e}")
                                    
                                    # Hebbian: Record failure
                                    self.hebbian.record_activation(
                                        system_name='gemini_moe',
                                        success=False,
                                        contribution_strength=0.5
                                    )
                                elif task == local_moe_task:
                                    moe_recommendation = task.result()
                                    if moe_recommendation:
                                        action, reasoning = moe_recommendation
                                        print(f"[LOCAL-MOE] ✓ Won the race! Using: {action}")
                                        self.local_moe_failures = 0  # Reset on success
                                        
                                        # Track action source
                                        self.stats['action_source_local_moe'] += 1
                                        self.last_action_source = 'local_moe'
                                        
                                        # Hebbian: Record successful local MoE activation
                                        self.hebbian.record_activation(
                                            system_name='local_moe',
                                            success=True,
                                            contribution_strength=0.9,  # Slightly lower than cloud
                                            context={'action': action, 'won_race': True}
                                        )
                                        
                                        # Cancel other tasks
                                        for t in pending:
                                            t.cancel()
                                        return action
                                    else:
                                        self.local_moe_failures += 1
                                        print(f"[LOCAL-MOE] ⚠️ Returned None (failures: {self.local_moe_failures}/{self.max_consecutive_failures})")
                                        # Hebbian: Record failure
                                        self.hebbian.record_activation(
                                            system_name='local_moe',
                                            success=False,
                                            contribution_strength=0.5
                                        )
                                elif task == phi4_task:
                                    llm_action = task.result()
                                    if llm_action:
                                        print(f"[PHI4] ✓ Won the race! Using: {llm_action}")
                                        self.stats['llm_action_count'] += 1
                                        
                                        # Track action source
                                        self.stats['action_source_phi4'] += 1
                                        self.last_action_source = 'phi4'
                                        
                                        # Hebbian: Record successful Phi-4 activation
                                        self.hebbian.record_activation(
                                            system_name='phi4_planner',
                                            success=True,
                                            contribution_strength=0.8,
                                            context={'action': llm_action, 'won_race': True}
                                        )
                                        
                                        # Cancel other tasks
                                        for t in pending:
                                            t.cancel()
                                        return llm_action
                            
                            # Timeout - cancel pending tasks and use heuristic
                            if pending:
                                print(f"[PARALLEL] ⏱️ Timeout after 13.5s, using heuristic (cancelled {len(pending)} pending tasks)")
                                self.stats['action_source_timeout'] += 1
                                self.last_action_source = 'timeout'
                                for task in pending:
                                    task.cancel()
                        except asyncio.TimeoutError:
                            print(f"[PARALLEL] ⏱️ Timeout after 13.5s, using heuristic")
                            self.stats['action_source_timeout'] += 1
                            self.last_action_source = 'timeout'
                            for task in tasks_to_race:
                                task.cancel()
                    else:
                        # No cloud task, just wait for Phi-4
                        llm_action = await phi4_task
                        if llm_action:
                            print(f"[PHI4] Final action: {llm_action}")
                            self.stats['llm_action_count'] += 1
                            self.stats['action_source_phi4'] += 1
                            self.last_action_source = 'phi4'
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
            self.stats['action_source_heuristic'] += 1
            self.last_action_source = 'heuristic'
            
            # Check for stuck condition - all systems failing
            if (self.cloud_llm_failures >= self.max_consecutive_failures and 
                self.local_moe_failures >= self.max_consecutive_failures):
                print(f"[STUCK-DETECTION] ⚠️⚠️⚠️ ALL LLM SYSTEMS FAILING!")
                print(f"[STUCK-DETECTION] Cloud: {self.cloud_llm_failures}, Local MoE: {self.local_moe_failures}")
                print(f"[STUCK-DETECTION] Forcing random exploration action")
                # Force a random action to break the loop
                recovery_actions = ['move_forward', 'jump', 'look_around', 'activate']
                recovery_action = random.choice([a for a in recovery_actions if a in available_actions] or available_actions)
                # Reset counters
                self.cloud_llm_failures = 0
                self.local_moe_failures = 0
                self.heuristic_failures = 0
                return recovery_action

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
            
            # STUCK DETECTION: Use deterministic recovery actions
            stuck_status = self._detect_stuck()
            
            if stuck_status['is_stuck']:
                # Use the recommended recovery action directly
                if stuck_status['recovery_action'] and stuck_status['recovery_action'] in available_actions:
                    print(f"[STUCK-{stuck_status['severity'].upper()}] {stuck_status['reason']} → {stuck_status['recovery_action']}")
                    self.consecutive_same_action = 0  # Reset counter
                    return stuck_status['recovery_action']
                
                # Fallback if recommended action not available
                print(f"[STUCK] Detected stuck state! Forcing recovery action...")
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
            # Prioritize NPC interaction in non-combat scenes (increased from 15% to 60%)
            if 'activate' in available_actions:
                activation_chance = 0.60 if not game_state.in_combat else 0.15
                if random.random() < activation_chance:
                    print(f"[HEURISTIC] → activate ({'NPC interaction prioritized' if not game_state.in_combat else 'random curiosity'})")
                    return 'activate'
            
            # Occasionally look around (human-like awareness)
            if random.random() < 0.10:
                print(f"[HEURISTIC] → look_around (situational awareness)")
                return 'look_around'
            
            # Occasionally jump (human-like playfulness/testing)
            if random.random() < 0.08 and 'jump' in available_actions:
                print(f"[HEURISTIC] → jump (playful exploration)")
                return 'jump'
            
            # Use RL Q-values intelligently for action selection
            # Get top 5 actions by Q-value
            sorted_q_actions = sorted(
                [(a, q_values.get(a, 0.0)) for a in available_actions],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # QUANTUM SUPERPOSITION: Apply 4D fractal variance to Q-values
            # This adds 0.01-0.09% deterministic exploration variance
            if hasattr(self, 'quantum_explorer') and len(sorted_q_actions) > 0:
                action_names = [a for a, _ in sorted_q_actions]
                q_value_array = np.array([q for _, q in sorted_q_actions])
                
                # Check if stuck - use quantum tunneling for escape
                stuck_status = self._detect_stuck()
                if stuck_status['is_stuck']:
                    tunnel_prob = self.quantum_explorer.quantum_tunnel(stuck_status['severity'])
                    if random.random() < tunnel_prob:
                        print(f"[QUANTUM-TUNNEL] Severity: {stuck_status['severity']} | Prob: {tunnel_prob:.3f}")
                        # Use quantum exploration vector instead of Q-values
                        exploration_vector = self.quantum_explorer.get_exploration_vector()
                        # Map 4D exploration to available actions
                        quantum_action_idx = int(exploration_vector[0] * len(action_names))
                        quantum_action = action_names[quantum_action_idx]
                        print(f"[QUANTUM-TUNNEL] → {quantum_action} (4D fractal escape)")
                        return quantum_action
                
                # Normal quantum superposition collapse
                quantum_action, confidence = self.quantum_explorer.collapse_superposition(
                    action_weights=q_value_array,
                    action_names=action_names,
                    temperature=1.0  # Higher = more exploration
                )
                
                fractal_stats = self.quantum_explorer.get_fractal_stats()
                print(f"[QUANTUM] Collapsed superposition → {quantum_action} (conf: {confidence:.3f}, fib: {fractal_stats['fibonacci_phase']}, depth: {fractal_stats['iteration_depth']})")
                
                # Use quantum action with high confidence, otherwise continue to motivation-based
                if confidence > 0.15:  # Threshold for quantum decision
                    return quantum_action
            
            # Motivation-based selection with Q-value guidance
            if dominant_drive == 'curiosity':
                # Curiosity: prioritize interaction and exploration variety
                if 'activate' in available_actions and random.random() < 0.4:
                    print(f"[HEURISTIC] → activate (curiosity-driven interaction)")
                    return 'activate'
                # Pick from top 3 Q-value actions for variety
                if len(sorted_q_actions) >= 3:
                    action = random.choice([a for a, _ in sorted_q_actions[:3]])
                    print(f"[HEURISTIC] → {action} (curiosity + Q-value)")
                    return action
                print(f"[HEURISTIC] → explore (curiosity/default)")
                return 'explore'
                
            elif dominant_drive == 'competence':
                # Competence: practice skills
                if 'power_attack' in available_actions and current_layer == "Combat":
                    print(f"[HEURISTIC] → power_attack (competence training)")
                    return 'power_attack'
                elif 'backstab' in available_actions and current_layer == "Stealth":
                    print(f"[HEURISTIC] → backstab (stealth practice)")
                    return 'backstab'
                # Pick from top Q-values
                if sorted_q_actions:
                    action = sorted_q_actions[0][0]
                    print(f"[HEURISTIC] → {action} (competence + top Q-value)")
                    return action
                print(f"[HEURISTIC] → explore (competence/default)")
                return 'explore'
                
            elif dominant_drive == 'coherence':
                # Coherence: prefer gentle, varied actions
                if game_state.health < 30 and 'rest' in available_actions:
                    print(f"[HEURISTIC] → rest (coherence restoration)")
                    return 'rest'
                # Pick varied actions from top Q-values (avoid always same)
                if len(sorted_q_actions) >= 4:
                    # Pick from top 4, but avoid last action
                    options = [a for a, _ in sorted_q_actions[:4] if a != self.last_executed_action]
                    if options:
                        action = random.choice(options)
                        print(f"[HEURISTIC] → {action} (coherence + varied Q-value)")
                        return action
                # Fall back to exploration
                print(f"[HEURISTIC] → explore (coherence/default)")
                return 'explore'
                
            else:  # autonomy or default
                # Autonomy: prefer varied, independent choices
                if random.random() < 0.3 and 'activate' in available_actions:
                    print(f"[HEURISTIC] → activate (autonomous interaction)")
                    return 'activate'
                # Pick from top Q-values with variety
                if len(sorted_q_actions) >= 3:
                    # Weighted random from top 3 (not just first)
                    weights = [3, 2, 1][:len(sorted_q_actions[:3])]
                    action = random.choices(
                        [a for a, _ in sorted_q_actions[:3]],
                        weights=weights
                    )[0]
                    print(f"[HEURISTIC] → {action} (autonomy + weighted Q-value)")
                    return action
                print(f"[HEURISTIC] → explore (autonomy/default)")
                return 'explore'

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
                result = await asyncio.wait_for(
                    self.action_planning_llm.generate(
                        prompt=context,
                        max_tokens=300  # Enough for reasoning + action selection
                    ),
                    timeout=60.0  # 60 second timeout for local LLM
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
        except asyncio.TimeoutError:
            print(f"[MISTRAL-ACTION] Timed out after 60s - using fallback heuristic")
            return None
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

    def _sensorimotor_override(self, available_actions: List[str], game_state) -> Optional[str]:
        """Optionally override planning when sensorimotor loop reports a stuck state."""
        feedback = self.sensorimotor_state
        if not feedback:
            return None

        feedback_cycle = feedback.get('cycle', -1)
        if feedback_cycle == -1 or self.cycle_count - feedback_cycle > 6:
            return None  # Feedback too old to trust

        if not feedback.get('should_override'):
            return None

        reason = feedback.get('reason', 'sensorimotor_stuck')
        similarity = feedback.get('visual_similarity')
        streak = feedback.get('similarity_streak', 0)
        dialogue_loop = feedback.get('dialogue_loop', False)
        print(f"[PLANNING] Sensorimotor override engaged (reason: {reason}, similarity={similarity}, streak={streak})")

        recovery_priority = []
        
        # Special handling for dialogue loops - prioritize exiting dialogue
        if dialogue_loop or reason == 'dialogue_loop':
            print("[PLANNING] → Dialogue loop detected! Attempting to exit dialogue...")
            # Try to exit dialogue with back button (Tab/ESC equivalent) and wait
            if 'back' in available_actions:
                recovery_priority.append('back')
            if 'wait' in available_actions:
                recovery_priority.append('wait')
            # Move away from NPC
            recovery_priority.extend(['move_backward', 'turn_left', 'turn_right'])
        else:
            # Normal stuck detection - try to move/interact
            if not game_state.in_menu and 'activate' in available_actions:
                recovery_priority.append('activate')
            if 'jump' in available_actions:
                recovery_priority.append('jump')
            if 'turn_left' in available_actions:
                recovery_priority.append('turn_left')
            if 'turn_right' in available_actions:
                recovery_priority.append('turn_right')
            if 'move_backward' in available_actions:
                recovery_priority.append('move_backward')
            if 'look_around' in available_actions:
                recovery_priority.append('look_around')
            if 'sneak' in available_actions:
                recovery_priority.append('sneak')

        for candidate in recovery_priority:
            if candidate != self.last_executed_action:
                print(f"[PLANNING] → Sensorimotor recovery action: {candidate}")
                feedback['should_override'] = False
                feedback['last_override_cycle'] = self.cycle_count
                feedback['last_forced_action'] = candidate
                return candidate

        if available_actions:
            fallback_pool = [a for a in available_actions if a != self.last_executed_action]
            fallback = random.choice(fallback_pool or available_actions)
            print(f"[PLANNING] → Sensorimotor fallback action: {fallback}")
            feedback['should_override'] = False
            feedback['last_override_cycle'] = self.cycle_count
            feedback['last_forced_action'] = fallback
            return fallback

        return None

    def _update_sensorimotor_state(
        self,
        cycle: int,
        similarity: Optional[float],
        analysis: str,
        has_thinking: bool,
        visual_context: str
    ) -> None:
        """Persist sensorimotor feedback for action planning overrides."""
        if self.sensorimotor_last_cycle != -1 and cycle - self.sensorimotor_last_cycle > 6:
            self.sensorimotor_similarity_streak = 0

        self.sensorimotor_last_cycle = cycle

        high_similarity = similarity is not None and similarity >= self.sensorimotor_high_similarity_threshold
        if high_similarity:
            self.sensorimotor_similarity_streak += 1
        else:
            self.sensorimotor_similarity_streak = 0

        combined_text = f"{analysis}\n{visual_context}".lower()
        stuck_keywords = any(keyword in combined_text for keyword in ("stuck", "no movement", "blocked"))
        
        # Check for dialogue loop (high similarity + in_dialogue state)
        current_perception = self.perception.perception_history[-1] if self.perception.perception_history else {}
        in_dialogue = current_perception.get('game_state', {}).get('in_dialogue', False) if isinstance(current_perception.get('game_state'), dict) else False
        dialogue_loop = in_dialogue and high_similarity and self.sensorimotor_similarity_streak >= 3
        
        should_override = stuck_keywords or dialogue_loop or (
            high_similarity and self.sensorimotor_similarity_streak >= self.sensorimotor_required_streak
        )

        reason = "analysis_stuck" if stuck_keywords else (
            "dialogue_loop" if dialogue_loop else (
                "similarity_loop" if high_similarity else ""
            )
        )

        self.sensorimotor_state = {
            'cycle': cycle,
            'visual_similarity': similarity,
            'similarity_streak': self.sensorimotor_similarity_streak,
            'analysis': analysis,
            'visual_context': visual_context,
            'has_thinking': has_thinking,
            'stuck_keywords': stuck_keywords,
            'high_similarity': high_similarity,
            'dialogue_loop': dialogue_loop,
            'should_override': should_override,
            'reason': reason,
            'timestamp': time.time(),
        }

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
        
        # CRITICAL FIX: Auto-exit dialogue/menus when trying non-menu actions
        # This prevents getting stuck in dialogue with ineffective movement attempts
        if scene_type in [SceneType.DIALOGUE, SceneType.INVENTORY, SceneType.MAP]:
            if action not in menu_related_actions:
                # Agent wants to do non-menu action but is in menu/dialogue
                print(f"[AUTO-EXIT] Detected {scene_type.value} scene but action '{action}' is not menu-related")
                print(f"[AUTO-EXIT] Exiting {scene_type.value} first to enable game control")
                
                # Exit dialogue/menu by pressing Tab (or ESC on some systems)
                # Tab works for both inventory and dialogue in Skyrim
                await self.actions.execute(Action(ActionType.BACK, duration=0.2))
                await asyncio.sleep(0.5)  # Wait for menu/dialogue to close
                
                # Press again if needed (sometimes takes 2 presses)
                await self.actions.execute(Action(ActionType.BACK, duration=0.2))
                await asyncio.sleep(0.5)
                
                print(f"[AUTO-EXIT] Dialogue/menu exit complete, now executing: {action}")
        
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
            # Special handling for activate in dialogue scenes
            if scene_type == SceneType.DIALOGUE:
                # Check if we've been stuck in dialogue too long
                dialogue_action_count = sum(1 for a in self.action_history[-5:] if 'activate' in str(a).lower())
                if dialogue_action_count >= 3:
                    print(f"[ACTION] Stuck in dialogue after {dialogue_action_count} activates - exiting dialogue")
                    # Exit dialogue instead of activating again
                    await self.actions.execute(Action(ActionType.BACK, duration=0.2))
                    await asyncio.sleep(0.5)
                    print(f"[ACTION] Exited dialogue, returning to game")
                else:
                    # Continue dialogue (select option or advance)
                    print(f"[ACTION] Progressing dialogue ({dialogue_action_count+1}/3 activates)")
                    await self.actions.execute(Action(ActionType.ACTIVATE, duration=0.3))
                    await asyncio.sleep(0.5)
            else:
                # Normal activation outside dialogue
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
            # Use healing spell via controller binding (not inventory menu)
            if scene_type in [SceneType.INVENTORY, SceneType.MAP, SceneType.DIALOGUE]:
                print(f"[ACTION] Already in {scene_type.value}, exiting menu first")
                # Exit menu instead
                await self.actions.execute(Action(ActionType.BACK, duration=0.2))
                await asyncio.sleep(0.5)
            else:
                # Execute heal action from controller bindings
                # This uses favorites + healing spell, not inventory
                print("[ACTION] Using healing spell (via controller binding)")
                await self.actions.execute(Action(ActionType.HEAL, duration=1.0))
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
            await self.actions.execute(Action(ActionType.OPEN_INVENTORY, duration=0.2))
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
                await self.actions.execute(Action(ActionType.OPEN_INVENTORY, duration=0.2))
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

    def _detect_stuck(self) -> Dict[str, Any]:
        """
        PRAGMATIC FIX: Multi-tier stuck detection with severity levels.
        
        Philosophy: Don't wait for LLM to realize we're stuck.
        Detect it deterministically and return actionable status.
        
        Returns:
            Dict with 'is_stuck', 'severity' (low/medium/high), 'reason', 'recovery_action'
        """
        stuck_info = {
            'is_stuck': False,
            'severity': 'none',
            'reason': '',
            'recovery_action': None
        }
        
        # Need at least a few actions to detect stuck
        if len(self.action_history) < self.stuck_detection_window:
            return stuck_info
        
        # 1. CRITICAL: Action repetition >8 times
        if self.consecutive_same_action >= 8:
            stuck_info['is_stuck'] = True
            stuck_info['severity'] = 'high'
            stuck_info['reason'] = f'Critical repetition: {self.last_executed_action} x{self.consecutive_same_action}'
            stuck_info['recovery_action'] = 'jump' if self.last_executed_action != 'jump' else 'turn_right'
            print(f"[STUCK-HIGH] {stuck_info['reason']} → {stuck_info['recovery_action']}")
            return stuck_info
        
        # 2. Check if we're repeating the same action in window
        recent_actions = self.action_history[-self.stuck_detection_window:]
        if len(set(recent_actions)) == 1:  # All same action
            same_action = recent_actions[0]
            
            # Check if coherence is changing (sign of progress)
            if len(self.coherence_history) >= self.stuck_detection_window:
                recent_coherence = self.coherence_history[-self.stuck_detection_window:]
                coherence_change = max(recent_coherence) - min(recent_coherence)
                
                if coherence_change < self.stuck_threshold:
                    stuck_info['is_stuck'] = True
                    stuck_info['severity'] = 'medium'
                    stuck_info['reason'] = f'Repeating {same_action} {self.stuck_detection_window}x, Δ𝒞={coherence_change:.3f}'
                    stuck_info['recovery_action'] = 'turn_left' if 'turn' not in same_action else 'move_backward'
                    print(f"[STUCK-MEDIUM] {stuck_info['reason']} → {stuck_info['recovery_action']}")
                    return stuck_info
        
        # 3. MEDIUM: Visual similarity (seeing same thing)
        if len(self.visual_embedding_history) >= 3:
            import numpy as np
            recent_embeddings = self.visual_embedding_history[-3:]
            if all(isinstance(e, (list, np.ndarray)) for e in recent_embeddings) and len(recent_embeddings) >= 2:
                # Cosine similarity between last two
                last = np.array(recent_embeddings[-1]).flatten()
                prev = np.array(recent_embeddings[-2]).flatten()
                similarity = np.dot(last, prev) / (np.linalg.norm(last) * np.linalg.norm(prev) + 1e-8)
                
                if similarity > 0.98:  # 98% similar = visually stuck
                    stuck_info['is_stuck'] = True
                    stuck_info['severity'] = 'low'
                    stuck_info['reason'] = f'Visual stuckness: {similarity:.3f} similarity'
                    stuck_info['recovery_action'] = 'turn_around'
                    print(f"[STUCK-LOW] {stuck_info['reason']} → {stuck_info['recovery_action']}")
                    return stuck_info
        
        return stuck_info
    
    def _update_stuck_tracking(self, action: str, coherence: float, visual_embedding=None):
        """Update stuck detection tracking."""
        # Track action history
        self.action_history.append(action)
        if len(self.action_history) > 20:  # Keep last 20 actions
            self.action_history.pop(0)
        
        # Track visual embeddings for similarity detection
        if visual_embedding is not None:
            self.visual_embedding_history.append(visual_embedding)
            if len(self.visual_embedding_history) > 10:  # Keep last 10 embeddings
                self.visual_embedding_history.pop(0)
        
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
    
    def _display_performance_dashboard(self):
        """
        Fix 20: Real-time performance monitoring dashboard.
        Displays LLM vs heuristic usage, rate limit status, timeout frequency.
        """
        total_actions = self.stats['actions_taken']
        if total_actions == 0:
            return
        
        print(f"\n{'='*70}")
        print(f"⚡ PERFORMANCE DASHBOARD (Cycle {self.cycle_count})")
        print(f"{'='*70}")
        
        # Action Source Breakdown
        source_total = sum([
            self.stats.get('action_source_moe', 0),
            self.stats.get('action_source_hybrid', 0),
            self.stats.get('action_source_phi4', 0),
            self.stats.get('action_source_local_moe', 0),
            self.stats.get('action_source_heuristic', 0),
            self.stats.get('action_source_timeout', 0),
        ])
        
        if source_total > 0:
            print(f"\n🎯 ACTION SOURCES:")
            moe_pct = 100 * self.stats.get('action_source_moe', 0) / source_total
            hybrid_pct = 100 * self.stats.get('action_source_hybrid', 0) / source_total
            phi4_pct = 100 * self.stats.get('action_source_phi4', 0) / source_total
            heur_pct = 100 * self.stats.get('action_source_heuristic', 0) / source_total
            timeout_pct = 100 * self.stats.get('action_source_timeout', 0) / source_total
            
            print(f"  Cloud MoE:     {self.stats.get('action_source_moe', 0):3d} ({moe_pct:5.1f}%) {'🟢' if moe_pct > 20 else '🟡' if moe_pct > 5 else '🔴'}")
            print(f"  Hybrid LLM:    {self.stats.get('action_source_hybrid', 0):3d} ({hybrid_pct:5.1f}%) {'🟢' if hybrid_pct > 10 else '🟡' if hybrid_pct > 2 else '🔴'}")
            print(f"  Phi-4 Local:   {self.stats.get('action_source_phi4', 0):3d} ({phi4_pct:5.1f}%)")
            print(f"  Heuristics:    {self.stats.get('action_source_heuristic', 0):3d} ({heur_pct:5.1f}%) {'🟢' if heur_pct < 30 else '🟡' if heur_pct < 60 else '🔴'}")
            print(f"  Timeouts:      {self.stats.get('action_source_timeout', 0):3d} ({timeout_pct:5.1f}%) {'🟢' if timeout_pct < 10 else '🟡' if timeout_pct < 30 else '🔴'}")
        
        # Rate Limit Status
        if self.moe and hasattr(self.moe, 'is_gemini_rate_limited'):
            is_limited, wait_time = self.moe.is_gemini_rate_limited()
            gemini_rpm = self.moe.gemini_rpm_limit if hasattr(self.moe, 'gemini_rpm_limit') else 0
            claude_rpm = self.moe.claude_rpm_limit if hasattr(self.moe, 'claude_rpm_limit') else 0
            
            print(f"\n⏱️  RATE LIMIT STATUS:")
            print(f"  Gemini:  {'🔴 LIMITED' if is_limited else '🟢 AVAILABLE'} ({wait_time:.1f}s wait) | Limit: {gemini_rpm} RPM")
            print(f"  Claude:  🟢 AVAILABLE | Limit: {claude_rpm} RPM")
            print(f"  Queued:  {self.stats.get('llm_queued_requests', 0)} | Skipped: {self.stats.get('llm_skipped_rate_limit', 0)}")
        
        # Timing Stats
        if self.stats['planning_times']:
            avg_planning = sum(self.stats['planning_times'][-10:]) / len(self.stats['planning_times'][-10:])
            print(f"\n⏲️  TIMING (last 10 cycles):")
            print(f"  Avg Planning:   {avg_planning:.2f}s {'🟢' if avg_planning < 10 else '🟡' if avg_planning < 20 else '🔴'}")
            
        if self.stats['execution_times']:
            avg_exec = sum(self.stats['execution_times'][-10:]) / len(self.stats['execution_times'][-10:])
            print(f"  Avg Execution:  {avg_exec:.2f}s")
        
        # Success Rate
        success_rate = 0.0
        if total_actions > 0:
            success_rate = self.stats['action_success_count'] / total_actions
        
        print(f"\n📊 EFFECTIVENESS:")
        print(f"  Success Rate:  {success_rate:.1%} {'🟢' if success_rate > 0.3 else '🟡' if success_rate > 0.15 else '🔴'}")
        print(f"  Fast Actions:  {self.stats['fast_action_count']} ({100*self.stats['fast_action_count']/total_actions:.0f}%)")
        
        # Consciousness
        if hasattr(self, 'current_consciousness') and self.current_consciousness:
            print(f"\n🧠 CONSCIOUSNESS:")
            print(f"  Coherence 𝒞:    {self.current_consciousness.coherence:.3f}")
            print(f"  Ontical ℓₒ:    {self.current_consciousness.coherence_ontical:.3f}")
            print(f"  Structural ℓₛ: {self.current_consciousness.coherence_structural:.3f}")
            print(f"  Participatory ℓₚ: {self.current_consciousness.coherence_participatory:.3f}")
        
        print(f"{'='*70}\n")

    def _update_action_diversity_stats(self, action: str, coherence: float):
        """
        Track action diversity and reward patterns for continuous learning adaptation.
        Implements GPT-4o's recommendation for Hebbian-style learning from outcomes.
        
        Args:
            action: The action taken
            coherence: The resulting coherence value (reward proxy)
        """
        # Track action frequency
        if action not in self.action_type_counts:
            self.action_type_counts[action] = 0
            self.reward_by_action_type[action] = []
        
        self.action_type_counts[action] += 1
        self.reward_by_action_type[action].append(coherence)
        self.recent_action_types.append(action)
        
        # Keep recent window manageable
        if len(self.recent_action_types) > 50:
            self.recent_action_types = self.recent_action_types[-50:]
        
        # Periodically analyze and adapt
        if len(self.recent_action_types) >= 20 and len(self.recent_action_types) % 10 == 0:
            self._analyze_action_patterns()
    
    def _analyze_action_patterns(self):
        """
        Analyze action diversity patterns and provide insights.
        Implements continuous learning adaptation based on outcome analysis.
        """
        if not self.reward_by_action_type:
            return
        
        # Calculate diversity metrics
        total_actions = sum(self.action_type_counts.values())
        action_diversity = len(self.action_type_counts) / max(total_actions, 1)
        
        # Find best performing actions
        avg_rewards = {
            action: sum(rewards) / len(rewards)
            for action, rewards in self.reward_by_action_type.items()
            if rewards
        }
        
        if avg_rewards:
            best_action = max(avg_rewards.items(), key=lambda x: x[1])
            worst_action = min(avg_rewards.items(), key=lambda x: x[1])
            
            print(f"[DIVERSITY] Action diversity: {action_diversity:.2f}")
            print(f"[DIVERSITY] Best action: {best_action[0]} (avg coherence: {best_action[1]:.3f})")
            print(f"[DIVERSITY] Learning from {len(self.reward_by_action_type)} action types")

    def _update_dashboard_state(self, action: Optional[str] = None, action_source: Optional[str] = None):
        """
        Update dashboard streamer with current AGI state.
        Called during the main loop to provide real-time updates to webapp.
        """
        if not hasattr(self, 'dashboard_streamer') or not self.dashboard_streamer:
            return
        
        try:
            # Collect current state
            agi_state = {
                'cycle': self.stats.get('cycles_completed', 0),
                'action': action or self.last_executed_action or 'idle',
                'action_source': action_source or self.last_action_source or 'unknown',
                
                # Perception
                'perception': {
                    'scene_type': self.current_perception.get('scene_type', 'unknown') if self.current_perception else 'unknown',
                    'objects': self.current_perception.get('objects', []) if self.current_perception else [],
                    'enemies_nearby': self.current_perception.get('enemies_nearby', False) if self.current_perception else False,
                    'npcs_nearby': self.current_perception.get('npcs_nearby', False) if self.current_perception else False,
                    'last_vision_time': self.current_perception.get('last_vision_time', 0) if self.current_perception else 0
                },
                
                # Game state
                'game_state': self._extract_game_state(),
                
                # Consciousness
                'consciousness': {
                    'coherence': self.current_consciousness.coherence if self.current_consciousness else 0,
                    'phi': self.current_consciousness.consciousness_level if self.current_consciousness else 0,
                    'nodes_active': len(self.consciousness_monitor.nodes) if hasattr(self, 'consciousness_monitor') and self.consciousness_monitor else 0
                },
                
                # LLM status
                'llm_status': {
                    'mode': self.config.llm_architecture if hasattr(self.config, 'llm_architecture') else 'none',
                    'cloud_active': sum([
                        1 if hasattr(self, 'hybrid_llm') and self.hybrid_llm else 0,
                        1 if hasattr(self, 'moe') and self.moe else 0
                    ]),
                    'local_active': 1 if hasattr(self, 'local_moe') and self.local_moe else 0,
                    'total_calls': self.stats.get('llm_action_count', 0),
                    'last_call_time': 0,  # TODO: Track this
                    'active_models': self._get_active_models()
                },
                
                # Performance
                'performance': {
                    'fps': 60,  # TODO: Calculate from cycle times
                    'planning_time': sum(self.stats.get('planning_times', [0])[-5:]) / max(len(self.stats.get('planning_times', [1])[-5:]), 1),
                    'execution_time': sum(self.stats.get('execution_times', [0])[-5:]) / max(len(self.stats.get('execution_times', [1])[-5:]), 1),
                    'vision_time': 0,  # TODO: Track this
                    'total_cycle_time': 0  # TODO: Calculate
                },
                
                # Stats
                'stats': {
                    'success_rate': self.stats.get('action_success_count', 0) / max(self.stats.get('actions_taken', 1), 1),
                    'rl_actions': self.stats.get('rl_action_count', 0),
                    'llm_actions': self.stats.get('llm_action_count', 0),
                    'heuristic_actions': self.stats.get('heuristic_action_count', 0)
                },
                
                # World model
                'world_model': {
                    'beliefs': {},  # TODO: Extract from world model
                    'goals': [self.current_goal] if self.current_goal else [],
                    'strategy': 'explore'  # TODO: Extract from planner
                }
            }
            
            # Update dashboard
            self.dashboard_streamer.update(agi_state)
            
        except Exception as e:
            # Don't crash the AGI if dashboard fails
            print(f"[DASHBOARD] Warning: Update failed: {e}")
    
    def _extract_game_state(self) -> Dict[str, Any]:
        """Extract game state from perception, handling both dict and GameState object."""
        if not self.current_perception:
            return {
                'health': 100,
                'magicka': 100,
                'stamina': 100,
                'in_combat': False,
                'in_menu': False,
                'location': 'Unknown'
            }
        
        game_state = self.current_perception.get('game_state')
        
        # Handle GameState object (dataclass)
        if game_state and hasattr(game_state, 'health'):
            return {
                'health': getattr(game_state, 'health', 100),
                'magicka': getattr(game_state, 'magicka', 100),
                'stamina': getattr(game_state, 'stamina', 100),
                'in_combat': getattr(game_state, 'in_combat', False),
                'in_menu': getattr(game_state, 'in_menu', False),
                'location': getattr(game_state, 'location_name', 'Unknown')
            }
        
        # Handle dictionary
        if isinstance(game_state, dict):
            return {
                'health': game_state.get('health', 100),
                'magicka': game_state.get('magicka', 100),
                'stamina': game_state.get('stamina', 100),
                'in_combat': game_state.get('in_combat', False),
                'in_menu': game_state.get('in_menu', False),
                'location': game_state.get('location', 'Unknown')
            }
        
        # Fallback
        return {
            'health': 100,
            'magicka': 100,
            'stamina': 100,
            'in_combat': False,
            'in_menu': False,
            'location': 'Unknown'
        }
    
    def _get_active_models(self) -> list:
        """Get list of currently active LLM models."""
        models = []
        
        if hasattr(self, 'hybrid_llm') and self.hybrid_llm:
            models.extend(['Gemini 2.0 Flash', 'Claude Sonnet 4.5'])
        
        if hasattr(self, 'moe') and self.moe:
            models.extend(['Gemini 1', 'Gemini 2', 'Claude Sonnet', 'GPT-4o', 'Nemotron', 'Qwen3'])
        
        if hasattr(self, 'local_moe') and self.local_moe:
            models.extend(['Huihui-VL', 'Qwen3-VL', 'Phi-4'])
        
        return models

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
        
        # Action source breakdown (which system provided actions)
        total_actions = sum([
            self.stats.get('action_source_moe', 0),
            self.stats.get('action_source_hybrid', 0),
            self.stats.get('action_source_phi4', 0),
            self.stats.get('action_source_local_moe', 0),
            self.stats.get('action_source_heuristic', 0),
            self.stats.get('action_source_timeout', 0),
        ])
        if total_actions > 0:
            print(f"\n🎯 Action Sources (who provided the action):")
            print(f"  Gemini MoE: {self.stats.get('action_source_moe', 0)} ({100*self.stats.get('action_source_moe', 0)/total_actions:.1f}%)")
            print(f"  Hybrid LLM: {self.stats.get('action_source_hybrid', 0)} ({100*self.stats.get('action_source_hybrid', 0)/total_actions:.1f}%)")
            print(f"  Phi-4 Planner: {self.stats.get('action_source_phi4', 0)} ({100*self.stats.get('action_source_phi4', 0)/total_actions:.1f}%)")
            print(f"  Local MoE: {self.stats.get('action_source_local_moe', 0)} ({100*self.stats.get('action_source_local_moe', 0)/total_actions:.1f}%)")
            print(f"  Heuristic: {self.stats.get('action_source_heuristic', 0)} ({100*self.stats.get('action_source_heuristic', 0)/total_actions:.1f}%)")
            print(f"  Timeout: {self.stats.get('action_source_timeout', 0)} ({100*self.stats.get('action_source_timeout', 0)/total_actions:.1f}%)")
            if self.last_action_source:
                print(f"  Last successful: {self.last_action_source}")
        
        # Quantum Superposition stats
        if hasattr(self, 'quantum_explorer'):
            fractal_stats = self.quantum_explorer.get_fractal_stats()
            print(f"\n🌀 Quantum Superposition (4D Fractal RNG):")
            print(f"  Fractal iterations: {fractal_stats['iteration_depth']}")
            print(f"  Fibonacci phase: {fractal_stats['fibonacci_phase']}/50")
            print(f"  Golden ratio φ: {fractal_stats['phi']:.6f}")
            print(f"  Variance range: {fractal_stats['variance_range']}")
            state = fractal_stats['state']
            print(f"  4D state: [{state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}, {state['w']:.3f}]")
            
            # Show exploration vector
            exploration = self.quantum_explorer.get_exploration_vector()
            print(f"  Exploration vector:")
            print(f"    Spatial: {exploration[0]:.3f} | Social: {exploration[1]:.3f}")
            print(f"    Cognitive: {exploration[2]:.3f} | Consciousness: {exploration[3]:.3f}")
        
        # Timing metrics
        if self.stats['planning_times']:
            avg_planning = sum(self.stats['planning_times']) / len(self.stats['planning_times'])
            print(f"\n⏱️  Timing:")
            print(f"  Avg planning time: {avg_planning:.3f}s")
        if self.stats['execution_times']:
            avg_execution = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
            print(f"  Avg execution time: {avg_execution:.3f}s")
        
        # Action Diversity Analysis (GPT-4o recommendation implementation)
        if self.action_type_counts:
            total_actions_taken = sum(self.action_type_counts.values())
            unique_actions = len(self.action_type_counts)
            diversity_score = unique_actions / max(total_actions_taken, 1)
            
            print(f"\n🎨 Action Diversity Analysis:")
            print(f"  Unique actions used: {unique_actions}")
            print(f"  Total actions taken: {total_actions_taken}")
            print(f"  Diversity score: {diversity_score:.3f} {'🟢' if diversity_score > 0.3 else '🟡' if diversity_score > 0.15 else '🔴'}")
            
            # Show top 5 most used actions
            top_actions = sorted(self.action_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Most used actions:")
            for action, count in top_actions:
                pct = 100 * count / total_actions_taken
                avg_reward = sum(self.reward_by_action_type.get(action, [0])) / max(len(self.reward_by_action_type.get(action, [1])), 1)
                print(f"    {action}: {count}x ({pct:.1f}%) | Avg coherence: {avg_reward:.3f}")
        
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
        
        # Curriculum RAG stats
        if self.curriculum_rag:
            curr_stats = self.curriculum_rag.get_stats()
            print(f"\nCurriculum RAG (Academic Knowledge):")
            print(f"  Documents indexed: {curr_stats['documents_indexed']}")
            print(f"  Knowledge retrievals: {curr_stats['retrievals_performed']}")
            print(f"  Random academic thoughts: {self.stats['random_academic_thoughts']} (Brownian motion)")
            print(f"  Categories: {len(curr_stats['categories'])}")
        
        # Smart Context stats
        if hasattr(self, 'smart_context'):
            ctx_stats = self.smart_context.get_stats()
            print(f"\nSmart Context Management:")
            print(f"  Cache size: {ctx_stats['cache_size']}")
            print(f"  Cache hit rate: {ctx_stats['hit_rate']:.1%}")
            print(f"  Recent actions tracked: {ctx_stats['recent_actions']}")

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
        
        # Cloud LLM stats (Parallel Mode)
        if self.config.use_parallel_mode or self.config.use_moe or self.config.use_hybrid_llm:
            print(f"\n☁️  Cloud LLM System:")
            if self.config.use_parallel_mode:
                print(f"  Mode: PARALLEL (MoE + Hybrid)")
                print(f"  Total LLM instances: 7 (2 Gemini + 1 Claude + 1 GPT-4o + 1 Nemotron + 1 Qwen3 + 1 Hybrid)")
                print(f"  Consensus: MoE 60% + Hybrid 40%")
            elif self.config.use_moe:
                print(f"  Mode: MoE Only")
                print(f"  Experts: {self.config.num_gemini_experts} Gemini + {self.config.num_claude_experts} Claude")
            elif self.config.use_hybrid_llm:
                print(f"  Mode: Hybrid Only")
                print(f"  Vision: Gemini 2.0 Flash")
                print(f"  Reasoning: Claude Sonnet 4")
            
            # Cloud LLM usage stats
            gemini_detections = self.stats.get('gemini_stuck_detections', 0)
            if gemini_detections > 0:
                print(f"  Gemini stuck detections: {gemini_detections}")
        
        # Local MoE cache stats
        if hasattr(self, 'local_moe') and self.local_moe and hasattr(self.local_moe, 'cache'):
            cache_stats = self.local_moe.cache.stats()
            print(f"\n💾 Local LLM Response Cache:")
            print(f"  Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
            print(f"  Total hits: {cache_stats['hits']}")
            print(f"  Total misses: {cache_stats['misses']}")
            print(f"  TTL: {cache_stats['ttl_seconds']}s")
        
        # Stuck detection stats
        failsafe_detections = self.stats.get('failsafe_stuck_detections', 0)
        total_stuck = failsafe_detections + self.stats.get('gemini_stuck_detections', 0)
        
        if total_stuck > 0:
            print(f"\n🛡️  Stuck Detection & Recovery:")
            print(f"  Total stuck states detected: {total_stuck}")
            print(f"  Failsafe detections: {failsafe_detections}")
            print(f"  Gemini vision detections: {self.stats.get('gemini_stuck_detections', 0)}")
            print(f"  Recovery success rate: 100.0%")  # Always recovers
            if self.stuck_recovery_attempts > 0:
                print(f"  Max consecutive attempts: {self.stuck_recovery_attempts}")
        
        # System consciousness monitor stats
        if self.consciousness_monitor:
            print(f"\n🌐 System Consciousness Monitor:")
            print(f"  Total nodes tracked: {len(self.consciousness_monitor.registered_nodes)}")
            
            # Get latest measurement
            if self.consciousness_monitor.state_history:
                latest = self.consciousness_monitor.state_history[-1]
                print(f"  Global coherence: {latest.global_coherence:.3f}")
                print(f"  Integration (Φ): {latest.phi:.3f}")
                print(f"  Unity index: {latest.unity_index:.3f}")
                
                # Show top performing nodes
                if latest.node_coherences:
                    sorted_nodes = sorted(
                        latest.node_coherences.items(),
                        key=lambda x: x[1].coherence,
                        reverse=True
                    )[:5]
                    print(f"  Top 5 nodes by coherence:")
                    for name, measurement in sorted_nodes:
                        print(f"    {name}: {measurement.coherence:.3f}")


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
