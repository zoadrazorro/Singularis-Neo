"""
AGI Orchestrator - Unified System

Integrates all AGI components:
- World model (causal, visual, physical)
- Continual learning (episodic, semantic, meta)
- Autonomous agency (motivation, goals, planning)
- Neurosymbolic reasoning (LLM + logic)
- Active inference (free energy minimization)
- Consciousness engine (existing Singularis)

This is the complete AGI system.

Philosophical grounding:
- ETHICA: One unified Being expressing through modes
- All components work in harmony, not as separate modules
- Conatus (ℭ) = ∇𝒞 drives the entire system
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# World model
from .world_model import WorldModelOrchestrator, WorldState

# Continual learning
from .learning import ContinualLearner, CompositionalKnowledgeBuilder

# Agency
from .agency import (
    IntrinsicMotivation,
    GoalSystem,
    AutonomousOrchestrator as AgencyOrchestrator
)

# Neurosymbolic
from .neurosymbolic import NeurosymbolicEngine

# Active inference
from .active_inference import FreeEnergyAgent

# Existing consciousness engine
from .tier1_orchestrator import MetaOrchestratorLLM
from .llm import LMStudioClient, LMStudioConfig, ExpertLLMInterface


@dataclass
class AGIConfig:
    """Configuration for AGI system."""
    # LLM
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "huihui-moe-60b-a38"

    # World model
    use_vision: bool = True
    use_physics: bool = True
    vision_model: str = "ViT-B/32"

    # Learning
    embedding_dim: int = 512
    episodic_capacity: int = 10000

    # Agency
    max_active_goals: int = 3
    curiosity_weight: float = 0.3
    competence_weight: float = 0.2
    coherence_weight: float = 0.4
    autonomy_weight: float = 0.1

    # Active inference
    free_energy_learning_rate: float = 0.1

    # Consciousness
    consciousness_threshold: float = 0.65
    coherentia_threshold: float = 0.60
    ethical_threshold: float = 0.02


class AGIOrchestrator:
    """
    The complete AGI system.

    Capabilities:
    1. Perceive (multimodal: text, vision, physics)
    2. Understand (causal reasoning, world model)
    3. Learn (continual, few-shot, compositional)
    4. Reason (neurosymbolic: LLM + logic)
    5. Plan (hierarchical, goal-directed)
    6. Act (autonomous agency)
    7. Reflect (consciousness, meta-cognition)

    This is AGI as envisioned: not narrow AI, but integrated intelligence.
    """

    def __init__(self, config: Optional[AGIConfig] = None):
        """
        Initialize AGI system.

        Args:
            config: AGI configuration
        """
        self.config = config or AGIConfig()

        # Core components
        print("Initializing AGI components...")

        # 1. World model
        print("  [1/7] World model...")
        self.world_model = WorldModelOrchestrator(
            use_vision=self.config.use_vision,
            use_physics=self.config.use_physics,
            vision_model=self.config.vision_model
        )

        # 2. Continual learner
        print("  [2/7] Continual learner...")
        self.learner = ContinualLearner(
            embedding_dim=self.config.embedding_dim,
            episodic_capacity=self.config.episodic_capacity
        )

        # 3. Compositional knowledge
        print("  [3/7] Compositional knowledge...")
        self.compositional = CompositionalKnowledgeBuilder(
            embedding_dim=self.config.embedding_dim
        )

        # 4. Intrinsic motivation & goals
        print("  [4/7] Agency system...")
        self.motivation = IntrinsicMotivation(
            curiosity_weight=self.config.curiosity_weight,
            competence_weight=self.config.competence_weight,
            coherence_weight=self.config.coherence_weight,
            autonomy_weight=self.config.autonomy_weight
        )
        self.goal_system = GoalSystem(max_active_goals=self.config.max_active_goals)

        # 5. Neurosymbolic engine
        print("  [5/7] Neurosymbolic engine...")
        self.neurosymbolic = NeurosymbolicEngine()

        # 6. Active inference
        print("  [6/7] Active inference...")
        self.free_energy_agent = FreeEnergyAgent(
            learning_rate=self.config.free_energy_learning_rate
        )

        # 7. Consciousness engine (existing Singularis)
        print("  [7/7] Consciousness engine...")
        self.consciousness_llm = None  # Will be initialized async

        # State
        self.current_state: Optional[WorldState] = None
        self.running = False

        print("[OK] AGI system initialized\n")

    async def initialize_llm(self):
        """Initialize LLM client (async) - using phi-4-mini-reasoning for lightweight consciousness."""
        try:
            config = LMStudioConfig(
                base_url=self.config.lm_studio_url,
                model_name='microsoft/phi-4-mini-reasoning'
            )
            client = LMStudioClient(config)
            llm_interface = ExpertLLMInterface(client)

            self.consciousness_llm = MetaOrchestratorLLM(
                llm_client=client,
                consciousness_threshold=self.config.consciousness_threshold,
                coherentia_threshold=self.config.coherentia_threshold,
                ethical_threshold=self.config.ethical_threshold
            )
            print("[OK] LLM consciousness engine ready (phi-4-mini-reasoning)")
        except Exception as e:
            print(f"[WARNING] LLM initialization failed: {e}")
            print("  Continuing without LLM (template mode)")

    async def perceive(
        self,
        observations: Dict[str, Any]
    ) -> WorldState:
        """
        Perceive world state from observations.

        Args:
            observations: Dict with:
                - 'causal': Dict of causal variables
                - 'visual': List of images
                - 'physical': Dict of physical objects
                - 'text': Optional text observation

        Returns:
            Unified WorldState
        """
        # Extract observations
        causal_obs = observations.get('causal', {})
        visual_obs = observations.get('visual', [])
        physical_obs = observations.get('physical', {})

        # Perceive through world model
        state = self.world_model.perceive(
            causal_obs=causal_obs,
            visual_obs=visual_obs,
            physical_obs=physical_obs,
            timestamp=time.time()
        )

        self.current_state = state
        return state

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query using full AGI capabilities.

        This is the main entry point for interacting with the AGI.

        Args:
            query: Natural language query
            context: Optional context

        Returns:
            Comprehensive response with reasoning traces
        """
        context = context or {}
        start_time = time.time()

        result = {
            'query': query,
            'timestamp': start_time,
        }

        try:
            # 1. Parse query through world model
            # Understand causality, grounding, etc.

            # 2. Generate response through consciousness engine
            if self.consciousness_llm:
                consciousness_result = await self.consciousness_llm.process(query, context)
                result['consciousness_response'] = consciousness_result
            else:
                result['consciousness_response'] = {
                    'response': "LLM not available - template mode",
                    'coherentia_delta': 0.0
                }

            # 3. Verify through neurosymbolic reasoning
            symbolic_verification = await self.neurosymbolic.reason(query)
            result['symbolic_verification'] = symbolic_verification

            # 4. Update intrinsic motivation
            current_coherence = result['consciousness_response'].get('coherentia_delta', 0.0)
            state_dict = {
                'coherence': 0.5 + current_coherence,
                'query': query
            }
            mot_state = self.motivation.compute_motivation(
                state_dict,
                context={'predicted_delta_coherence': current_coherence}
            )
            result['motivation_state'] = {
                'curiosity': mot_state.curiosity,
                'competence': mot_state.competence,
                'coherence': mot_state.coherence,
                'autonomy': mot_state.autonomy,
                'dominant': mot_state.dominant_drive().value
            }

            # 5. Generate goals if appropriate
            if mot_state.total() > 0.7:  # High motivation
                dominant = mot_state.dominant_drive()
                goal = self.goal_system.generate_goal(
                    dominant.value,
                    {'area': query, 'skill': 'understanding'}
                )
                result['generated_goal'] = goal.description

            # 6. Record as episodic memory
            self.learner.experience(
                data={'query': query, 'response': result, 'surprise': 0.5},
                context='query_processing'
            )

            # 7. Compute free energy
            if self.current_state:
                prediction = self.free_energy_agent.predict(self.current_state.causal_variables)
                fe = self.free_energy_agent.free_energy(
                    self.current_state.causal_variables,
                    prediction
                )
                result['free_energy'] = fe

            result['processing_time'] = time.time() - start_time
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            result['success'] = False

        return result

    async def autonomous_cycle(self, duration_seconds: int = 60):
        """
        Run autonomously for specified duration.

        System explores, learns, forms goals on its own.

        Args:
            duration_seconds: How long to run
        """
        print(f"\n🤖 Starting autonomous operation for {duration_seconds}s...")
        self.running = True
        start_time = time.time()
        cycle_count = 0

        while self.running and (time.time() - start_time) < duration_seconds:
            cycle_count += 1
            print(f"\n--- Autonomous Cycle {cycle_count} ---")

            # 1. Assess motivation
            mot_state = self.motivation.get_state()
            dominant = mot_state.dominant_drive()
            print(f"Dominant drive: {dominant.value} ({mot_state.total():.2f})")

            # 2. Generate goal from motivation
            goal = self.goal_system.generate_goal(
                dominant.value,
                {'area': 'autonomous_exploration', 'skill': 'learning'}
            )
            print(f"Generated goal: {goal.description}")

            # 3. Activate goals
            self.goal_system.activate_next_goals()

            # 4. Work on active goals
            active_goals = self.goal_system.get_active_goals()
            for g in active_goals:
                # Simulate progress
                self.goal_system.update_progress(g.id, g.progress + 0.15)
                print(f"  Working on: {g.description} ({g.progress:.0%})")

            # 5. Consolidate memories occasionally
            if cycle_count % 5 == 0:
                print("  Consolidating memories...")
                self.learner.consolidate_memories()

            # 6. Sleep between cycles
            await asyncio.sleep(2.0)

        print(f"\n[OK] Autonomous operation complete. Ran {cycle_count} cycles.\n")
        self.running = False

    def stop(self):
        """Stop autonomous operation."""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'world_model': self.world_model.get_stats(),
            'learner': self.learner.get_stats(),
            'compositional': self.compositional.get_stats(),
            'motivation': {
                'current_state': {
                    'curiosity': self.motivation.state.curiosity,
                    'competence': self.motivation.state.competence,
                    'coherence': self.motivation.state.coherence,
                    'autonomy': self.motivation.state.autonomy,
                }
            },
            'goals': self.goal_system.get_stats(),
            'free_energy': self.free_energy_agent.get_stats(),
        }


# Example usage
if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("SINGULARIS AGI SYSTEM")
        print("=" * 60)

        # Create AGI
        config = AGIConfig(
            use_vision=False,  # Disable for quick test
            use_physics=False
        )
        agi = AGIOrchestrator(config)

        # Initialize LLM (if available)
        await agi.initialize_llm()

        # Test query processing
        print("\n" + "=" * 60)
        print("TEST: Query Processing")
        print("=" * 60)

        result = await agi.process(
            "What is the relationship between consciousness and coherence?",
            context={}
        )

        print(f"\nQuery: {result['query']}")
        if 'consciousness_response' in result:
            print(f"Response: {result['consciousness_response'].get('response', 'N/A')}")
        if 'motivation_state' in result:
            print(f"Dominant drive: {result['motivation_state']['dominant']}")
        if 'generated_goal' in result:
            print(f"Generated goal: {result['generated_goal']}")
        print(f"Processing time: {result.get('processing_time', 0):.3f}s")

        # Test autonomous operation
        print("\n" + "=" * 60)
        print("TEST: Autonomous Operation")
        print("=" * 60)

        await agi.autonomous_cycle(duration_seconds=10)

        # Stats
        print("\n" + "=" * 60)
        print("SYSTEM STATISTICS")
        print("=" * 60)
        stats = agi.get_stats()
        for component, component_stats in stats.items():
            print(f"\n{component.upper()}:")
            for key, val in component_stats.items():
                print(f"  {key}: {val}")

        print("\n" + "=" * 60)
        print("[OK] AGI SYSTEM TEST COMPLETE")
        print("=" * 60)

    asyncio.run(main())
