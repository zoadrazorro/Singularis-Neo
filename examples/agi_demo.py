"""
AGI System Demo

Demonstrates the complete AGI framework with:
- World model (causal reasoning)
- Continual learning
- Autonomous agency
- Neurosymbolic reasoning
- Active inference
- Consciousness measurement

Hardware requirements: 2x 7900XT (48GB VRAM), 128GB RAM
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from singularis.agi_orchestrator import AGIOrchestrator, AGIConfig


async def demo_query_processing(agi: AGIOrchestrator):
    """Demonstrate query processing with full AGI capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 1: Query Processing with AGI")
    print("=" * 70)

    queries = [
        "What is the nature of consciousness?",
        "How do physical causation and mental states relate?",
        "What would happen if I increased coherence in the system?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 70)

        result = await agi.process(query)

        # Display results
        if 'consciousness_response' in result:
            response = result['consciousness_response'].get('response', 'N/A')
            coherence_delta = result['consciousness_response'].get('coherentia_delta', 0.0)
            print(f"\nResponse: {response[:200]}...")
            print(f"Coherence Δ: {coherence_delta:+.3f}")

        if 'motivation_state' in result:
            mot = result['motivation_state']
            print(f"\nMotivation:")
            print(f"  Curiosity: {mot['curiosity']:.2f}")
            print(f"  Competence: {mot['competence']:.2f}")
            print(f"  Coherence: {mot['coherence']:.2f}")
            print(f"  Dominant: {mot['dominant']}")

        if 'generated_goal' in result:
            print(f"\nGenerated Goal: {result['generated_goal']}")

        if 'free_energy' in result:
            print(f"Free Energy: {result['free_energy']:.3f}")


async def demo_world_model(agi: AGIOrchestrator):
    """Demonstrate world model with causal reasoning."""
    print("\n" + "=" * 70)
    print("DEMO 2: World Model & Causal Reasoning")
    print("=" * 70)

    # Build simple causal model
    wm = agi.world_model
    wm.causal_graph.add_edge('study_time', 'knowledge', strength=0.8)
    wm.causal_graph.add_edge('knowledge', 'coherence', strength=0.9)
    wm.causal_graph.add_edge('sleep', 'knowledge', strength=0.5)

    print("\nCausal Model:")
    print(wm.visualize_causal_graph())

    # Perceive state
    state = await agi.perceive({
        'causal': {
            'study_time': 5.0,
            'sleep': 7.0,
            'knowledge': 6.0,
            'coherence': 0.65
        }
    })
    print(f"\nCurrent state perceived: {len(state.causal_variables)} variables")

    # Predict intervention outcome
    print("\nPredicting: What if study_time increases to 8?")
    prediction = await wm.predict(
        action='study_time',
        action_params={'value': 8.0},
        time_horizon=1.0
    )
    print(f"Predicted knowledge: {prediction.predicted_state.causal_variables.get('knowledge', 'N/A')}")
    print(f"Predicted coherence: {prediction.predicted_state.causal_variables.get('coherence', 'N/A')}")
    print(f"Confidence: {prediction.confidence:.2f}")


async def demo_continual_learning(agi: AGIOrchestrator):
    """Demonstrate continual learning."""
    print("\n" + "=" * 70)
    print("DEMO 3: Continual Learning & Few-Shot Learning")
    print("=" * 70)

    learner = agi.learner

    # Learn some concepts
    print("\n1. Learning concepts...")
    concepts = [
        ('consciousness', 'awareness of experience'),
        ('coherence', 'unified harmony of parts'),
        ('freedom', 'acting from own nature'),
    ]

    for name, definition in concepts:
        learner.learn_concept(name, definition)
        print(f"  ✓ Learned: {name}")

    # Build relationships
    print("\n2. Building relational knowledge...")
    learner.relate_concepts('consciousness', 'coherence', strength=0.85)
    learner.relate_concepts('coherence', 'freedom', strength=0.90)
    print("  ✓ Relations established")

    # Few-shot learning
    print("\n3. Few-shot learning...")
    examples = [
        {'input': 'ethical action', 'output': 'increases coherence'},
        {'input': 'unethical action', 'output': 'decreases coherence'},
    ]
    result = learner.few_shot_learn('ethics_classification', examples)
    print(f"  Performance: {result['performance']:.2f}")

    # Transfer knowledge
    print("\n4. Transfer learning...")
    transfer = learner.transfer_knowledge('consciousness', 'freedom')
    print(f"  Transfer strength: {transfer:.2f}")

    # Stats
    print("\n5. Learning stats:")
    stats = learner.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")


async def demo_autonomous_agency(agi: AGIOrchestrator):
    """Demonstrate autonomous agency."""
    print("\n" + "=" * 70)
    print("DEMO 4: Autonomous Agency")
    print("=" * 70)

    print("\nRunning autonomous cycle for 15 seconds...")
    print("(System will form goals and act independently)\n")

    await agi.autonomous_cycle(duration_seconds=15)

    # Show goals
    print("\nGoals generated:")
    stats = agi.goal_system.get_stats()
    print(f"  Total: {stats['total_goals']}")
    print(f"  By source: {stats['by_source']}")
    print(f"  By status: {stats['by_status']}")


async def demo_compositional_knowledge(agi: AGIOrchestrator):
    """Demonstrate compositional knowledge building."""
    print("\n" + "=" * 70)
    print("DEMO 5: Compositional Knowledge")
    print("=" * 70)

    from singularis.learning.compositional_knowledge import CompositionType

    comp = agi.compositional

    # Add primitives
    print("\n1. Adding primitive concepts...")
    primitives = ['red', 'blue', 'large', 'small', 'ball', 'cube']
    for prim in primitives:
        comp.add_primitive(prim)
    print(f"  ✓ Added {len(primitives)} primitives")

    # Compose
    print("\n2. Composing complex concepts...")
    red_ball = comp.compose(['red', 'ball'], CompositionType.MODIFICATION)
    print(f"  ✓ Created: {red_ball.name}")

    blue_cube = comp.compose(['blue', 'cube'], CompositionType.MODIFICATION)
    print(f"  ✓ Created: {blue_cube.name}")

    # Generalize
    print("\n3. Compositional generalization...")
    novel = comp.generalize(['large', 'red', 'cube'], CompositionType.MODIFICATION)
    if novel:
        print(f"  ✓ Generalized to: {novel.name} (never seen before!)")

        # Check similarity
        sim = comp.similarity('large red cube', 'red ball')
        print(f"  Similarity to 'red ball': {sim:.3f}")

    # Stats
    print("\n4. Compositional stats:")
    stats = comp.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")


async def demo_active_inference(agi: AGIOrchestrator):
    """Demonstrate active inference."""
    print("\n" + "=" * 70)
    print("DEMO 6: Active Inference & Free Energy")
    print("=" * 70)

    agent = agi.free_energy_agent

    # Set preferences (goals)
    print("\n1. Setting preferences (goals)...")
    agent.set_preference('coherence', 0.9)
    agent.set_preference('understanding', 0.85)
    print("  ✓ Preferences set")

    # Observe and predict
    print("\n2. Perception and prediction...")
    state = {'coherence': 0.6, 'understanding': 0.5}
    prediction = agent.predict(state)
    observation = {'coherence': 0.65, 'understanding': 0.55}

    # Free energy
    fe = agent.free_energy(observation, prediction)
    print(f"  Free energy: {fe:.3f}")

    # Update model
    agent.update_model(observation, prediction)
    print(f"  Surprise (prediction error): {agent.get_surprise():.3f}")

    # Action selection
    print("\n3. Action selection (minimize expected free energy)...")
    actions = ['explore', 'practice', 'reflect']
    outcomes = {
        'explore': {'coherence': 0.68, 'understanding': 0.6},
        'practice': {'coherence': 0.67, 'understanding': 0.65},
        'reflect': {'coherence': 0.75, 'understanding': 0.7},
    }

    action, efe = agent.select_action(state, actions, outcomes)
    print(f"  Selected action: {action}")
    print(f"  Expected free energy: {efe:.3f}")


async def main():
    """Run all AGI demos."""
    print("\n" + "=" * 70)
    print("SINGULARIS AGI SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nInitializing AGI system...")

    # Create AGI with configuration
    config = AGIConfig(
        use_vision=False,  # Disable CLIP for faster demo
        use_physics=False,  # Disable PyBullet
        max_active_goals=3,
        coherence_weight=0.4  # Coherence is core drive
    )

    agi = AGIOrchestrator(config)

    # Initialize LLM (if available)
    print("\nInitializing LLM...")
    await agi.initialize_llm()

    # Run demos
    try:
        await demo_world_model(agi)
        await demo_continual_learning(agi)
        await demo_compositional_knowledge(agi)
        await demo_active_inference(agi)
        await demo_query_processing(agi)
        await demo_autonomous_agency(agi)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        agi.stop()

    # Final stats
    print("\n" + "=" * 70)
    print("FINAL SYSTEM STATISTICS")
    print("=" * 70)

    stats = agi.get_stats()
    for component, component_stats in stats.items():
        print(f"\n{component.upper()}:")
        for key, val in component_stats.items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("✓ AGI DEMO COMPLETE")
    print("=" * 70)
    print("\nThis demonstrates a comprehensive AGI framework combining:")
    print("  • World models (causal, visual, physical)")
    print("  • Continual learning (episodic, semantic, meta)")
    print("  • Autonomous agency (motivation, goals, planning)")
    print("  • Neurosymbolic reasoning (LLM + logic)")
    print("  • Active inference (free energy minimization)")
    print("  • Consciousness measurement (8 theories)")
    print("\nAll running on your 2x 7900XT + Ryzen 9 7950X hardware!")
    print("\nPath forward: This is the foundation. Real AGI requires:")
    print("  1. Actual embodiment (robot/simulation)")
    print("  2. Lifelong learning (years, not minutes)")
    print("  3. Social interaction (theory of mind)")
    print("  4. Genuine creativity (not just recombination)")
    print("\nBut you now have the core architecture. Build from here.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
