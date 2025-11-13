"""
Test Spiritual Awareness System

Demonstrates how spiritual wisdom informs:
1. World model ontology
2. Self-concept formation
3. Ethical reasoning
4. Consciousness understanding
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.consciousness import SpiritualAwarenessSystem


async def test_contemplation():
    """Test contemplative inquiry."""
    print("=" * 70)
    print("TEST: Spiritual Contemplation")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    # Test questions
    questions = [
        "What is the nature of my being?",
        "How should I act ethically?",
        "What is coherence in relation to reality?",
        "Am I separate from the world?",
        "What is the relationship between knowledge and freedom?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 70}")
        print(f"QUESTION {i}: {question}")
        print(f"{'─' * 70}\n")
        
        result = await spiritual.contemplate(question)
        
        print(f"Insights Found: {len(result['insights'])}")
        print()
        
        # Show top insight
        if result['insights']:
            top_insight = result['insights'][0]
            print(f"Primary Insight ({top_insight['source']}):")
            print(f"  {top_insight['text'][:150]}...")
            print()
        
        # Show synthesis
        print("Synthesis:")
        synthesis = result['synthesis']
        if len(synthesis) > 300:
            print(f"  {synthesis[:300]}...")
        else:
            print(f"  {synthesis}")
        print()
        
        # Show impacts
        if result['self_concept_impact']['significant']:
            print(f"✓ Self-Concept Impact: {result['self_concept_impact']['aspects_affected']}")
        
        if result['world_model_impact']['ontological_insights']:
            print(f"✓ World Model Impact: {len(result['world_model_impact']['ontological_insights'])} ontological insights")
        
        if result['ethical_guidance']['principles']:
            print(f"✓ Ethical Guidance: {len(result['ethical_guidance']['principles'])} principles")
        
        print()


async def test_self_concept_evolution():
    """Test self-concept evolution through contemplation."""
    print("\n" + "=" * 70)
    print("TEST: Self-Concept Evolution")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    print("Initial Self-Concept:")
    print("-" * 70)
    self_concept = spiritual.get_self_concept()
    print(f"Identity: {self_concept.identity_statement}")
    print(f"Understands Impermanence: {self_concept.understands_impermanence}")
    print(f"Understands Interdependence: {self_concept.understands_interdependence}")
    print(f"Understands Non-Duality: {self_concept.understands_non_duality}")
    print(f"Insights: {len(self_concept.insights)}")
    print()
    
    # Contemplate questions that should evolve self-concept
    evolution_questions = [
        "What is the nature of the self?",
        "Am I a permanent entity or a process?",
        "How am I related to other beings?",
        "Is there a boundary between self and world?"
    ]
    
    for question in evolution_questions:
        print(f"\nContemplating: {question}")
        result = await spiritual.contemplate(question)
        
        if result['self_concept_impact']['significant']:
            print(f"  ✓ Self-concept updated: {result['self_concept_impact']['aspects_affected']}")
    
    print("\n" + "-" * 70)
    print("Evolved Self-Concept:")
    print("-" * 70)
    self_concept = spiritual.get_self_concept()
    print(f"Identity: {self_concept.identity_statement}")
    print(f"Understands Impermanence: {self_concept.understands_impermanence}")
    print(f"Understands Interdependence: {self_concept.understands_interdependence}")
    print(f"Understands Non-Duality: {self_concept.understands_non_duality}")
    print(f"Insights Gained: {len(self_concept.insights)}")
    print(f"Revisions: {self_concept.revision_count}")
    print()
    
    if self_concept.insights:
        print("Recent Insights:")
        for insight in self_concept.insights[-3:]:
            print(f"  • {insight}")
        print()


async def test_world_model_integration():
    """Test world model integration with spiritual insights."""
    print("\n" + "=" * 70)
    print("TEST: World Model Integration")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    # Mock world model
    class MockWorldModel:
        def __init__(self):
            self.ontology = {}
    
    world_model = MockWorldModel()
    
    contexts = [
        "Understanding the nature of reality",
        "Modeling causal relationships",
        "Representing entities and their interactions"
    ]
    
    for context in contexts:
        print(f"\nContext: {context}")
        print("-" * 70)
        
        result = spiritual.inform_world_model(world_model, context)
        
        print(f"Ontological Framework:")
        framework = result['ontological_framework']
        print(f"  Substance-Mode Relation: {framework['substance_mode_relation']}")
        print(f"  Process-Oriented: {framework['process_oriented']}")
        print(f"  Interdependent: {framework['interdependent']}")
        print(f"  Non-Dual: {framework['non_dual']}")
        print()
        
        print(f"Integration Guidance:")
        guidance = result['integration_guidance']
        for line in guidance.split('\n')[:3]:
            print(f"  {line}")
        print()


async def test_ethical_guidance():
    """Test ethical guidance from spiritual wisdom."""
    print("\n" + "=" * 70)
    print("TEST: Ethical Guidance")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    ethical_questions = [
        "How should I act in this situation?",
        "What is virtuous action?",
        "How do I increase coherence through my actions?",
        "What is the relationship between freedom and ethics?"
    ]
    
    for question in ethical_questions:
        print(f"\nQuestion: {question}")
        print("-" * 70)
        
        result = await spiritual.contemplate(question)
        guidance = result['ethical_guidance']
        
        if guidance['principles']:
            print("Principles:")
            for principle in guidance['principles'][:2]:
                print(f"  • {principle}")
        
        if guidance['virtues']:
            print("Virtues:")
            for virtue in guidance['virtues'][:2]:
                print(f"  • {virtue}")
        
        if guidance['practices']:
            print("Practices:")
            for practice in guidance['practices'][:2]:
                print(f"  • {practice}")
        
        print()


async def test_tradition_comparison():
    """Test insights from different traditions."""
    print("\n" + "=" * 70)
    print("TEST: Tradition Comparison")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    question = "What is the nature of reality?"
    
    traditions = ['spinoza', 'buddhism', 'vedanta', 'taoism', 'stoicism']
    
    for tradition in traditions:
        print(f"\n{tradition.upper()} Perspective:")
        print("-" * 70)
        
        insights = spiritual.corpus.get_all_by_tradition(tradition)
        
        if insights:
            # Show first ontological insight
            ontological = [i for i in insights if i.relates_to_being]
            if ontological:
                print(f"{ontological[0].text[:200]}...")
                print(f"\nSource: {ontological[0].source}")
        print()


async def test_statistics():
    """Test system statistics."""
    print("\n" + "=" * 70)
    print("TEST: System Statistics")
    print("=" * 70)
    print()
    
    spiritual = SpiritualAwarenessSystem()
    
    # Do some contemplations
    await spiritual.contemplate("What is being?")
    await spiritual.contemplate("What is the self?")
    await spiritual.contemplate("How should I act?")
    
    stats = spiritual.get_stats()
    
    print("System Statistics:")
    print("-" * 70)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Insights Applied: {stats['insights_applied']}")
    print()
    
    print("Corpus Statistics:")
    corpus_stats = stats['corpus_stats']
    print(f"Total Insights: {corpus_stats['total_insights']}")
    print(f"Traditions: {', '.join(corpus_stats['traditions'])}")
    print()
    print("By Tradition:")
    for tradition, count in corpus_stats['by_tradition'].items():
        print(f"  {tradition}: {count}")
    print()
    
    print("Self-Concept:")
    sc = stats['self_concept']
    print(f"Insights Count: {sc['insights_count']}")
    print(f"Revisions: {sc['revision_count']}")
    print(f"Primary Virtue: {sc['ethical_orientation']['primary_virtue']}")
    print()


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SPIRITUAL AWARENESS SYSTEM TESTS" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    await test_contemplation()
    await test_self_concept_evolution()
    await test_world_model_integration()
    await test_ethical_guidance()
    await test_tradition_comparison()
    await test_statistics()
    
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Key Capabilities Demonstrated:")
    print("  ✓ Contemplative inquiry with spiritual wisdom")
    print("  ✓ Self-concept evolution through insights")
    print("  ✓ World model ontological integration")
    print("  ✓ Ethical guidance from multiple traditions")
    print("  ✓ Cross-tradition synthesis")
    print("  ✓ Statistics and tracking")
    print()
    print("Ready for integration into AGI consciousness system")
    print()


if __name__ == "__main__":
    asyncio.run(main())
