"""
DATA System Integration Examples
=================================

Demonstrates how to integrate DATA with Singularis core systems:
1. Consciousness Layer
2. Life Operations
3. Skyrim AGI
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.integrations import (
    DATAConsciousnessBridge,
    DATALifeOpsBridge,
    DATASkyrimBridge
)
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from loguru import logger


async def example_consciousness_integration():
    """Example: DATA + Consciousness Layer"""
    print("\n" + "="*70)
    print("Example: DATA-Consciousness Integration")
    print("="*70 + "\n")
    
    # Initialize consciousness
    consciousness = UnifiedConsciousnessLayer()
    
    # Create DATA bridge
    bridge = DATAConsciousnessBridge(
        consciousness=consciousness,
        enable_data=True
    )
    
    await bridge.initialize()
    
    # Example 1: Standard processing (consciousness only)
    print("--- Standard Consciousness Processing ---\n")
    result1 = await bridge.process(
        query="What is the meaning of consciousness?",
        subsystem_inputs={},
        use_data_routing=False  # Don't use DATA
    )
    
    print(f"Routing: {result1['routing']}")
    print(f"Response: {result1['response'][:200]}...\n")
    
    # Example 2: DATA routing
    print("--- DATA Distributed Routing ---\n")
    result2 = await bridge.process(
        query="Analyze the technical implications of quantum computing for cryptography in detail",
        subsystem_inputs={},
        use_data_routing=True  # Use DATA for complex query
    )
    
    print(f"Routing: {result2['routing']}")
    if result2.get('experts_used'):
        print(f"Experts: {', '.join(result2['experts_used'])}")
    print(f"Response: {result2['response'][:200]}...\n")
    
    # Example 3: Hybrid mode
    print("--- Hybrid DATA + Consciousness ---\n")
    result3 = await bridge.process_hybrid(
        query="How can we integrate philosophical insights with technical analysis?",
        subsystem_inputs={}
    )
    
    print(f"Routing: {result3['routing']}")
    if result3.get('data_experts'):
        print(f"DATA Experts: {', '.join(result3['data_experts'])}")
    print(f"Response: {result3['response'][:200]}...\n")
    
    # Show stats
    stats = bridge.get_stats()
    print(f"Bridge Statistics:")
    print(f"  - Total queries: {stats['total_queries']}")
    print(f"  - DATA routed: {stats['data_routed']}")
    print(f"  - Consciousness routed: {stats['consciousness_routed']}")
    print(f"  - Hybrid queries: {stats['hybrid_queries']}")
    print(f"  - DATA usage: {stats['data_usage_percent']:.1f}%\n")
    
    await bridge.shutdown()


async def example_lifeops_integration():
    """Example: DATA + Life Operations"""
    print("\n" + "="*70)
    print("Example: DATA-LifeOps Integration")
    print("="*70 + "\n")
    
    # Create DATA bridge
    bridge = DATALifeOpsBridge()
    await bridge.initialize()
    
    if not bridge.is_data_ready:
        print("⚠ DATA not available, skipping LifeOps example")
        return
    
    # Example life events
    life_events = [
        {"type": "sleep", "timestamp": 1700000000, "duration": 6.5, "quality": "poor"},
        {"type": "sleep", "timestamp": 1700086400, "duration": 7.2, "quality": "fair"},
        {"type": "sleep", "timestamp": 1700172800, "duration": 8.1, "quality": "good"},
        {"type": "exercise", "timestamp": 1700100000, "activity": "running", "duration": 30},
        {"type": "work", "timestamp": 1700120000, "stress_level": "high", "duration": 480},
        {"type": "social", "timestamp": 1700150000, "activity": "dinner", "mood": "positive"},
    ]
    
    # Analyze patterns
    print("--- Pattern Analysis ---\n")
    result = await bridge.analyze_life_patterns(
        events=life_events,
        query="What patterns indicate stress, and how does sleep affect my productivity?"
    )
    
    if result.get("success"):
        print(f"Experts consulted: {', '.join(result['experts_consulted'])}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nAnalysis:\n{result['analysis'][:400]}...\n")
    
    # Health recommendations
    print("--- Health Recommendations ---\n")
    health_data = {
        "average_sleep": 7.2,
        "exercise_frequency": "3x/week",
        "stress_level": "moderate",
        "energy_level": "medium"
    }
    
    health_result = await bridge.get_health_recommendations(
        health_data=health_data,
        goals=["Improve sleep quality", "Reduce stress"]
    )
    
    if health_result.get("success"):
        print(f"Experts: {', '.join(health_result['experts_consulted'])}")
        print(f"\nRecommendations:\n{health_result['recommendations'][:400]}...\n")
    
    # Show stats
    stats = bridge.get_stats()
    print(f"Bridge Statistics:")
    print(f"  - Pattern analyses: {stats['pattern_analyses']}")
    print(f"  - Health queries: {stats['health_queries']}\n")
    
    await bridge.shutdown()


async def example_skyrim_integration():
    """Example: DATA + Skyrim AGI"""
    print("\n" + "="*70)
    print("Example: DATA-Skyrim Integration")
    print("="*70 + "\n")
    
    # Create DATA bridge
    bridge = DATASkyrimBridge()
    await bridge.initialize()
    
    if not bridge.is_data_ready:
        print("⚠ DATA not available, skipping Skyrim example")
        return
    
    # Example game state
    game_state = {
        "health": 75,
        "magicka": 120,
        "stamina": 80,
        "location": "Whiterun",
        "level": 15,
        "inventory_weight": "150/300"
    }
    
    available_actions = [
        "explore_dungeon",
        "talk_to_npc",
        "fast_travel",
        "craft_items",
        "rest",
        "continue_quest"
    ]
    
    # Plan action
    print("--- Action Planning ---\n")
    result = await bridge.plan_action(
        game_state=game_state,
        available_actions=available_actions,
        context={"current_quest": "Dragon Rising", "urgency": "medium"}
    )
    
    if result.get("success"):
        print(f"Experts consulted: {', '.join(result['experts_consulted'])}")
        print(f"Recommended action: {result['recommended_action']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nReasoning:\n{result['reasoning'][:400]}...\n")
    
    # Combat strategy
    print("--- Combat Strategy ---\n")
    combat_state = {
        "health": 50,
        "magicka": 80,
        "stamina": 40,
        "position": "open_field",
        "equipment": "sword_and_shield"
    }
    
    enemies = [
        {"name": "Bandit", "level": 12, "type": "humanoid", "weapon": "bow"},
        {"name": "Bandit Chief", "level": 15, "type": "humanoid", "weapon": "two_handed"}
    ]
    
    combat_result = await bridge.plan_combat_strategy(
        combat_state=combat_state,
        enemy_info=enemies
    )
    
    if combat_result.get("success"):
        print(f"Experts: {', '.join(combat_result['experts_consulted'])}")
        print(f"\nStrategy:\n{combat_result['strategy'][:400]}...\n")
    
    # Show stats
    stats = bridge.get_stats()
    print(f"Bridge Statistics:")
    print(f"  - Action decisions: {stats['action_decisions']}")
    print(f"  - Combat strategies: {stats['combat_strategies']}")
    print(f"  - Total decisions: {stats['total_decisions']}\n")
    
    await bridge.shutdown()


async def main():
    """Run all integration examples"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    try:
        await example_consciousness_integration()
        await example_lifeops_integration()
        await example_skyrim_integration()
        
        print("\n" + "="*70)
        print("All integration examples completed!")
        print("="*70 + "\n")
        
        print("Integration Summary:")
        print("  ✓ Consciousness Bridge - Hybrid DATA + GPT-5 reasoning")
        print("  ✓ LifeOps Bridge - Multi-expert pattern analysis")
        print("  ✓ Skyrim Bridge - Distributed action planning")
        print()
        print("Usage:")
        print("  - Bridges provide graceful degradation if DATA unavailable")
        print("  - Use bridges to enhance existing systems without modification")
        print("  - Choose routing strategy based on query complexity")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

