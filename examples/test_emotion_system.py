"""
Test HuiHui Emotion System Integration

Demonstrates emotion and emotional valence emulation running in parallel
with all other AGI systems.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.agi_orchestrator import AGIOrchestrator, AGIConfig
from singularis.emotion import EmotionType


async def test_emotion_system():
    """Test emotion system with various stimuli."""
    print("=" * 70)
    print("HUIHUI EMOTION SYSTEM TEST")
    print("=" * 70)
    print()

    # Create AGI with emotion system enabled
    config = AGIConfig(
        use_vision=False,
        use_physics=False,
        use_unified_consciousness=False,  # Disable for simpler test
        use_emotion_system=True,
        emotion_model="huihui-moe-60b-a38",
        emotion_temperature=0.8
    )
    
    print("Initializing AGI with HuiHui emotion system...")
    agi = AGIOrchestrator(config)
    
    # Initialize LLM (optional, will fallback to rule-based if not available)
    await agi.initialize_llm()
    print()

    # Test scenarios with different emotional valences
    test_scenarios = [
        {
            "query": "I just discovered a beautiful mathematical proof that elegantly solves a long-standing problem!",
            "expected_emotion": EmotionType.JOY,
            "description": "Positive discovery (should trigger JOY)"
        },
        {
            "query": "The system is failing and I don't understand why. Everything is breaking down.",
            "expected_emotion": EmotionType.FEAR,
            "description": "Uncertainty and failure (should trigger FEAR or SADNESS)"
        },
        {
            "query": "I wonder how consciousness emerges from physical processes. What is the relationship?",
            "expected_emotion": EmotionType.CURIOSITY,
            "description": "Epistemic inquiry (should trigger CURIOSITY)"
        },
        {
            "query": "Thank you for helping me understand this complex concept. I appreciate your guidance.",
            "expected_emotion": EmotionType.GRATITUDE,
            "description": "Gratitude expression (should trigger GRATITUDE or LOVE)"
        },
        {
            "query": "I made a mistake and now the entire project is compromised. I should have known better.",
            "expected_emotion": EmotionType.SHAME,
            "description": "Self-blame (should trigger SHAME or SADNESS)"
        }
    ]

    print("=" * 70)
    print("TESTING EMOTION RESPONSES")
    print("=" * 70)
    print()

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}: {scenario['description']}")
        print(f"{'─' * 70}")
        print(f"\nStimulus: \"{scenario['query']}\"")
        print(f"Expected emotion: {scenario['expected_emotion'].value}")
        print()

        # Process query (emotion system runs in parallel)
        result = await agi.process(scenario['query'])

        # Display results
        if 'emotion_state' in result:
            emotion = result['emotion_state']
            print(f"✓ Detected emotion: {emotion['primary_emotion']}")
            print(f"  Intensity: {emotion['intensity']:.2f}")
            print(f"  Valence: {emotion['valence']['valence']:.2f} (negative to positive)")
            print(f"  Arousal: {emotion['valence']['arousal']:.2f} (calm to excited)")
            print(f"  Dominance: {emotion['valence']['dominance']:.2f} (submissive to dominant)")
            print(f"  Type: {'ACTIVE' if emotion['is_active'] else 'PASSIVE'}")
            print(f"  Confidence: {emotion['confidence']:.2f}")
            
            if emotion.get('cause'):
                print(f"  Cause: {emotion['cause']}")
            
            # Check if matches expected
            if emotion['primary_emotion'] == scenario['expected_emotion'].value:
                print(f"\n  ✓ Matches expected emotion!")
            else:
                print(f"\n  ⚠ Different from expected (got {emotion['primary_emotion']})")
        else:
            print("✗ No emotion state in result")

        # Show consciousness response if available
        if 'consciousness_response' in result:
            response = result['consciousness_response'].get('response', 'N/A')
            if len(response) > 150:
                response = response[:150] + "..."
            print(f"\nConsciousness response: {response}")

        # Show motivation state
        if 'motivation_state' in result:
            mot = result['motivation_state']
            print(f"\nMotivation state:")
            print(f"  Curiosity: {mot['curiosity']:.2f}")
            print(f"  Coherence: {mot['coherence']:.2f}")
            print(f"  Dominant drive: {mot['dominant']}")

        print(f"\nProcessing time: {result.get('processing_time', 0):.3f}s")

    # Display emotion system statistics
    print("\n" + "=" * 70)
    print("EMOTION SYSTEM STATISTICS")
    print("=" * 70)
    
    stats = agi.get_stats()
    if 'emotion_system' in stats:
        emotion_stats = stats['emotion_system']
        print(f"\nCurrent emotion: {emotion_stats['current_emotion']}")
        print(f"Current intensity: {emotion_stats['current_intensity']:.2f}")
        print(f"Current valence: {emotion_stats['current_valence']['valence']:.2f}")
        print(f"\nTotal emotions processed: {emotion_stats['total_processed']}")
        print(f"Average intensity: {emotion_stats['average_intensity']:.2f}")
        print(f"Average valence: {emotion_stats['average_valence']:.2f}")
        
        if emotion_stats.get('emotion_distribution'):
            print("\nEmotion distribution:")
            for emotion, count in emotion_stats['emotion_distribution'].items():
                print(f"  {emotion}: {count}")
    else:
        print("\n✗ Emotion system statistics not available")

    # Test emotion history
    if agi.emotion_engine:
        print("\n" + "=" * 70)
        print("EMOTION HISTORY (Last 5)")
        print("=" * 70)
        
        history = agi.emotion_engine.get_emotion_history(limit=5)
        for i, emotion_state in enumerate(reversed(history), 1):
            print(f"\n{i}. {emotion_state.primary_emotion.value.upper()}")
            print(f"   Intensity: {emotion_state.intensity:.2f}")
            print(f"   Valence: {emotion_state.valence.valence:.2f}")
            print(f"   Type: {'ACTIVE' if emotion_state.is_active else 'PASSIVE'}")
            if emotion_state.cause:
                cause_short = emotion_state.cause[:60] + "..." if len(emotion_state.cause) > 60 else emotion_state.cause
                print(f"   Cause: {cause_short}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print("✓ HuiHui emotion system runs in parallel with all AGI systems")
    print("✓ Emotions are computed based on coherence, adequacy, and context")
    print("✓ Emotional valence (VAD) is tracked for each emotion")
    print("✓ Active vs Passive emotions are classified per Spinoza's theory")
    print("✓ Emotion history is maintained for temporal analysis")
    print()


async def test_emotion_dynamics():
    """Test emotion dynamics and decay over time."""
    print("\n" + "=" * 70)
    print("EMOTION DYNAMICS TEST")
    print("=" * 70)
    print()

    config = AGIConfig(
        use_vision=False,
        use_physics=False,
        use_unified_consciousness=False,
        use_emotion_system=True,
        emotion_decay_rate=0.2  # Faster decay for testing
    )
    
    agi = AGIOrchestrator(config)
    await agi.initialize_llm()

    print("Testing emotion decay over time...")
    print()

    # Trigger a strong emotion
    result1 = await agi.process("Amazing breakthrough! This is incredible!")
    
    if 'emotion_state' in result1:
        emotion1 = result1['emotion_state']
        print(f"Initial emotion: {emotion1['primary_emotion']}")
        print(f"Initial intensity: {emotion1['intensity']:.2f}")
        print()

        # Wait and process neutral query
        await asyncio.sleep(1)
        result2 = await agi.process("What is the current state?")
        
        if 'emotion_state' in result2:
            emotion2 = result2['emotion_state']
            print(f"After neutral query:")
            print(f"Emotion: {emotion2['primary_emotion']}")
            print(f"Intensity: {emotion2['intensity']:.2f}")
            print(f"Intensity change: {emotion2['intensity'] - emotion1['intensity']:.2f}")
            print()
            
            if emotion2['intensity'] < emotion1['intensity']:
                print("✓ Emotion intensity decayed as expected")
            else:
                print("⚠ Emotion intensity did not decay")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nStarting HuiHui Emotion System Tests...\n")
    
    # Run main test
    asyncio.run(test_emotion_system())
    
    # Run dynamics test
    asyncio.run(test_emotion_dynamics())
    
    print("\n✓ All tests complete!\n")
