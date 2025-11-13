"""
Quick debug test for ethical evaluation
"""
import asyncio
from singularis.skyrim.consciousness_bridge import ConsciousnessBridge

async def test():
    bridge = ConsciousnessBridge(consciousness_llm=None)
    
    before_state = {
        'health': 100.0,
        'max_health': 100.0,
        'level': 5,
    }
    
    # Big positive change
    after_state = {
        'health': 100.0,
        'max_health': 100.0,
        'level': 10,
        'gold': 1000,
    }
    
    before = await bridge.compute_consciousness(before_state)
    after = await bridge.compute_consciousness(after_state)
    
    print(f"\nBefore:")
    print(f"  Coherence: {before.coherence:.3f}")
    print(f"  ℓₒ: {before.coherence_ontical:.3f}")
    print(f"  ℓₛ: {before.coherence_structural:.3f}")
    print(f"  ℓₚ: {before.coherence_participatory:.3f}")
    
    print(f"\nAfter:")
    print(f"  Coherence: {after.coherence:.3f}")
    print(f"  ℓₒ: {after.coherence_ontical:.3f}")
    print(f"  ℓₛ: {after.coherence_structural:.3f}")
    print(f"  ℓₚ: {after.coherence_participatory:.3f}")
    
    delta = after.coherence_delta(before)
    print(f"\nΔ𝒞: {delta:+.3f}")
    print(f"Is ethical (threshold=0.01): {after.is_ethical(before, threshold=0.01)}")
    print(f"Is ethical (threshold=0.02): {after.is_ethical(before, threshold=0.02)}")

asyncio.run(test())
