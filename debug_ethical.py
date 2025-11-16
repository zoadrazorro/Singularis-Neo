"""
Quick debug test for ethical evaluation.

This script is designed to test the `ConsciousnessBridge`'s ability to compute
coherence and evaluate the ethical dimension of a state change, as defined by the
Singularis framework.

It works by:
1.  Initializing a `ConsciousnessBridge`.
2.  Defining two distinct game states: a `before_state` and an `after_state`
    that represents a significant positive change (e.g., leveling up, gaining gold).
3.  Asynchronously computing the consciousness state (including coherence and its
    Three Lumina components) for both the `before` and `after` states.
4.  Printing the coherence values for both states.
5.  Calculating the change in coherence (Δ𝒞), which is the core of the ethical evaluation.
6.  Printing whether the state change is considered "ethical" based on different
    positive coherence delta thresholds.

This provides a quick and isolated way to verify that the ethical evaluation
mechanism is functioning as expected.
"""
import asyncio
from singularis.skyrim.consciousness_bridge import ConsciousnessBridge

async def test():
    """Runs the test for ethical evaluation."""
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

if __name__ == "__main__":
    asyncio.run(test())
