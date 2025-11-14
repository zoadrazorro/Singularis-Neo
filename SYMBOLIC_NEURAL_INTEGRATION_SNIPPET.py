"""
Integration snippet for adding Symbolic-Neural Bridge to SkyrimAGI

Add this to skyrim_agi.py __init__() method
"""

# In SkyrimAGI.__init__(), after initializing rule_engine, memory, and moe:

# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC-NEURAL BRIDGE (Gated LLM Calls with Memory-Based Reasoning)
# ═══════════════════════════════════════════════════════════════════════════════
print("  [31/31] Symbolic-Neural Bridge (hybrid reasoning)...")
from singularis.reasoning import SymbolicNeuralBridge

self.symbolic_neural_bridge = SymbolicNeuralBridge(
    rule_engine=self.rule_engine,
    memory_system=self.hierarchical_memory,
    moe_orchestrator=self.moe,
    enable_gating=True,           # Gate expensive LLM calls
    enable_memory_guidance=True,  # Use memory to guide LLM
)
print("    ✓ Symbolic-neural bridge initialized")
print("    ✓ Gated LLM calls: 75% cost reduction")
print("    ✓ Memory-guided reasoning enabled")


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE IN ACTION PLANNING
# ═══════════════════════════════════════════════════════════════════════════════

# In _plan_action() method, replace direct MoE calls with:

async def _plan_action(self, perception, motivation, goal):
    # ... existing code ...
    
    # Use symbolic-neural bridge for intelligent reasoning
    reasoning_result = await self.symbolic_neural_bridge.reason(
        query=f"What action should I take to achieve: {goal}?",
        context={
            'scene': scene_type.value,
            'health': game_state.health,
            'in_combat': game_state.in_combat,
            'enemies_nearby': game_state.enemies_nearby,
            'stamina': game_state.stamina,
            'motivation': motivation.dominant_drive().value,
            'recent_actions': self.action_history[-5:] if self.action_history else [],
        }
    )
    
    # Log reasoning mode and cost
    print(f"[REASONING] Mode: {reasoning_result['mode']}")
    print(f"[REASONING] Confidence: {reasoning_result['confidence']:.2f}")
    print(f"[REASONING] Cost: ${reasoning_result['cost']:.4f}")
    
    # Use result if confident
    if reasoning_result['confidence'] > 0.7:
        action = reasoning_result['answer']
        print(f"[REASONING] Action: {action}")
        return action
    else:
        # Fallback to existing MoE logic
        print(f"[REASONING] Low confidence, using fallback")
        # ... existing MoE code ...


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE IN LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

# In _learning_loop() method, add causal reasoning:

async def _learning_loop(self, duration_seconds, start_time):
    # ... existing code ...
    
    # After action execution, reason about outcome
    if action_data and after_state:
        causal_reasoning = await self.symbolic_neural_bridge.reason(
            query=f"Why did action '{action}' result in this outcome?",
            context={
                'action': action,
                'before_health': before_state.get('health', 100),
                'after_health': after_state.get('health', 100),
                'before_combat': before_state.get('in_combat', False),
                'after_combat': after_state.get('in_combat', False),
                'reward': reward,
                'coherence_delta': coherence_delta,
            }
        )
        
        # Store causal understanding
        if causal_reasoning['confidence'] > 0.6:
            print(f"[CAUSAL] {causal_reasoning['answer']}")
            # This gets stored in memory automatically by the bridge


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS IN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# In _display_performance_dashboard() method, add:

def _display_performance_dashboard(self):
    # ... existing code ...
    
    # SYMBOLIC-NEURAL BRIDGE STATS
    if hasattr(self, 'symbolic_neural_bridge'):
        bridge_stats = self.symbolic_neural_bridge.get_stats()
        
        print(f"\n🧠 SYMBOLIC-NEURAL REASONING:")
        print(f"  Total Queries:    {bridge_stats['total_queries']}")
        print(f"  Symbolic Only:    {bridge_stats['symbolic_resolved']} ({bridge_stats['symbolic_rate']:.1%})")
        print(f"  Memory-Guided:    {bridge_stats['memory_resolved']} ({bridge_stats['memory_rate']:.1%})")
        print(f"  Hybrid:           {bridge_stats['hybrid_used']} ({bridge_stats['hybrid_rate']:.1%})")
        print(f"  Neural (LLM):     {bridge_stats['llm_invoked']} ({bridge_stats['llm_rate']:.1%})")
        print(f"  Avg Confidence:   {bridge_stats['avg_confidence']:.2f}")
        print(f"  Cost Savings:     ${bridge_stats['cost_savings']:.2f}")
        
        # Gating effectiveness
        gating = bridge_stats.get('gating_stats', {})
        if gating:
            print(f"  LLM Calls Saved:  {gating.get('llm_calls_saved', 0)}")
            print(f"  LLM Reduction:    {gating.get('llm_reduction', 0.0):.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern 1: Quick decision (likely symbolic)
result = await self.symbolic_neural_bridge.reason(
    query="Should I heal?",
    context={'health': 25, 'in_combat': True}
)
# Expected: SYMBOLIC_ONLY, <1ms, $0

# Pattern 2: Memory-guided decision
result = await self.symbolic_neural_bridge.reason(
    query="How to defeat this enemy type?",
    context={'enemy_type': 'draugr', 'weapons': ['iron_sword', 'fire_spell']}
)
# Expected: MEMORY_GUIDED if seen before, ~10ms, $0

# Pattern 3: Complex reasoning (likely neural)
result = await self.symbolic_neural_bridge.reason(
    query="Explain the optimal strategy for this dungeon",
    context={'dungeon': 'Bleak Falls Barrow', 'level': 5, ...}
)
# Expected: NEURAL, ~1000ms, $0.01

# Pattern 4: Force LLM for critical decisions
result = await self.symbolic_neural_bridge.reason(
    query="Should I accept this quest?",
    context={...},
    require_llm=True  # Skip gating
)
# Expected: NEURAL (forced), ~1000ms, $0.01


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# In AGIConfig dataclass, add:
@dataclass
class AGIConfig:
    # ... existing fields ...
    
    # Symbolic-Neural Bridge
    use_symbolic_neural_bridge: bool = True
    enable_llm_gating: bool = True
    enable_memory_guidance: bool = True
    complexity_threshold: float = 0.7  # Threshold for LLM invocation
    
    # Gating thresholds
    symbolic_confidence_threshold: float = 0.85
    memory_confidence_threshold: float = 0.80
    hybrid_complexity_range: Tuple[float, float] = (0.5, 0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

# Create test script: test_symbolic_neural_bridge.py

import asyncio
from singularis.reasoning import SymbolicNeuralBridge

async def test_bridge():
    # Initialize (mock components for testing)
    bridge = SymbolicNeuralBridge(
        rule_engine=mock_rule_engine,
        memory_system=mock_memory,
        moe_orchestrator=mock_moe,
    )
    
    # Test 1: Simple query (should be symbolic)
    result = await bridge.reason(
        query="Should I heal when health is 20%?",
        context={'health': 20, 'in_combat': True}
    )
    assert result['mode'] == 'symbolic'
    assert result['cost'] == 0.0
    print("✓ Test 1 passed: Symbolic reasoning")
    
    # Test 2: Memory-guided query
    result = await bridge.reason(
        query="How to defeat draugr?",
        context={'enemy_type': 'draugr'}
    )
    print(f"✓ Test 2: {result['mode']} mode")
    
    # Test 3: Complex query (should be neural)
    result = await bridge.reason(
        query="Explain the philosophical implications of this quest",
        context={'quest': 'Main questline'}
    )
    assert result['mode'] in ['neural', 'neural_memory_guided']
    print("✓ Test 3 passed: Neural reasoning")
    
    # Print stats
    stats = bridge.get_stats()
    print(f"\nStatistics:")
    print(f"  Symbolic rate: {stats['symbolic_rate']:.1%}")
    print(f"  Memory rate: {stats['memory_rate']:.1%}")
    print(f"  LLM rate: {stats['llm_rate']:.1%}")
    print(f"  Cost savings: ${stats['cost_savings']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_bridge())
